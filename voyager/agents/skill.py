"""Enhanced Skill Manager for Voyager Evolved (Linux Optimized).

Features:
- Skill difficulty ratings
- Skill prerequisites
- Success rate tracking
- Skill composition (combine simple skills)
- Skill versioning (improve skills over time)
- Better indexing for faster retrieval
"""

import os
import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

import voyager.utils as U
from langchain.schema import HumanMessage, SystemMessage
from langchain.vectorstores import Chroma

from voyager.prompts import load_prompt
from voyager.control_primitives import load_control_primitives
from voyager.llm import get_llm, get_embeddings


@dataclass
class SkillMetadata:
    """Metadata for a skill including difficulty and prerequisites."""
    name: str
    description: str
    difficulty: float = 0.5  # 0 (trivial) to 1 (expert)
    prerequisites: List[str] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    avg_execution_time: float = 0.0
    version: int = 1
    created_at: float = field(default_factory=time.time)
    last_used: float = 0.0
    last_modified: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)
    composed_from: List[str] = field(default_factory=list)  # Skills this is composed from
    
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5  # Neutral prior
        return self.success_count / total
    
    def record_execution(self, success: bool, duration: float):
        """Record a skill execution."""
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        
        # Update average execution time
        total = self.success_count + self.failure_count
        self.avg_execution_time = (
            (self.avg_execution_time * (total - 1) + duration) / total
        )
        self.last_used = time.time()
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SkillMetadata':
        return cls(**data)


@dataclass
class SkillComposition:
    """Represents a composition of multiple skills."""
    name: str
    component_skills: List[str]
    sequence: List[str]  # Order of execution
    total_difficulty: float = 0.0
    success_rate: float = 0.5
    
    def calculate_difficulty(self, skill_metadata: Dict[str, SkillMetadata]):
        """Calculate combined difficulty."""
        if not self.component_skills:
            return
        
        difficulties = []
        for skill_name in self.component_skills:
            if skill_name in skill_metadata:
                difficulties.append(skill_metadata[skill_name].difficulty)
        
        if difficulties:
            # Combined difficulty is higher than individual
            avg_diff = sum(difficulties) / len(difficulties)
            max_diff = max(difficulties)
            self.total_difficulty = min(1.0, avg_diff * 0.7 + max_diff * 0.3 + len(difficulties) * 0.05)


class SkillIndex:
    """Efficient skill indexing for faster retrieval."""
    
    def __init__(self):
        self.by_tag: Dict[str, Set[str]] = defaultdict(set)
        self.by_difficulty: Dict[str, Set[str]] = defaultdict(set)  # "easy", "medium", "hard"
        self.by_prerequisite: Dict[str, Set[str]] = defaultdict(set)
        self.recently_used: List[str] = []
        self.most_successful: List[Tuple[str, float]] = []
    
    def add_skill(self, name: str, metadata: SkillMetadata):
        """Add a skill to the index."""
        # Index by tags
        for tag in metadata.tags:
            self.by_tag[tag].add(name)
        
        # Index by difficulty
        if metadata.difficulty < 0.33:
            self.by_difficulty["easy"].add(name)
        elif metadata.difficulty < 0.66:
            self.by_difficulty["medium"].add(name)
        else:
            self.by_difficulty["hard"].add(name)
        
        # Index by what this skill enables
        for prereq in metadata.prerequisites:
            self.by_prerequisite[prereq].add(name)
    
    def remove_skill(self, name: str, metadata: SkillMetadata):
        """Remove a skill from the index."""
        for tag in metadata.tags:
            self.by_tag[tag].discard(name)
        
        for difficulty in ["easy", "medium", "hard"]:
            self.by_difficulty[difficulty].discard(name)
        
        for prereq in metadata.prerequisites:
            self.by_prerequisite[prereq].discard(name)
    
    def update_recently_used(self, name: str):
        """Update recently used list."""
        if name in self.recently_used:
            self.recently_used.remove(name)
        self.recently_used.insert(0, name)
        self.recently_used = self.recently_used[:20]  # Keep last 20
    
    def update_success_ranking(self, name: str, success_rate: float):
        """Update success ranking."""
        # Remove old entry if exists
        self.most_successful = [(n, r) for n, r in self.most_successful if n != name]
        self.most_successful.append((name, success_rate))
        self.most_successful.sort(key=lambda x: x[1], reverse=True)
        self.most_successful = self.most_successful[:50]  # Keep top 50
    
    def get_by_tag(self, tag: str) -> Set[str]:
        return self.by_tag.get(tag, set())
    
    def get_by_difficulty(self, difficulty: str) -> Set[str]:
        return self.by_difficulty.get(difficulty, set())
    
    def get_skills_enabled_by(self, prerequisite: str) -> Set[str]:
        """Get skills that require this skill as a prerequisite."""
        return self.by_prerequisite.get(prerequisite, set())


class SkillManager:
    """Enhanced Skill Manager with difficulty, prerequisites, and versioning.
    
    Manages a library of learned skills with:
    - Difficulty ratings
    - Prerequisite tracking
    - Success rate monitoring
    - Skill composition
    - Version control
    """
    
    def __init__(
        self,
        model_name=None,
        temperature=0,
        retrieval_top_k=5,
        request_timout=120,
        ckpt_dir="ckpt",
        resume=False,
        llm_provider=None,  # Ignored, backward compatibility
    ):
        # Use Ollama LLM
        self.llm = get_llm(
            model_name=model_name,
            temperature=temperature,
            request_timeout=request_timout,
        )
        
        # Setup directories
        U.f_mkdir(f"{ckpt_dir}/skill/code")
        U.f_mkdir(f"{ckpt_dir}/skill/description")
        U.f_mkdir(f"{ckpt_dir}/skill/vectordb")
        U.f_mkdir(f"{ckpt_dir}/skill/metadata")
        U.f_mkdir(f"{ckpt_dir}/skill/versions")
        
        # Control primitives
        self.control_primitives = load_control_primitives()
        
        # Skill storage
        self.skills: Dict[str, Dict] = {}
        self.skill_metadata: Dict[str, SkillMetadata] = {}
        self.skill_index = SkillIndex()
        self.compositions: Dict[str, SkillComposition] = {}
        
        # Settings
        self.retrieval_top_k = retrieval_top_k
        self.ckpt_dir = ckpt_dir
        
        # Vector database for semantic search
        self.vectordb = Chroma(
            collection_name="skill_vectordb",
            embedding_function=get_embeddings(),
            persist_directory=f"{ckpt_dir}/skill/vectordb",
        )
        
        if resume:
            self._load_state()
        
        # Verify sync
        assert self.vectordb._collection.count() == len(self.skills), (
            f"Skill Manager's vectordb is not synced with skills.json.\n"
            f"There are {self.vectordb._collection.count()} skills in vectordb but {len(self.skills)} skills in skills.json.\n"
            f"Did you set resume=False when initializing the manager?\n"
            f"You may need to manually delete the vectordb directory for running from scratch."
        )
    
    def _load_state(self):
        """Load skill state from disk."""
        print(f"\033[33mLoading Skill Manager from {self.ckpt_dir}/skill\033[0m")
        
        # Load skills
        skills_path = f"{self.ckpt_dir}/skill/skills.json"
        if U.f_exists(skills_path):
            self.skills = U.load_json(skills_path)
        
        # Load metadata
        metadata_path = f"{self.ckpt_dir}/skill/metadata/skill_metadata.json"
        if U.f_exists(metadata_path):
            data = U.load_json(metadata_path)
            self.skill_metadata = {
                name: SkillMetadata.from_dict(m) for name, m in data.items()
            }
            
            # Rebuild index
            for name, meta in self.skill_metadata.items():
                self.skill_index.add_skill(name, meta)
        else:
            # Create metadata for existing skills
            for name in self.skills:
                if name not in self.skill_metadata:
                    self.skill_metadata[name] = SkillMetadata(
                        name=name,
                        description=self.skills[name].get("description", ""),
                        tags=self._infer_tags(name, self.skills[name].get("code", ""))
                    )
        
        # Load compositions
        compositions_path = f"{self.ckpt_dir}/skill/compositions.json"
        if U.f_exists(compositions_path):
            data = U.load_json(compositions_path)
            self.compositions = {
                name: SkillComposition(**c) for name, c in data.items()
            }
        
        print(f"\033[33mLoaded {len(self.skills)} skills with metadata\033[0m")
    
    def _save_state(self):
        """Save skill state to disk."""
        # Save skills
        U.dump_json(self.skills, f"{self.ckpt_dir}/skill/skills.json")
        
        # Save metadata
        metadata_data = {name: meta.to_dict() for name, meta in self.skill_metadata.items()}
        U.dump_json(metadata_data, f"{self.ckpt_dir}/skill/metadata/skill_metadata.json")
        
        # Save compositions
        compositions_data = {name: asdict(c) for name, c in self.compositions.items()}
        U.dump_json(compositions_data, f"{self.ckpt_dir}/skill/compositions.json")
        
        # Persist vector database
        self.vectordb.persist()
    
    def _infer_tags(self, name: str, code: str) -> List[str]:
        """Infer tags from skill name and code."""
        tags = []
        name_lower = name.lower()
        code_lower = code.lower()
        
        # Action type tags
        if "mine" in name_lower or "dig" in name_lower:
            tags.append("mining")
        if "craft" in name_lower:
            tags.append("crafting")
        if "build" in name_lower or "place" in name_lower:
            tags.append("building")
        if "smelt" in name_lower or "furnace" in code_lower:
            tags.append("smelting")
        if "farm" in name_lower or "plant" in name_lower:
            tags.append("farming")
        if "kill" in name_lower or "attack" in code_lower:
            tags.append("combat")
        if "explore" in name_lower or "goto" in name_lower:
            tags.append("navigation")
        if "collect" in name_lower or "gather" in name_lower:
            tags.append("gathering")
        
        # Material tags
        materials = ["wood", "stone", "iron", "gold", "diamond", "copper", "coal"]
        for mat in materials:
            if mat in name_lower or mat in code_lower:
                tags.append(mat)
        
        return tags
    
    def _estimate_difficulty(self, name: str, code: str, prerequisites: List[str]) -> float:
        """Estimate skill difficulty from code complexity."""
        difficulty = 0.3  # Base difficulty
        
        # Code complexity
        lines = code.count('\n')
        if lines > 50:
            difficulty += 0.2
        elif lines > 20:
            difficulty += 0.1
        
        # Control flow complexity
        if code.count('if') + code.count('while') + code.count('for') > 5:
            difficulty += 0.15
        
        # Async complexity
        if code.count('await') > 5:
            difficulty += 0.1
        
        # Prerequisites add difficulty
        difficulty += len(prerequisites) * 0.05
        
        # Certain operations are harder
        hard_ops = ["pathfinder", "combat", "enchant", "nether", "end"]
        for op in hard_ops:
            if op in code.lower():
                difficulty += 0.1
        
        return min(1.0, difficulty)
    
    @property
    def programs(self):
        """Get all programs as a string."""
        programs = ""
        for skill_name, entry in self.skills.items():
            programs += f"{entry['code']}\n\n"
        for primitives in self.control_primitives:
            programs += f"{primitives}\n\n"
        return programs
    
    def add_new_skill(self, info: Dict, prerequisites: List[str] = None):
        """Add a new skill with metadata."""
        if info["task"].startswith("Deposit useless items into the chest at"):
            return
        
        program_name = info["program_name"]
        program_code = info["program_code"]
        
        # Generate description
        skill_description = self.generate_skill_description(program_name, program_code)
        print(f"\033[33mSkill Manager generated description for {program_name}:\n{skill_description}\033[0m")
        
        # Handle existing skill (versioning)
        version = 1
        if program_name in self.skills:
            print(f"\033[33mSkill {program_name} already exists. Creating new version!\033[0m")
            
            # Archive old version
            old_version = self.skill_metadata.get(program_name, SkillMetadata(name=program_name, description="")).version
            version = old_version + 1
            
            # Save old version
            self._archive_version(program_name, old_version)
            
            # Remove from vectordb
            self.vectordb._collection.delete(ids=[program_name])
            
            # Update index
            if program_name in self.skill_metadata:
                self.skill_index.remove_skill(program_name, self.skill_metadata[program_name])
        
        # Infer tags and prerequisites
        tags = self._infer_tags(program_name, program_code)
        prereqs = prerequisites or self._infer_prerequisites(program_name, program_code)
        
        # Calculate difficulty
        difficulty = self._estimate_difficulty(program_name, program_code, prereqs)
        
        # Create metadata
        metadata = SkillMetadata(
            name=program_name,
            description=skill_description,
            difficulty=difficulty,
            prerequisites=prereqs,
            version=version,
            tags=tags
        )
        
        # Add to vectordb
        self.vectordb.add_texts(
            texts=[skill_description],
            ids=[program_name],
            metadatas=[{"name": program_name, "difficulty": difficulty, "version": version}],
        )
        
        # Store skill
        self.skills[program_name] = {
            "code": program_code,
            "description": skill_description,
        }
        self.skill_metadata[program_name] = metadata
        
        # Update index
        self.skill_index.add_skill(program_name, metadata)
        
        # Save to disk
        dumped_name = f"{program_name}V{version}" if version > 1 else program_name
        U.dump_text(program_code, f"{self.ckpt_dir}/skill/code/{dumped_name}.js")
        U.dump_text(skill_description, f"{self.ckpt_dir}/skill/description/{dumped_name}.txt")
        
        self._save_state()
        
        return metadata
    
    def _archive_version(self, name: str, version: int):
        """Archive an old skill version."""
        if name not in self.skills:
            return
        
        archive_path = f"{self.ckpt_dir}/skill/versions/{name}_v{version}.json"
        archive_data = {
            "code": self.skills[name]["code"],
            "description": self.skills[name]["description"],
            "metadata": self.skill_metadata[name].to_dict() if name in self.skill_metadata else {}
        }
        U.dump_json(archive_data, archive_path)
    
    def _infer_prerequisites(self, name: str, code: str) -> List[str]:
        """Infer skill prerequisites from code."""
        prerequisites = []
        name_lower = name.lower()
        code_lower = code.lower()
        
        # Tool-based prerequisites
        if "iron" in name_lower:
            if "pickaxe" in name_lower:
                prerequisites.append("craftStonePickaxe")
            prerequisites.append("smeltIronIngot")
        
        if "diamond" in name_lower:
            prerequisites.append("craftIronPickaxe")
        
        if "stone" in name_lower and "pickaxe" in name_lower:
            prerequisites.append("craftWoodenPickaxe")
        
        # Check for skill calls in code
        for existing_skill in self.skills:
            if existing_skill in code and existing_skill != name:
                prerequisites.append(existing_skill)
        
        return list(set(prerequisites))  # Remove duplicates
    
    def generate_skill_description(self, program_name: str, program_code: str) -> str:
        """Generate a description for a skill."""
        messages = [
            SystemMessage(content=load_prompt("skill")),
            HumanMessage(
                content=program_code + "\n\n" + f"The main function is `{program_name}`."
            ),
        ]
        skill_description = f"    // {self.llm(messages).content}"
        return f"async function {program_name}(bot) {{\n{skill_description}\n}}"
    
    def record_skill_execution(self, skill_name: str, success: bool, duration: float):
        """Record a skill execution for tracking."""
        if skill_name not in self.skill_metadata:
            return
        
        metadata = self.skill_metadata[skill_name]
        metadata.record_execution(success, duration)
        
        # Update indices
        self.skill_index.update_recently_used(skill_name)
        self.skill_index.update_success_ranking(skill_name, metadata.success_rate())
        
        # Adjust difficulty based on success rate
        if metadata.success_count + metadata.failure_count >= 5:
            if metadata.success_rate() > 0.8:
                metadata.difficulty = max(0.1, metadata.difficulty - 0.05)
            elif metadata.success_rate() < 0.3:
                metadata.difficulty = min(0.95, metadata.difficulty + 0.05)
        
        self._save_state()
    
    def retrieve_skills(self, query: str, difficulty_filter: str = None, 
                       tags: List[str] = None) -> List[str]:
        """Retrieve relevant skills with optional filtering."""
        k = min(self.vectordb._collection.count(), self.retrieval_top_k)
        if k == 0:
            return []
        
        print(f"\033[33mSkill Manager retrieving for {k} skills\033[0m")
        
        # Get candidates from vectordb
        docs_and_scores = self.vectordb.similarity_search_with_score(query, k=k * 2)
        
        skills = []
        for doc, score in docs_and_scores:
            skill_name = doc.metadata['name']
            
            # Apply filters
            if skill_name not in self.skill_metadata:
                continue
            
            meta = self.skill_metadata[skill_name]
            
            # Difficulty filter
            if difficulty_filter:
                if difficulty_filter == "easy" and meta.difficulty > 0.33:
                    continue
                elif difficulty_filter == "medium" and (meta.difficulty < 0.33 or meta.difficulty > 0.66):
                    continue
                elif difficulty_filter == "hard" and meta.difficulty < 0.66:
                    continue
            
            # Tag filter
            if tags and not any(tag in meta.tags for tag in tags):
                continue
            
            skills.append(self.skills[skill_name]["code"])
            
            if len(skills) >= self.retrieval_top_k:
                break
        
        print(f"\033[33mSkill Manager retrieved {len(skills)} skills\033[0m")
        return skills
    
    def retrieve_by_prerequisites(self, available_skills: Set[str]) -> List[str]:
        """Retrieve skills whose prerequisites are met."""
        accessible = []
        
        for name, meta in self.skill_metadata.items():
            if all(prereq in available_skills for prereq in meta.prerequisites):
                accessible.append(name)
        
        return accessible
    
    def get_skill_chain(self, target_skill: str) -> List[str]:
        """Get the chain of prerequisites needed for a skill."""
        if target_skill not in self.skill_metadata:
            return []
        
        chain = []
        visited = set()
        
        def collect_prereqs(skill_name):
            if skill_name in visited:
                return
            visited.add(skill_name)
            
            if skill_name not in self.skill_metadata:
                return
            
            for prereq in self.skill_metadata[skill_name].prerequisites:
                collect_prereqs(prereq)
            
            chain.append(skill_name)
        
        collect_prereqs(target_skill)
        return chain
    
    def compose_skills(self, name: str, component_skills: List[str], 
                      sequence: List[str] = None) -> Optional[SkillComposition]:
        """Create a composed skill from multiple skills."""
        # Verify all component skills exist
        for skill in component_skills:
            if skill not in self.skills:
                print(f"\033[31mSkill {skill} not found for composition\033[0m")
                return None
        
        composition = SkillComposition(
            name=name,
            component_skills=component_skills,
            sequence=sequence or component_skills
        )
        composition.calculate_difficulty(self.skill_metadata)
        
        self.compositions[name] = composition
        
        # Create combined code
        combined_code = f"async function {name}(bot) {{\n"
        for skill in composition.sequence:
            combined_code += f"  await {skill}(bot);\n"
        combined_code += "}"
        
        # Add as a new skill
        self.add_new_skill({
            "program_name": name,
            "program_code": combined_code,
            "task": f"Composed skill: {name}"
        }, prerequisites=component_skills)
        
        # Mark as composed
        if name in self.skill_metadata:
            self.skill_metadata[name].composed_from = component_skills
        
        return composition
    
    def get_skill_stats(self) -> Dict:
        """Get statistics about the skill library."""
        if not self.skill_metadata:
            return {"total_skills": 0}
        
        difficulties = [m.difficulty for m in self.skill_metadata.values()]
        success_rates = [m.success_rate() for m in self.skill_metadata.values() 
                        if m.success_count + m.failure_count > 0]
        
        tags_count = defaultdict(int)
        for meta in self.skill_metadata.values():
            for tag in meta.tags:
                tags_count[tag] += 1
        
        return {
            "total_skills": len(self.skills),
            "avg_difficulty": sum(difficulties) / len(difficulties) if difficulties else 0,
            "avg_success_rate": sum(success_rates) / len(success_rates) if success_rates else 0.5,
            "skills_by_difficulty": {
                "easy": len([d for d in difficulties if d < 0.33]),
                "medium": len([d for d in difficulties if 0.33 <= d < 0.66]),
                "hard": len([d for d in difficulties if d >= 0.66])
            },
            "top_tags": dict(sorted(tags_count.items(), key=lambda x: x[1], reverse=True)[:10]),
            "most_successful": self.skill_index.most_successful[:5],
            "recently_used": self.skill_index.recently_used[:5],
            "compositions": len(self.compositions)
        }
    
    def get_skill_metadata(self, skill_name: str) -> Optional[SkillMetadata]:
        """Get metadata for a specific skill."""
        return self.skill_metadata.get(skill_name)

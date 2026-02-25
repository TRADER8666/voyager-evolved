"""Voyager Evolved - Enhanced Minecraft AI Agent.

This is the main class that integrates all evolved features:
- Player observation and learning
- Evolutionary goal generation
- Human-like behavior patterns
- Personality-driven decision making
"""

import copy
import json
import os
import time
from typing import Dict, List, Optional, Any

import voyager.utils as U
from voyager.env import VoyagerEnv
from voyager.agents import ActionAgent, CriticAgent, CurriculumAgent, SkillManager

from .config import EvolvedConfig
from .player_observer import PlayerObserver
from .evolutionary_goals import EvolutionaryGoalSystem
from .human_behavior import HumanBehaviorSystem
from .observational_learning import ObservationalLearningIntegration
from .personality import PersonalityEngine


class VoyagerEvolved:
    """Enhanced Voyager agent with emergent behaviors and observational learning.
    
    Key Features:
    1. Player Observation: Detects and learns from other players on the server
    2. Evolutionary Goals: Generates goals based on survival, curiosity, and fitness
    3. Human-like Behavior: Adds natural movement, pauses, and imperfections
    4. Observational Learning: Incorporates observed strategies into skills
    5. Personality System: Traits that influence decision-making and evolution
    
    Uses Ollama for local LLM - no API key required!
    """
    
    def __init__(
        self,
        mc_port: int = None,
        azure_login: Dict[str, str] = None,
        server_port: int = 3000,
        env_wait_ticks: int = 20,
        env_request_timeout: int = 600,
        max_iterations: int = 160,
        reset_placed_if_failed: bool = False,
        # Model names - None uses default (llama2)
        action_agent_model_name: str = None,
        action_agent_temperature: float = 0,
        action_agent_task_max_retries: int = 4,
        action_agent_show_chat_log: bool = True,
        action_agent_show_execution_error: bool = True,
        curriculum_agent_model_name: str = None,
        curriculum_agent_temperature: float = 0,
        curriculum_agent_qa_model_name: str = None,
        curriculum_agent_qa_temperature: float = 0,
        curriculum_agent_warm_up: Dict[str, int] = None,
        curriculum_agent_core_inventory_items: str = r".*_log|.*_planks|stick|crafting_table|furnace"
        r"|cobblestone|dirt|coal|.*_pickaxe|.*_sword|.*_axe",
        curriculum_agent_mode: str = "auto",
        critic_agent_model_name: str = None,
        critic_agent_temperature: float = 0,
        critic_agent_mode: str = "auto",
        skill_manager_model_name: str = None,
        skill_manager_temperature: float = 0,
        skill_manager_retrieval_top_k: int = 5,
        llm_request_timeout: int = 240,
        ckpt_dir: str = "ckpt",
        skill_library_dir: str = None,
        resume: bool = False,
        # Evolved-specific parameters
        evolved_config: EvolvedConfig = None,
        evolved_config_path: str = None,
    ):
        """Initialize VoyagerEvolved with enhanced features.
        
        Uses Ollama for local LLM - no API key required!
        
        :param evolved_config: EvolvedConfig instance for evolved features
        :param evolved_config_path: Path to load EvolvedConfig from JSON
        """
        # Load or create evolved config
        if evolved_config_path and os.path.exists(evolved_config_path):
            self.evolved_config = EvolvedConfig.load(evolved_config_path)
            print(f"\033[36mLoaded evolved config from {evolved_config_path}\033[0m")
        elif evolved_config:
            self.evolved_config = evolved_config
        else:
            self.evolved_config = EvolvedConfig()
        
        # Initialize environment
        self.env = VoyagerEnv(
            mc_port=mc_port,
            azure_login=azure_login,
            server_port=server_port,
            request_timeout=env_request_timeout,
        )
        self.env_wait_ticks = env_wait_ticks
        self.reset_placed_if_failed = reset_placed_if_failed
        self.max_iterations = max_iterations
        
        # Initialize agents with Ollama LLM
        self.action_agent = ActionAgent(
            model_name=action_agent_model_name,
            temperature=action_agent_temperature,
            request_timout=llm_request_timeout,
            ckpt_dir=ckpt_dir,
            resume=resume,
            chat_log=action_agent_show_chat_log,
            execution_error=action_agent_show_execution_error,
        )
        self.action_agent_task_max_retries = action_agent_task_max_retries
        
        self.curriculum_agent = CurriculumAgent(
            model_name=curriculum_agent_model_name,
            temperature=curriculum_agent_temperature,
            qa_model_name=curriculum_agent_qa_model_name,
            qa_temperature=curriculum_agent_qa_temperature,
            request_timout=llm_request_timeout,
            ckpt_dir=ckpt_dir,
            resume=resume,
            mode=curriculum_agent_mode,
            warm_up=curriculum_agent_warm_up,
            core_inventory_items=curriculum_agent_core_inventory_items,
        )
        
        self.critic_agent = CriticAgent(
            model_name=critic_agent_model_name,
            temperature=critic_agent_temperature,
            request_timout=llm_request_timeout,
            mode=critic_agent_mode,
        )
        
        self.skill_manager = SkillManager(
            model_name=skill_manager_model_name,
            temperature=skill_manager_temperature,
            retrieval_top_k=skill_manager_retrieval_top_k,
            request_timout=llm_request_timeout,
            ckpt_dir=skill_library_dir if skill_library_dir else ckpt_dir,
            resume=True if resume or skill_library_dir else False,
        )
        
        self.recorder = U.EventRecorder(ckpt_dir=ckpt_dir, resume=resume)
        self.resume = resume
        self.ckpt_dir = ckpt_dir
        
        # Initialize evolved components
        self._init_evolved_components()
        
        # Initialize rollout variables
        self.action_agent_rollout_num_iter = -1
        self.task = None
        self.context = ""
        self.messages = None
        self.conversations = []
        self.last_events = None
        self.current_goal_metadata = None
        
        print("\033[36m╔══════════════════════════════════════════════════════════╗\033[0m")
        print("\033[36m║            VOYAGER EVOLVED - Enhanced AI Agent           ║\033[0m")
        print("\033[36m╚══════════════════════════════════════════════════════════╝\033[0m")
        print(f"\033[36mFeatures enabled:\033[0m")
        print(f"  • Player Observation: {self.evolved_config.enable_player_observation}")
        print(f"  • Evolutionary Goals: {self.evolved_config.enable_evolutionary_goals}")
        print(f"  • Human Behavior: {self.evolved_config.enable_human_behavior}")
        print(f"  • Observational Learning: {self.evolved_config.enable_observational_learning}")
    
    def _init_evolved_components(self):
        """Initialize evolved behavior components."""
        # Personality Engine (always enabled as it affects other systems)
        self.personality = PersonalityEngine(
            config=self.evolved_config,
            ckpt_dir=self.ckpt_dir,
            resume=self.resume
        )
        
        # Player Observer
        if self.evolved_config.enable_player_observation:
            self.player_observer = PlayerObserver(
                config=self.evolved_config,
                ckpt_dir=self.ckpt_dir,
                resume=self.resume
            )
        else:
            self.player_observer = None
        
        # Evolutionary Goal System
        if self.evolved_config.enable_evolutionary_goals:
            self.goal_system = EvolutionaryGoalSystem(
                config=self.evolved_config,
                personality_engine=self.personality,
                player_observer=self.player_observer,
                ckpt_dir=self.ckpt_dir,
                resume=self.resume
            )
        else:
            self.goal_system = None
        
        # Human Behavior System
        if self.evolved_config.enable_human_behavior:
            self.human_behavior = HumanBehaviorSystem(
                config=self.evolved_config,
                personality_engine=self.personality
            )
        else:
            self.human_behavior = None
        
        # Observational Learning Integration
        if self.evolved_config.enable_observational_learning and self.player_observer:
            self.observational_learning = ObservationalLearningIntegration(
                observer=self.player_observer,
                skill_manager=self.skill_manager,
                config=self.evolved_config,
                ckpt_dir=self.ckpt_dir,
                resume=self.resume
            )
        else:
            self.observational_learning = None
    
    def reset(self, task, context="", reset_env=True):
        """Reset agent for a new task with evolved behaviors."""
        self.action_agent_rollout_num_iter = 0
        self.task = task
        self.context = context
        
        if reset_env:
            self.env.reset(
                options={
                    "mode": "soft",
                    "wait_ticks": self.env_wait_ticks,
                }
            )
        
        difficulty = (
            "easy" if len(self.curriculum_agent.completed_tasks) > 15 else "peaceful"
        )
        
        # Step to peek an observation
        events = self.env.step(
            "bot.chat(`/time set ${getNextTime()}`);\n"
            + f"bot.chat('/difficulty {difficulty}');"
        )
        
        # Process events with evolved systems
        self._process_events_evolved(events)
        
        # Add human-like behavior if enabled
        if self.human_behavior:
            self.human_behavior.update_interesting_things(events)
        
        skills = self.skill_manager.retrieve_skills(query=self.context)
        
        # Add learned strategies to context
        if self.observational_learning:
            strategies = self.observational_learning.get_relevant_strategies(
                context={"task": task},
                task_type=task,
                top_k=3
            )
            if strategies:
                strategy_context = "\n\nLearned strategies from observing other players:\n"
                for s in strategies:
                    strategy_context += f"- {s.description}\n"
                self.context += strategy_context
        
        print(f"\033[33mRender Action Agent system message with {len(skills)} skills\033[0m")
        system_message = self.action_agent.render_system_message(skills=skills)
        human_message = self.action_agent.render_human_message(
            events=events, code="", task=self.task, context=self.context, critique=""
        )
        self.messages = [system_message, human_message]
        print(f"\033[32m****Action Agent human message****\n{human_message.content}\033[0m")
        assert len(self.messages) == 2
        self.conversations = []
        return self.messages
    
    def _process_events_evolved(self, events: List):
        """Process game events through evolved systems."""
        # Player observation
        if self.player_observer:
            observation_results = self.player_observer.process_events(events)
            
            if observation_results.get("updated"):
                players = observation_results.get("players_detected", [])
                if players and self.evolved_config.log_observations:
                    print(f"\033[36m[Observer] Detected players: {', '.join(players)}\033[0m")
                
                new_behaviors = observation_results.get("new_behaviors", [])
                if new_behaviors and self.evolved_config.log_observations:
                    for b in new_behaviors:
                        print(f"\033[36m[Observer] Recorded behavior: {b['activity']} by {b['player_name']}\033[0m")
        
        # Process observations for learning
        if self.observational_learning:
            new_strategies = self.observational_learning.process_new_observations()
            if new_strategies and self.evolved_config.log_observations:
                for s in new_strategies:
                    print(f"\033[35m[Learning] Learned new strategy: {s.description}\033[0m")
        
        # Update survival state if goal system is active
        if self.goal_system:
            self.goal_system.update_survival_state(events)
    
    def close(self):
        """Close environment and save all evolved states."""
        # Save evolved states
        self._save_evolved_states()
        self.env.close()
    
    def _save_evolved_states(self):
        """Save all evolved component states."""
        if self.evolved_config.save_personality_state:
            self.personality.save_state()
        
        if self.player_observer:
            self.player_observer.save_state()
        
        if self.goal_system:
            self.goal_system.save_state()
        
        if self.observational_learning and self.evolved_config.save_learned_behaviors:
            self.observational_learning.save_state()
        
        # Save evolved config
        self.evolved_config.save(f"{self.ckpt_dir}/evolved_config.json")
        
        print("\033[36m[VoyagerEvolved] Saved all evolved states\033[0m")
    
    def step(self):
        """Perform one step with evolved behaviors."""
        if self.action_agent_rollout_num_iter < 0:
            raise ValueError("Agent must be reset before stepping")
        
        # Add human-like pause before action
        if self.human_behavior:
            should_break, break_duration = self.human_behavior.should_take_break()
            if should_break:
                print(f"\033[33m[Human] Taking a short break ({break_duration:.1f}s)...\033[0m")
                time.sleep(break_duration)
        
        ai_message = self.action_agent.llm(self.messages)
        print(f"\033[34m****Action Agent ai message****\n{ai_message.content}\033[0m")
        
        self.conversations.append(
            (self.messages[0].content, self.messages[1].content, ai_message.content)
        )
        
        parsed_result = self.action_agent.process_ai_message(message=ai_message)
        success = False
        
        if isinstance(parsed_result, dict):
            code = parsed_result["program_code"] + "\n" + parsed_result["exec_code"]
            
            # Add human-like behaviors to code
            if self.human_behavior:
                code = self.human_behavior.generate_natural_action_code(
                    code, 
                    action_type=self._infer_action_type(code)
                )
                
                # Maybe simulate a mistake
                should_mistake, mistake_type = self.human_behavior.should_make_mistake()
                if should_mistake:
                    print(f"\033[33m[Human] Made a small mistake ({mistake_type}), correcting...\033[0m")
            
            events = self.env.step(
                code,
                programs=self.skill_manager.programs,
            )
            
            # Process events with evolved systems
            self._process_events_evolved(events)
            
            self.recorder.record(events, self.task)
            self.action_agent.update_chest_memory(events[-1][1]["nearbyChests"])
            
            success, critique = self.critic_agent.check_task_success(
                events=events,
                task=self.task,
                context=self.context,
                chest_observation=self.action_agent.render_chest_observation(),
                max_retries=5,
            )
            
            # Update personality based on result
            if success:
                self.personality.record_success(self.task)
            else:
                self.personality.record_failure(self.task)
            
            if self.reset_placed_if_failed and not success:
                # Revert placed blocks
                blocks = []
                positions = []
                for event_type, event in events:
                    if event_type == "onSave" and event["onSave"].endswith("_placed"):
                        block = event["onSave"].split("_placed")[0]
                        position = event["status"]["position"]
                        blocks.append(block)
                        positions.append(position)
                new_events = self.env.step(
                    f"await givePlacedItemBack(bot, {U.json_dumps(blocks)}, {U.json_dumps(positions)})",
                    programs=self.skill_manager.programs,
                )
                events[-1][1]["inventory"] = new_events[-1][1]["inventory"]
                events[-1][1]["voxels"] = new_events[-1][1]["voxels"]
            
            new_skills = self.skill_manager.retrieve_skills(
                query=self.context + "\n\n" + self.action_agent.summarize_chatlog(events)
            )
            system_message = self.action_agent.render_system_message(skills=new_skills)
            human_message = self.action_agent.render_human_message(
                events=events,
                code=parsed_result["program_code"],
                task=self.task,
                context=self.context,
                critique=critique,
            )
            self.last_events = copy.deepcopy(events)
            self.messages = [system_message, human_message]
        else:
            assert isinstance(parsed_result, str)
            self.recorder.record([], self.task)
            print(f"\033[34m{parsed_result} Trying again!\033[0m")
        
        assert len(self.messages) == 2
        self.action_agent_rollout_num_iter += 1
        
        done = (
            self.action_agent_rollout_num_iter >= self.action_agent_task_max_retries
            or success
        )
        
        info = {
            "task": self.task,
            "success": success,
            "conversations": self.conversations,
        }
        
        if success:
            assert (
                "program_code" in parsed_result and "program_name" in parsed_result
            ), "program and program_name must be returned when success"
            info["program_code"] = parsed_result["program_code"]
            info["program_name"] = parsed_result["program_name"]
        else:
            print(
                f"\033[32m****Action Agent human message****\n{self.messages[-1].content}\033[0m"
            )
        
        return self.messages, 0, done, info
    
    def _infer_action_type(self, code: str) -> str:
        """Infer action type from code for human behavior."""
        code_lower = code.lower()
        
        if "goto" in code_lower or "moveto" in code_lower:
            return "move"
        elif "mine" in code_lower or "dig" in code_lower:
            return "mine"
        elif "place" in code_lower or "build" in code_lower:
            return "build"
        elif "craft" in code_lower:
            return "craft"
        elif "attack" in code_lower or "fight" in code_lower:
            return "fight"
        
        return "general"
    
    def rollout(self, *, task, context, reset_env=True):
        """Perform rollout for a task."""
        self.reset(task=task, context=context, reset_env=reset_env)
        while True:
            messages, reward, done, info = self.step()
            if done:
                break
        return messages, reward, done, info
    
    def learn(self, reset_env=True):
        """Main learning loop with evolved goal generation."""
        if self.resume:
            self.env.reset(
                options={
                    "mode": "soft",
                    "wait_ticks": self.env_wait_ticks,
                }
            )
        else:
            self.env.reset(
                options={
                    "mode": "hard",
                    "wait_ticks": self.env_wait_ticks,
                }
            )
            self.resume = True
        
        self.last_events = self.env.step("")
        iteration = 0
        
        while True:
            if self.recorder.iteration > self.max_iterations:
                print("Iteration limit reached")
                break
            
            # Get next goal using evolved system or original curriculum
            if self.goal_system and self.evolved_config.enable_evolutionary_goals:
                task, context, goal_metadata = self.goal_system.generate_next_goal(
                    events=self.last_events,
                    completed_tasks=self.curriculum_agent.completed_tasks,
                    inventory=self.last_events[-1][1].get("inventory", {})
                )
                self.current_goal_metadata = goal_metadata
                
                if self.evolved_config.log_goal_evolution:
                    print(f"\033[35m[Goals] Generated goal ({goal_metadata['category']}): {task}\033[0m")
                    print(f"\033[35m[Goals] Priority: {goal_metadata['priority']:.2f}, Source: {goal_metadata.get('derived_from', 'unknown')}\033[0m")
            else:
                task, context = self.curriculum_agent.propose_next_task(
                    events=self.last_events,
                    chest_observation=self.action_agent.render_chest_observation(),
                    max_retries=5,
                )
                self.current_goal_metadata = None
            
            print(
                f"\033[35mStarting task {task} for at most {self.action_agent_task_max_retries} times\033[0m"
            )
            
            try:
                messages, reward, done, info = self.rollout(
                    task=task,
                    context=context,
                    reset_env=reset_env,
                )
            except Exception as e:
                time.sleep(3)
                info = {
                    "task": task,
                    "success": False,
                }
                
                self.last_events = self.env.reset(
                    options={
                        "mode": "hard",
                        "wait_ticks": self.env_wait_ticks,
                        "inventory": self.last_events[-1][1]["inventory"],
                        "equipment": self.last_events[-1][1]["status"]["equipment"],
                        "position": self.last_events[-1][1]["status"]["position"],
                    }
                )
                print("Your last round rollout terminated due to error:")
                print(f"\033[41m{e}\033[0m")
            
            # Record result in evolved systems
            if self.goal_system and self.current_goal_metadata:
                self.goal_system.record_goal_result(
                    self.current_goal_metadata["goal_id"],
                    info["success"]
                )
            
            if info["success"]:
                self.skill_manager.add_new_skill(info)
            
            self.curriculum_agent.update_exploration_progress(info)
            
            print(
                f"\033[35mCompleted tasks: {', '.join(self.curriculum_agent.completed_tasks)}\033[0m"
            )
            print(
                f"\033[35mFailed tasks: {', '.join(self.curriculum_agent.failed_tasks)}\033[0m"
            )
            
            # Periodic saves
            iteration += 1
            if iteration % self.evolved_config.checkpoint_interval == 0:
                self._save_evolved_states()
                
                # Print evolved statistics
                if self.evolved_config.verbose_mode:
                    self._print_evolved_stats()
        
        # Final save
        self._save_evolved_states()
        
        return {
            "completed_tasks": self.curriculum_agent.completed_tasks,
            "failed_tasks": self.curriculum_agent.failed_tasks,
            "skills": self.skill_manager.skills,
            "evolved_stats": self._get_evolved_stats()
        }
    
    def _print_evolved_stats(self):
        """Print statistics about evolved behaviors."""
        print("\n\033[36m═══════════════ Evolved Statistics ═══════════════\033[0m")
        
        # Personality
        personality_summary = self.personality.get_personality_summary()
        print(f"\033[36mPersonality:\033[0m")
        print(f"  Mood: {personality_summary['overall_mood']:.2f}")
        print(f"  Energy: {personality_summary['energy']:.2f}")
        print(f"  Consecutive Successes: {personality_summary['consecutive_successes']}")
        
        # Player observation
        if self.player_observer:
            obs_summary = self.player_observer.get_observation_summary()
            print(f"\033[36mObservation:\033[0m")
            print(f"  Players Observed: {obs_summary['total_players_observed']}")
            print(f"  Behaviors Recorded: {obs_summary['total_behaviors_recorded']}")
        
        # Goal system
        if self.goal_system:
            goal_stats = self.goal_system.get_goal_statistics()
            print(f"\033[36mGoals:\033[0m")
            print(f"  Total Generated: {goal_stats['total_goals_generated']}")
            print(f"  Completed: {goal_stats['completed']}")
            print(f"  Failed: {goal_stats['failed']}")
        
        # Learning
        if self.observational_learning:
            learn_summary = self.observational_learning.get_learning_summary()
            print(f"\033[36mLearning:\033[0m")
            print(f"  Strategies Learned: {learn_summary['total_strategies_learned']}")
            print(f"  Strategy Uses: {learn_summary['total_strategy_uses']}")
        
        print("\033[36m══════════════════════════════════════════════════\033[0m\n")
    
    def _get_evolved_stats(self) -> Dict:
        """Get all evolved statistics."""
        stats = {
            "personality": self.personality.get_personality_summary()
        }
        
        if self.player_observer:
            stats["observation"] = self.player_observer.get_observation_summary()
        
        if self.goal_system:
            stats["goals"] = self.goal_system.get_goal_statistics()
        
        if self.observational_learning:
            stats["learning"] = self.observational_learning.get_learning_summary()
        
        return stats
    
    # Support original Voyager methods for compatibility
    def decompose_task(self, task):
        """Decompose a task into subtasks."""
        if not self.last_events:
            self.last_events = self.env.reset(
                options={
                    "mode": "hard",
                    "wait_ticks": self.env_wait_ticks,
                }
            )
        return self.curriculum_agent.decompose_task(task, self.last_events)
    
    def inference(self, task=None, sub_goals=[], reset_mode="hard", reset_env=True):
        """Run inference on a specific task."""
        if not task and not sub_goals:
            raise ValueError("Either task or sub_goals must be provided")
        if not sub_goals:
            sub_goals = self.decompose_task(task)
        
        self.env.reset(
            options={
                "mode": reset_mode,
                "wait_ticks": self.env_wait_ticks,
            }
        )
        self.curriculum_agent.completed_tasks = []
        self.curriculum_agent.failed_tasks = []
        self.last_events = self.env.step("")
        
        while self.curriculum_agent.progress < len(sub_goals):
            next_task = sub_goals[self.curriculum_agent.progress]
            context = self.curriculum_agent.get_task_context(next_task)
            
            print(
                f"\033[35mStarting task {next_task} for at most {self.action_agent_task_max_retries} times\033[0m"
            )
            
            messages, reward, done, info = self.rollout(
                task=next_task,
                context=context,
                reset_env=reset_env,
            )
            self.curriculum_agent.update_exploration_progress(info)
            
            print(
                f"\033[35mCompleted tasks: {', '.join(self.curriculum_agent.completed_tasks)}\033[0m"
            )
            print(
                f"\033[35mFailed tasks: {', '.join(self.curriculum_agent.failed_tasks)}\033[0m"
            )

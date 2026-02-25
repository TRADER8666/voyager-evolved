"""Image embedding generation for visual memory.

GPU-accelerated embedding extraction for scene and object recognition.
"""

import logging
import time
import hashlib
from typing import Optional, List, Tuple, Dict, Any, Union
from dataclasses import dataclass, field
from collections import OrderedDict
import threading

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ImageEmbedding:
    """Embedding for an image or image region."""
    embedding: np.ndarray
    timestamp: float
    source_hash: str  # Hash of source image for deduplication
    embedding_type: str  # "scene", "object", "region"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def dim(self) -> int:
        return len(self.embedding)
    
    def similarity(self, other: 'ImageEmbedding') -> float:
        """Compute cosine similarity with another embedding."""
        return cosine_similarity(self.embedding, other.embedding)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'embedding': self.embedding.tolist(),
            'timestamp': self.timestamp,
            'source_hash': self.source_hash,
            'embedding_type': self.embedding_type,
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ImageEmbedding':
        """Create from dictionary."""
        return cls(
            embedding=np.array(data['embedding'], dtype=np.float32),
            timestamp=data['timestamp'],
            source_hash=data['source_hash'],
            embedding_type=data['embedding_type'],
            metadata=data.get('metadata', {}),
        )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class EmbeddingCache:
    """LRU cache for image embeddings.
    
    Avoids recomputing embeddings for similar images.
    """
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._cache: OrderedDict[str, ImageEmbedding] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[ImageEmbedding]:
        """Get embedding by key."""
        with self._lock:
            if key in self._cache:
                self._hits += 1
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                return self._cache[key]
            self._misses += 1
            return None
    
    def put(self, key: str, embedding: ImageEmbedding) -> None:
        """Store embedding in cache."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.max_size:
                    # Remove oldest
                    self._cache.popitem(last=False)
                self._cache[key] = embedding
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    
    @property
    def size(self) -> int:
        return len(self._cache)


class ImageEmbedder:
    """GPU-accelerated image embedding generator.
    
    Features:
    - Multiple backbone models (ResNet, EfficientNet, ViT)
    - Scene and object-level embeddings
    - Caching for efficiency
    - Batch processing support
    - Multi-GPU distribution
    """
    
    def __init__(
        self,
        model_name: str = "resnet50",
        embedding_dim: int = 512,
        normalize: bool = True,
        cache_size: int = 10000,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.normalize = normalize
        
        self._device = None
        self._model = None
        self._transform = None
        self._torch_available = False
        self._lock = threading.Lock()
        
        self.cache = EmbeddingCache(max_size=cache_size)
        
        self._initialize(device)
    
    def _initialize(self, device: Optional[str]) -> None:
        """Initialize embedding model."""
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms
            
            self._torch_available = True
            
            # Set device
            if device:
                self._device = torch.device(device)
            elif torch.cuda.is_available():
                self._device = torch.device('cuda:0')
            else:
                self._device = torch.device('cpu')
            
            logger.info(f"Embedder using device: {self._device}")
            
            # Load model
            self._load_model(models)
            
            # Set up transforms
            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
            
        except ImportError:
            logger.warning("PyTorch not available, embeddings disabled")
            self._torch_available = False
    
    def _load_model(self, models) -> None:
        """Load the embedding model."""
        import torch
        import torch.nn as nn
        
        model_loaders = {
            'resnet50': lambda: models.resnet50(weights='DEFAULT'),
            'resnet18': lambda: models.resnet18(weights='DEFAULT'),
            'efficientnet_b0': lambda: models.efficientnet_b0(weights='DEFAULT'),
            'efficientnet_b2': lambda: models.efficientnet_b2(weights='DEFAULT'),
        }
        
        loader = model_loaders.get(self.model_name)
        if loader is None:
            logger.warning(f"Unknown model {self.model_name}, using resnet50")
            loader = model_loaders['resnet50']
        
        model = loader()
        
        # Remove classification head to get features
        if 'resnet' in self.model_name:
            # ResNet: remove the final fc layer
            self._feature_dim = model.fc.in_features
            model.fc = nn.Identity()
        elif 'efficientnet' in self.model_name:
            # EfficientNet: remove classifier
            self._feature_dim = model.classifier[1].in_features
            model.classifier = nn.Identity()
        
        # Add projection layer if needed
        if self._feature_dim != self.embedding_dim:
            projection = nn.Sequential(
                nn.Linear(self._feature_dim, self.embedding_dim),
                nn.ReLU(),
                nn.Linear(self.embedding_dim, self.embedding_dim),
            )
            self._model = nn.Sequential(model, projection)
        else:
            self._model = model
        
        self._model.to(self._device)
        self._model.eval()
        
        # Enable FP16 for CUDA
        if self._device.type == 'cuda':
            self._model = self._model.half()
        
        logger.info(f"Loaded embedding model: {self.model_name}")
    
    def _image_hash(self, image: np.ndarray) -> str:
        """Compute hash of image for caching."""
        # Use a quick hash based on downsampled image
        try:
            import cv2
            small = cv2.resize(image, (32, 32))
        except:
            small = image[::32, ::32]
        
        return hashlib.md5(small.tobytes()).hexdigest()
    
    def embed(self, image: np.ndarray, use_cache: bool = True) -> ImageEmbedding:
        """Generate embedding for an image.
        
        Args:
            image: BGR image as numpy array
            use_cache: Whether to use caching
        
        Returns:
            ImageEmbedding object
        """
        image_hash = self._image_hash(image)
        
        # Check cache
        if use_cache:
            cached = self.cache.get(image_hash)
            if cached is not None:
                return cached
        
        # Generate embedding
        embedding_vec = self._compute_embedding(image)
        
        # Normalize if requested
        if self.normalize and embedding_vec is not None:
            norm = np.linalg.norm(embedding_vec)
            if norm > 0:
                embedding_vec = embedding_vec / norm
        
        embedding = ImageEmbedding(
            embedding=embedding_vec if embedding_vec is not None else np.zeros(self.embedding_dim, dtype=np.float32),
            timestamp=time.time(),
            source_hash=image_hash,
            embedding_type="scene",
        )
        
        # Cache result
        if use_cache:
            self.cache.put(image_hash, embedding)
        
        return embedding
    
    def _compute_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Compute embedding using the model."""
        if not self._torch_available or self._model is None:
            return None
        
        import torch
        
        try:
            # Convert BGR to RGB
            image_rgb = image[:, :, ::-1].copy()
            
            # Apply transforms
            tensor = self._transform(image_rgb)
            
            if self._device.type == 'cuda':
                tensor = tensor.half()
            
            tensor = tensor.unsqueeze(0).to(self._device)
            
            # Get embedding
            with torch.no_grad():
                embedding = self._model(tensor)
            
            return embedding.cpu().numpy().flatten().astype(np.float32)
            
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return None
    
    def embed_region(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        use_cache: bool = True,
    ) -> ImageEmbedding:
        """Generate embedding for a region of an image.
        
        Args:
            image: Full image
            bbox: Bounding box (x1, y1, x2, y2)
            use_cache: Whether to use caching
        
        Returns:
            ImageEmbedding for the region
        """
        x1, y1, x2, y2 = bbox
        region = image[y1:y2, x1:x2]
        
        if region.size == 0:
            return ImageEmbedding(
                embedding=np.zeros(self.embedding_dim, dtype=np.float32),
                timestamp=time.time(),
                source_hash="empty",
                embedding_type="object",
            )
        
        embedding = self.embed(region, use_cache=use_cache)
        embedding.embedding_type = "object"
        embedding.metadata['bbox'] = bbox
        
        return embedding
    
    def embed_batch(
        self,
        images: List[np.ndarray],
        use_cache: bool = True,
    ) -> List[ImageEmbedding]:
        """Generate embeddings for a batch of images.
        
        Optimized for GPU processing.
        """
        if not self._torch_available or self._model is None:
            return [self.embed(img, use_cache) for img in images]
        
        import torch
        
        results = []
        uncached_indices = []
        uncached_images = []
        
        # Check cache first
        for i, image in enumerate(images):
            image_hash = self._image_hash(image)
            if use_cache:
                cached = self.cache.get(image_hash)
                if cached is not None:
                    results.append((i, cached))
                    continue
            
            uncached_indices.append(i)
            uncached_images.append(image)
        
        # Process uncached in batch
        if uncached_images:
            try:
                # Prepare batch
                batch_tensors = []
                for image in uncached_images:
                    image_rgb = image[:, :, ::-1].copy()
                    tensor = self._transform(image_rgb)
                    batch_tensors.append(tensor)
                
                batch = torch.stack(batch_tensors)
                if self._device.type == 'cuda':
                    batch = batch.half()
                batch = batch.to(self._device)
                
                # Get embeddings
                with torch.no_grad():
                    embeddings = self._model(batch)
                
                embeddings = embeddings.cpu().numpy()
                
                # Create embedding objects
                for i, (idx, image) in enumerate(zip(uncached_indices, uncached_images)):
                    vec = embeddings[i].flatten().astype(np.float32)
                    
                    if self.normalize:
                        norm = np.linalg.norm(vec)
                        if norm > 0:
                            vec = vec / norm
                    
                    embedding = ImageEmbedding(
                        embedding=vec,
                        timestamp=time.time(),
                        source_hash=self._image_hash(image),
                        embedding_type="scene",
                    )
                    
                    if use_cache:
                        self.cache.put(embedding.source_hash, embedding)
                    
                    results.append((idx, embedding))
                    
            except Exception as e:
                logger.error(f"Batch embedding error: {e}")
                # Fall back to individual processing
                for idx, image in zip(uncached_indices, uncached_images):
                    results.append((idx, self.embed(image, use_cache)))
        
        # Sort by original index and return embeddings
        results.sort(key=lambda x: x[0])
        return [emb for _, emb in results]
    
    def find_similar(
        self,
        query: ImageEmbedding,
        candidates: List[ImageEmbedding],
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> List[Tuple[ImageEmbedding, float]]:
        """Find similar embeddings.
        
        Args:
            query: Query embedding
            candidates: List of candidate embeddings
            top_k: Number of results to return
            threshold: Minimum similarity threshold
        
        Returns:
            List of (embedding, similarity) tuples
        """
        results = []
        
        for candidate in candidates:
            sim = query.similarity(candidate)
            if sim >= threshold:
                results.append((candidate, sim))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedder statistics."""
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'device': str(self._device),
            'cache_size': self.cache.size,
            'cache_hit_rate': self.cache.hit_rate,
        }

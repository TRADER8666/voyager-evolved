"""Screen capture and frame buffer for vision processing.

Optimized for high-throughput Minecraft screenshot capture.
"""

import logging
import threading
import time
from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
import io

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Frame:
    """A single captured frame."""
    image: np.ndarray
    timestamp: float
    frame_number: int
    resolution: Tuple[int, int]
    compressed: bool = False
    metadata: dict = field(default_factory=dict)


class FrameBuffer:
    """Thread-safe circular buffer for frames.
    
    Features:
    - Fixed-size circular buffer to limit memory
    - Optional JPEG compression for storage efficiency
    - Thread-safe access for producer/consumer pattern
    - Frame lookup by timestamp or index
    """
    
    def __init__(
        self,
        max_size: int = 60,
        compress: bool = True,
        jpeg_quality: int = 85,
    ):
        self.max_size = max_size
        self.compress = compress
        self.jpeg_quality = jpeg_quality
        
        self._buffer: deque = deque(maxlen=max_size)
        self._lock = threading.RLock()
        self._frame_count = 0
        
        self._cv2_available = False
        try:
            import cv2
            self._cv2_available = True
        except ImportError:
            logger.warning("OpenCV not available, compression disabled")
            self.compress = False
    
    def add(self, image: np.ndarray, metadata: Optional[dict] = None) -> Frame:
        """Add a frame to the buffer."""
        with self._lock:
            timestamp = time.time()
            frame_number = self._frame_count
            self._frame_count += 1
            
            resolution = (image.shape[1], image.shape[0])  # width, height
            
            if self.compress and self._cv2_available:
                import cv2
                # Compress to JPEG bytes
                _, encoded = cv2.imencode(
                    '.jpg', image,
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                )
                stored_image = encoded.tobytes()
                compressed = True
            else:
                stored_image = image.copy()
                compressed = False
            
            frame = Frame(
                image=stored_image,
                timestamp=timestamp,
                frame_number=frame_number,
                resolution=resolution,
                compressed=compressed,
                metadata=metadata or {},
            )
            
            self._buffer.append(frame)
            return frame
    
    def get_latest(self, decompress: bool = True) -> Optional[Frame]:
        """Get the most recent frame."""
        with self._lock:
            if not self._buffer:
                return None
            
            frame = self._buffer[-1]
            
            if decompress and frame.compressed:
                return self._decompress_frame(frame)
            return frame
    
    def get_by_index(self, index: int, decompress: bool = True) -> Optional[Frame]:
        """Get frame by buffer index (0 = oldest)."""
        with self._lock:
            if index < 0 or index >= len(self._buffer):
                return None
            
            frame = self._buffer[index]
            
            if decompress and frame.compressed:
                return self._decompress_frame(frame)
            return frame
    
    def get_by_timestamp(
        self,
        timestamp: float,
        tolerance: float = 0.1,
        decompress: bool = True
    ) -> Optional[Frame]:
        """Get frame closest to given timestamp."""
        with self._lock:
            if not self._buffer:
                return None
            
            best_frame = None
            best_diff = float('inf')
            
            for frame in self._buffer:
                diff = abs(frame.timestamp - timestamp)
                if diff < best_diff and diff <= tolerance:
                    best_diff = diff
                    best_frame = frame
            
            if best_frame and decompress and best_frame.compressed:
                return self._decompress_frame(best_frame)
            return best_frame
    
    def get_range(
        self,
        start_time: float,
        end_time: float,
        decompress: bool = True
    ) -> List[Frame]:
        """Get all frames within time range."""
        with self._lock:
            frames = [
                f for f in self._buffer
                if start_time <= f.timestamp <= end_time
            ]
            
            if decompress:
                frames = [
                    self._decompress_frame(f) if f.compressed else f
                    for f in frames
                ]
            
            return frames
    
    def get_recent(self, n: int, decompress: bool = True) -> List[Frame]:
        """Get the N most recent frames."""
        with self._lock:
            frames = list(self._buffer)[-n:]
            
            if decompress:
                frames = [
                    self._decompress_frame(f) if f.compressed else f
                    for f in frames
                ]
            
            return frames
    
    def _decompress_frame(self, frame: Frame) -> Frame:
        """Decompress a JPEG-compressed frame."""
        if not frame.compressed:
            return frame
        
        import cv2
        image_array = np.frombuffer(frame.image, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        return Frame(
            image=image,
            timestamp=frame.timestamp,
            frame_number=frame.frame_number,
            resolution=frame.resolution,
            compressed=False,
            metadata=frame.metadata,
        )
    
    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()
    
    @property
    def size(self) -> int:
        """Current number of frames in buffer."""
        return len(self._buffer)
    
    @property
    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return len(self._buffer) >= self.max_size
    
    @property
    def memory_usage(self) -> int:
        """Estimate memory usage in bytes."""
        with self._lock:
            total = 0
            for frame in self._buffer:
                if frame.compressed:
                    total += len(frame.image)
                else:
                    total += frame.image.nbytes
            return total


class ScreenCapture:
    """High-performance screen capture for Minecraft.
    
    Features:
    - Multiple capture backends (mss, pyautogui, x11)
    - Automatic Minecraft window detection
    - Configurable FPS and resolution
    - Background capture thread
    - Frame buffer integration
    """
    
    def __init__(
        self,
        fps: int = 30,
        resolution: Tuple[int, int] = (1920, 1080),
        capture_method: str = "mss",
        buffer_size: int = 60,
        compress_buffer: bool = True,
        jpeg_quality: int = 85,
    ):
        self.fps = fps
        self.resolution = resolution
        self.capture_method = capture_method
        
        self.buffer = FrameBuffer(
            max_size=buffer_size,
            compress=compress_buffer,
            jpeg_quality=jpeg_quality,
        )
        
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._capture_lock = threading.Lock()
        
        self._mss = None
        self._capture_region: Optional[dict] = None
        self._frame_interval = 1.0 / fps
        
        self._stats = {
            'total_frames': 0,
            'dropped_frames': 0,
            'avg_capture_time': 0.0,
        }
        
        self._callbacks: List[Callable[[Frame], None]] = []
        
        self._initialize_capture()
    
    def _initialize_capture(self) -> None:
        """Initialize the capture backend."""
        if self.capture_method == "mss":
            self._initialize_mss()
        elif self.capture_method == "pyautogui":
            self._initialize_pyautogui()
        else:
            logger.warning(f"Unknown capture method: {self.capture_method}")
            self._initialize_mss()  # Fallback
    
    def _initialize_mss(self) -> None:
        """Initialize MSS (fast screen capture)."""
        try:
            import mss
            self._mss = mss.mss()
            
            # Find Minecraft window or use primary monitor
            minecraft_region = self._find_minecraft_window()
            if minecraft_region:
                self._capture_region = minecraft_region
            else:
                # Use primary monitor
                monitor = self._mss.monitors[1]
                self._capture_region = {
                    'left': monitor['left'],
                    'top': monitor['top'],
                    'width': min(self.resolution[0], monitor['width']),
                    'height': min(self.resolution[1], monitor['height']),
                }
            
            logger.info(f"MSS capture initialized: {self._capture_region}")
        except ImportError:
            logger.error("MSS not installed, capture disabled")
            self._mss = None
    
    def _initialize_pyautogui(self) -> None:
        """Initialize PyAutoGUI capture."""
        try:
            import pyautogui
            self._pyautogui = pyautogui
            logger.info("PyAutoGUI capture initialized")
        except ImportError:
            logger.error("PyAutoGUI not installed, falling back to MSS")
            self._initialize_mss()
    
    def _find_minecraft_window(self) -> Optional[dict]:
        """Try to find the Minecraft window."""
        try:
            # Try using wmctrl on Linux
            import subprocess
            result = subprocess.run(
                ['wmctrl', '-l', '-G'],
                capture_output=True, text=True
            )
            
            for line in result.stdout.splitlines():
                if 'minecraft' in line.lower():
                    parts = line.split()
                    if len(parts) >= 8:
                        return {
                            'left': int(parts[2]),
                            'top': int(parts[3]),
                            'width': int(parts[4]),
                            'height': int(parts[5]),
                        }
        except:
            pass
        
        return None
    
    def capture_frame(self) -> Optional[Frame]:
        """Capture a single frame."""
        if not self._mss and not hasattr(self, '_pyautogui'):
            return None
        
        start_time = time.time()
        
        try:
            if self._mss and self._capture_region:
                screenshot = self._mss.grab(self._capture_region)
                image = np.array(screenshot)
                # Convert BGRA to BGR
                image = image[:, :, :3]
            elif hasattr(self, '_pyautogui'):
                screenshot = self._pyautogui.screenshot()
                image = np.array(screenshot)
                # Convert RGB to BGR for OpenCV compatibility
                image = image[:, :, ::-1]
            else:
                return None
            
            # Resize if needed
            if image.shape[:2] != (self.resolution[1], self.resolution[0]):
                try:
                    import cv2
                    image = cv2.resize(image, self.resolution)
                except:
                    pass
            
            # Add to buffer
            frame = self.buffer.add(image, {
                'capture_time': time.time() - start_time,
            })
            
            # Update stats
            self._stats['total_frames'] += 1
            capture_time = time.time() - start_time
            self._stats['avg_capture_time'] = (
                0.9 * self._stats['avg_capture_time'] + 0.1 * capture_time
            )
            
            # Notify callbacks
            for callback in self._callbacks:
                try:
                    callback(frame)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
            
            return frame
            
        except Exception as e:
            logger.error(f"Capture error: {e}")
            self._stats['dropped_frames'] += 1
            return None
    
    def start(self) -> None:
        """Start background capture thread."""
        if self._running:
            return
        
        self._running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="ScreenCapture"
        )
        self._capture_thread.start()
        logger.info(f"Screen capture started at {self.fps} FPS")
    
    def stop(self) -> None:
        """Stop background capture."""
        self._running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None
        logger.info("Screen capture stopped")
    
    def _capture_loop(self) -> None:
        """Main capture loop."""
        next_capture = time.time()
        
        while self._running:
            current_time = time.time()
            
            if current_time >= next_capture:
                self.capture_frame()
                next_capture += self._frame_interval
                
                # If we're falling behind, reset timing
                if next_capture < current_time:
                    next_capture = current_time + self._frame_interval
                    self._stats['dropped_frames'] += 1
            else:
                # Sleep until next capture
                sleep_time = next_capture - current_time
                if sleep_time > 0.001:
                    time.sleep(sleep_time * 0.9)  # Sleep slightly less
    
    def add_callback(self, callback: Callable[[Frame], None]) -> None:
        """Add a callback for new frames."""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[Frame], None]) -> None:
        """Remove a frame callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def set_capture_region(
        self,
        left: int,
        top: int,
        width: int,
        height: int
    ) -> None:
        """Set the capture region manually."""
        self._capture_region = {
            'left': left,
            'top': top,
            'width': width,
            'height': height,
        }
        logger.info(f"Capture region set: {self._capture_region}")
    
    def get_stats(self) -> dict:
        """Get capture statistics."""
        return {
            **self._stats,
            'buffer_size': self.buffer.size,
            'buffer_memory': self.buffer.memory_usage,
            'running': self._running,
        }
    
    @property
    def latest_frame(self) -> Optional[Frame]:
        """Get the latest captured frame."""
        return self.buffer.get_latest()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()

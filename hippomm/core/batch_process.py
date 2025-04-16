import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from tqdm import tqdm
from .hippocampal_memory import HippocampalMemory
import yaml
import time
from skimage.metrics import structural_similarity as ssim
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import subprocess
import soundfile as sf
import re
from queue import Queue
from threading import Thread
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

def compute_frame_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Compute difference between two frames using multiple metrics for robustness
    
    Args:
        frame1: First frame
        frame2: Second frame
        
    Returns:
        Difference score (0-1, where 0 means identical)
    """
    # Convert to grayscale if needed
    if len(frame1.shape) == 3:
        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    else:
        frame1_gray = frame1
        
    if len(frame2.shape) == 3:
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    else:
        frame2_gray = frame2
    
    # Normalize frames to 0-1 range
    frame1_norm = frame1_gray.astype(float) / 255.0
    frame2_norm = frame2_gray.astype(float) / 255.0
    
    try:
        # Try SSIM first
        with np.errstate(divide='ignore', invalid='ignore'):
            score = ssim(frame1_norm, frame2_norm, data_range=1.0)
            if np.isfinite(score):  # Check if result is valid
                return 1.0 - score
    except Exception as e:
        logger.debug(f"SSIM calculation failed: {e}, falling back to MSE")
    
    # Fallback to MSE if SSIM fails or gives invalid results
    mse = np.mean((frame1_norm - frame2_norm) ** 2)
    
    # Normalize MSE to 0-1 range (assuming max possible MSE is 1.0 for normalized images)
    return min(1.0, mse)

def save_frame(frame: np.ndarray, frame_path: Path) -> bool:
    """
    Helper function to save a frame to disk
    
    Args:
        frame: Frame to save
        frame_path: Path to save the frame to
        
    Returns:
        bool: True if save was successful or frame already exists
    """
    try:
        # Check if frame already exists
        if frame_path.exists():
            return True
            
        # Ensure the frame is valid
        if frame is None or frame.size == 0:
            logger.error(f"Invalid frame data for {frame_path}")
            return False
            
        # Ensure parent directory exists
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the frame
        success = cv2.imwrite(str(frame_path), frame)
        
        if not success:
            logger.error(f"Failed to save frame to {frame_path}")
            return False
            
        # Verify the file was created
        if not frame_path.exists():
            logger.error(f"Frame file not found after save: {frame_path}")
            return False
            
        logger.debug(f"Successfully saved frame to {frame_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving frame to {frame_path}: {e}")
        return False

def extract_frames_from_video(
    video_path: str,
    storage_dir: Path,
    video_id: str,
    config: Dict[str, Any] = None,
    min_diff_threshold: float = 0.1,  # Minimum difference to consider a frame unique
    max_diff_threshold: float = 0.3,  # Maximum difference to force keyframe
    check_interval: int = 30  # Check every N frames for efficiency (30 = check every second for 30fps video)
) -> tuple[List[str], List[float], float]:
    """
    Extract frames from a video file using dynamic frame selection with parallel processing
    """
    if config is None:
        config = {}
    
    # Create frame storage directory
    frames_dir = storage_dir / 'frames' / video_id
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if metadata already exists
    metadata_file = frames_dir / 'metadata.yaml'
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata = yaml.safe_load(f)
                # Verify all frames exist
                all_frames_exist = all(Path(frame_path).exists() for frame_path in metadata.get('frame_paths', []))
                if all_frames_exist:
                    logger.info(f"Found existing complete frame extraction for video {video_id}, skipping")
                    return metadata['frame_paths'], metadata['frame_timestamps'], metadata['duration']
                else:
                    logger.warning(f"Found incomplete frame extraction for video {video_id}, reprocessing")
        except Exception as e:
            logger.warning(f"Error reading existing metadata for {video_id}, reprocessing: {e}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / video_fps
    
    logger.info(f"Processing video: {video_path}")
    logger.info(f"FPS: {video_fps}, Total frames: {total_frames}, Duration: {video_duration:.2f}s")
    
    logger.info(f"Frame storage directory: {frames_dir}")
    
    frame_paths = []
    frame_times = []
    last_saved_frame = None
    cumulative_diff = 0.0
    frames_since_last_save = 0
    last_save_time = 0.0
    failed_saves = 0
    
    # Initialize thread pool for parallel frame saving
    with ThreadPoolExecutor(max_workers=min(mp.cpu_count(), 4)) as executor:
        frame_save_futures = []
        
        frame_count = 0
        with tqdm(total=total_frames, desc="Extracting frames", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = frame_count / video_fps
                save_frame_flag = False
                
                # Always save first frame
                if last_saved_frame is None:
                    save_frame_flag = True
                # Ensure at least 1 second between saves
                elif current_time - last_save_time >= 1.0:
                    # Check frame difference
                    if frame_count % check_interval == 0:
                        diff = compute_frame_difference(frame, last_saved_frame)
                        cumulative_diff += diff
                        
                        # Save if significant change detected
                        if (diff > max_diff_threshold or
                            cumulative_diff > max_diff_threshold):
                            save_frame_flag = True
                
                if save_frame_flag:
                    # Create subdirectories by timestamp for better organization
                    timestamp_dir = frames_dir / f"t_{int(current_time):04d}"
                    timestamp_dir.mkdir(exist_ok=True)
                    
                    frame_path = timestamp_dir / f"frame_{frame_count:06d}.jpg"
                    
                    # Save frame directly (no threading for debugging)
                    if save_frame(frame.copy(), frame_path):
                        frame_paths.append(str(frame_path))
                        frame_times.append(current_time)
                        
                        # Update tracking variables
                        last_saved_frame = frame.copy()
                        cumulative_diff = 0.0
                        frames_since_last_save = 0
                        last_save_time = current_time
                        
                        logger.debug(f"Saved frame {len(frame_paths)} at time {current_time:.2f}s")
                    else:
                        failed_saves += 1
                        logger.warning(f"Failed to save frame at time {current_time:.2f}s")
                else:
                    frames_since_last_save += 1
                
                frame_count += 1
                pbar.update(1)
    
    cap.release()
    
    logger.info(f"Extracted {len(frame_paths)} frames, Failed saves: {failed_saves}")
    
    # Save frame metadata
    metadata = {
        'frame_count': len(frame_paths),
        'total_frames': total_frames,
        'video_fps': video_fps,
        'duration': video_duration,
        'extraction_params': {
            'min_diff_threshold': min_diff_threshold,
            'max_diff_threshold': max_diff_threshold,
            'check_interval': check_interval
        },
        'frame_timestamps': frame_times,  # Add timestamps to metadata
        'average_fps': len(frame_paths) / video_duration if video_duration > 0 else 0,
        'failed_saves': failed_saves,
        'frame_paths': frame_paths  # Add actual frame paths to metadata
    }
    
    metadata_file = frames_dir / 'metadata.yaml'
    with open(metadata_file, 'w') as f:
        yaml.dump(metadata, f)
    
    return frame_paths, frame_times, video_duration

def extract_audio_from_video(video_path: str, storage_dir: Path, video_id: str) -> tuple[Optional[np.ndarray], Optional[int]]:
    """
    Extract audio from a video file using ffmpeg and save to storage
    """
    try:
        # Create audio directory
        audio_dir = storage_dir / 'audio' / video_id
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        # Get video duration using ffprobe
        duration_cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        duration = float(subprocess.check_output(duration_cmd).decode().strip())
        
        # Temporary WAV file path
        temp_wav = audio_dir / 'temp_audio.wav'
        
        # Extract audio using ffmpeg
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit output
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',  # Mono
            str(temp_wav)
        ]
        
        # Run ffmpeg command
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.warning(f"FFmpeg audio extraction failed: {e.stderr}")
            return None, None
        
        # Check for silence using silencedetect filter
        silence_cmd = [
            'ffmpeg',
            '-v', 'error',
            '-i', str(temp_wav),
            '-af', 'silencedetect=n=-50dB:d=0.1',  # Detect silence below -50dB lasting more than 0.1s
            '-f', 'null',
            '-'
        ]
        
        silence_result = subprocess.run(silence_cmd, capture_output=True, text=True)
        
        # Parse silence detection output
        silence_starts = []
        silence_durations = []
        
        for line in silence_result.stderr.split('\n'):
            if 'silence_start' in line:
                silence_starts.append(float(line.split(':')[1].strip()))
            elif 'silence_duration' in line:
                silence_durations.append(float(line.split(':')[1].strip()))
        
        # Calculate total silence duration
        total_silence = sum(silence_durations) if silence_durations else 0
        
        # If the file is mostly silence (>90%), skip it
        if total_silence / duration > 0.9:
            logger.warning(f"Audio is mostly silence ({total_silence:.2f}s out of {duration:.2f}s), skipping")
            temp_wav.unlink()
            return None, None
        
        # Read the audio data using soundfile
        try:
            audio_data, sample_rate = sf.read(str(temp_wav))
            
            # Ensure audio data is 2D array (samples, channels)
            if len(audio_data.shape) == 1:
                audio_data = audio_data.reshape(-1, 1)
            
            # Save as numpy array
            audio_file = audio_dir / 'audio.npy'
            np.save(str(audio_file), audio_data)
            
            # Save audio metadata
            metadata = {
                'sample_rate': sample_rate,
                'duration': duration,
                'channels': audio_data.shape[1],
                'shape': audio_data.shape,
                'dtype': str(audio_data.dtype),
                'silence_analysis': {
                    'total_silence': total_silence,
                    'silence_segments': len(silence_starts),
                    'silence_percentage': (total_silence / duration) * 100
                }
            }
            
            metadata_file = audio_dir / 'metadata.yaml'
            with open(metadata_file, 'w') as f:
                yaml.dump(metadata, f)
            
            # Clean up temporary file
            temp_wav.unlink()
            
            return audio_data, sample_rate
            
        except Exception as e:
            logger.warning(f"Failed to read audio data: {e}")
            if temp_wav.exists():
                temp_wav.unlink()
            return None, None
            
    except Exception as e:
        logger.warning(f"Could not extract audio from video {video_path}: {e}")
        logger.exception("Full traceback:")
        # Clean up any temporary files
        temp_wav = audio_dir / 'temp_audio.wav'
        if temp_wav.exists():
            temp_wav.unlink()
    
    return None, None

def process_single_video(args: tuple) -> Dict[str, Any]:
    """
    Process a single video file (for multiprocessing)
    """
    video_file, memory_store_path, config = args
    try:
        video_id = video_file.stem
        logger.info(f"Processing video: {video_id}")
        
        # Create a ProcessPoolExecutor for parallel frame and audio extraction
        with ProcessPoolExecutor(max_workers=2) as executor:
            # Submit frame extraction task
            frames_future = executor.submit(
                extract_frames_from_video,
                str(video_file),
                memory_store_path,
                video_id,
                config
            )
            
            # Submit audio extraction task
            audio_future = executor.submit(
                extract_audio_from_video,
                str(video_file),
                memory_store_path,
                video_id
            )
            
            # Wait for both tasks to complete
            frame_paths, frame_times, duration = frames_future.result()
            audio_data, sample_rate = audio_future.result()
        
        return {
            "success": True,
            "video_id": video_id,
            "metadata": {
                "path": str(video_file),
                "duration": duration,
                "processed_timestamp": time.time(),
                "frame_count": len(frame_paths),
                "storage_dir": str(memory_store_path / 'frames' / video_id)
            },
            "frames": frame_paths,
            "frame_times": frame_times,
            "audio_data": audio_data,
            "sample_rate": sample_rate
        }
        
    except Exception as e:
        logger.error(f"Failed to process video {video_file}: {e}")
        logger.exception("Full traceback:")
        return {
            "success": False,
            "video_id": video_file.stem,
            "error": str(e)
        }

def process_video_folder(
    folder_path: str,
    memory_system: HippocampalMemory,
    config: Dict[str, Any],
    file_extensions: List[str] = ['.mp4', '.avi', '.mov', '.mkv'],
    skip_existing: bool = True,
    memory_store: Optional[str] = None,
    checkpoint_interval: int = 5,  # Save checkpoint every N videos
    sort_by: str = "name"  # Options: "name", "time", "size"
) -> Dict[str, Any]:
    """
    Process all videos in a folder using parallel processing
    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise ValueError(f"Folder not found: {folder_path}")
    
    # Update memory store location if provided
    if memory_store:
        config['storage']['base_dir'] = memory_store
        memory_store_path = Path(memory_store)
        memory_store_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using memory store location: {memory_store}")
    else:
        memory_store_path = Path(config['storage']['base_dir'])
        memory_store_path.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoint directory
    checkpoint_dir = memory_store_path / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Get list of video files
    video_files = []
    for ext in file_extensions:
        video_files.extend(folder_path.glob(f"*{ext}"))
    
    if not video_files:
        logger.warning(f"No video files found in {folder_path}")
        return {"processed": 0, "skipped": 0, "failed": 0}
    
    # Sort video files based on specified criterion
    if sort_by == "name":
        video_files.sort(key=lambda x: x.name)
    elif sort_by == "time":
        video_files.sort(key=lambda x: x.stat().st_mtime)  # Sort by modification time
    elif sort_by == "size":
        video_files.sort(key=lambda x: x.stat().st_size)
    else:
        logger.warning(f"Unknown sort criterion: {sort_by}, defaulting to name")
        video_files.sort(key=lambda x: x.name)
    
    # Filter out already processed videos
    if skip_existing:
        video_files = [v for v in video_files if v.stem not in memory_system.video_index]
    
    stats = {
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "videos": []  # Detailed stats for each video
    }
    
    # Log sorting information
    logger.info("=" * 80)
    logger.info(f"Starting batch processing of {len(video_files)} videos")
    logger.info(f"Sort criterion: {sort_by}")
    logger.info("=" * 80)
    
    for video_file in video_files:
        if sort_by == "time":
            logger.info(f"Queued: {video_file.name} (Modified: {time.ctime(video_file.stat().st_mtime)})")
        elif sort_by == "size":
            logger.info(f"Queued: {video_file.name} (Size: {video_file.stat().st_size / (1024*1024):.2f} MB)")
        else:
            logger.info(f"Queued: {video_file.name}")
    
    # Process videos sequentially for better control and immediate consolidation
    for i, video_file in enumerate(video_files, 1):
        logger.info("\n" + "=" * 80)
        logger.info(f"Processing video {i}/{len(video_files)}: {video_file.name}")
        logger.info("-" * 80)
        
        # Check if video already has theta events
        video_id = video_file.stem
        video_events_dir = memory_store_path / 'events' / video_id
        if video_events_dir.exists() and any(video_events_dir.glob('*.json')):
            logger.info(f"Video {video_id} already has theta events, skipping...")
            stats["skipped"] += 1
            stats["videos"].append({
                "name": video_file.name,
                "video_id": video_id,
                "status": "skipped",
                "reason": "existing theta events"
            })
            continue
            
        start_time = time.time()
        
        try:
            # Process single video
            result = process_single_video((video_file, memory_store_path, config))
            
            if result["success"]:
                # Add to memory system
                memory_system.add_video(result["video_id"], result["metadata"])
                
                # Process the sequence
                memory_system.process_sequence(
                    video_id=result["video_id"],
                    video_frames=result["frames"],
                    frame_times=result["frame_times"],
                    audio_data=result["audio_data"],
                    audio_sample_rate=result["sample_rate"]
                )
                # Immediately consolidate memories for this video
                logger.info(f"Consolidating memories for video: {result['video_id']}")
               
                memories = memory_system.short_term_buffer[result["video_id"]]
                if not memories:
                    print("No memories")
                    return
                    
                # Consolidate memories
                consolidated_events = memory_system.consolidate(memories)
                # Process through replay
                memory_system.replay(consolidated_events, video_id=result["video_id"])
                
                # Clear the video's short-term buffer
                del memory_system.short_term_buffer[result["video_id"]]
                
                # Update stats
                stats["processed"] += 1
                processing_time = time.time() - start_time
                
                # Log detailed information
                logger.info("-" * 80)
                logger.info(f"Successfully processed: {video_file.name}")
                logger.info(f"Video ID: {result['video_id']}")
                logger.info(f"Duration: {result['metadata']['duration']:.2f} seconds")
                logger.info(f"Frames extracted: {result['metadata']['frame_count']}")
                logger.info(f"Processing time: {processing_time:.2f} seconds")
                logger.info(f"Storage directory: {result['metadata']['storage_dir']}")
                
                # Add detailed stats
                stats["videos"].append({
                    "name": video_file.name,
                    "video_id": result["video_id"],
                    "duration": result["metadata"]["duration"],
                    "frame_count": result["metadata"]["frame_count"],
                    "processing_time": processing_time,
                    "success": True
                })
            else:
                stats["failed"] += 1
                logger.error(f"Failed to process video {video_file.name}: {result['error']}")
                stats["videos"].append({
                    "name": video_file.name,
                    "error": str(result["error"]),
                    "success": False
                })
            
            # Create checkpoint if needed
            if i % checkpoint_interval == 0:
                try:
                    # Save short-term memory state
                    short_term_paths = memory_system.save_short_term_buffer(
                        temp_dir=str(checkpoint_dir / 'temp_short_term')
                    )
                    
                    # Save checkpoint data
                    checkpoint_data = {
                        'timestamp': time.time(),
                        'stats': stats,
                        'processed_videos': [v["video_id"] for v in stats["videos"] if v["success"]],
                        'short_term_paths': short_term_paths
                    }
                    
                    checkpoint_file = checkpoint_dir / f'checkpoint_{int(time.time())}.json'
                    with open(checkpoint_file, 'w') as f:
                        json.dump(checkpoint_data, f, indent=2)
                        
                    logger.info(f"Created checkpoint: {checkpoint_file}")
                    
                    # Clean up old checkpoints (keep last 3)
                    old_checkpoints = sorted(checkpoint_dir.glob('checkpoint_*.json'))[:-3]
                    for old_cp in old_checkpoints:
                        old_cp.unlink()
                        
                except Exception as e:
                    logger.error(f"Failed to create checkpoint: {e}")
                    logger.exception("Full traceback:")
                    
        except Exception as e:
            stats["failed"] += 1
            logger.error(f"Unexpected error processing video {video_file.name}: {e}")
            logger.exception("Full traceback:")
            stats["videos"].append({
                "name": video_file.name,
                "error": str(e),
                "success": False
            })
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("Processing Summary:")
    logger.info(f"Total videos processed: {stats['processed']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Skipped: {stats['skipped']}")
    logger.info("=" * 80)
    
    # Log detailed stats for each video
    logger.info("\nDetailed Processing Results:")
    for video_stat in stats["videos"]:
        # Add a default success value if not present
        success = video_stat.get("success", False)
        
        if success:
            logger.info(f"\nVideo: {video_stat['name']}")
            logger.info(f"- Video ID: {video_stat.get('video_id', 'Unknown')}")
            logger.info(f"- Duration: {video_stat.get('duration', 0.0):.2f} seconds")
            logger.info(f"- Frames: {video_stat.get('frame_count', 0)}")
            logger.info(f"- Processing Time: {video_stat.get('processing_time', 0.0):.2f} seconds")
        else:
            logger.info(f"\nVideo: {video_stat['name']}")
            logger.info(f"- Status: Failed")
            logger.info(f"- Error: {video_stat.get('error', 'Unknown error')}")
    
    return stats

def process_memory_sync(
    memory_system: HippocampalMemory,
    frame_queue: Queue,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_interval: int = 100  # save checkpoint every n frames
) -> None:
    """process frames through memory system synchronously"""
    processed_frames = 0
    
    while True:
        # get item from queue
        item = frame_queue.get()
        if item is None:  # end signal
            # save final checkpoint before consolidation
            if checkpoint_dir:
                try:
                    short_term_paths = memory_system.save_short_term_buffer(
                        temp_dir=str(checkpoint_dir / 'temp_short_term')
                    )
                    checkpoint_data = {
                        'timestamp': time.time(),
                        'processed_frames': processed_frames,
                        'short_term_paths': short_term_paths
                    }
                    # make dir
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint_file = checkpoint_dir / f'checkpoint_final_{int(time.time())}.json'
                    with open(checkpoint_file, 'w') as f:
                        json.dump(checkpoint_data, f, indent=2)
                except Exception as e:
                    logger.error(f"failed to create final checkpoint: {e}")
            break
        
        if item["type"] == "frame":
            # process single frame
            memory_system.add_single_frame(
                item["frame_path"],
                item["frame_time"],
                item["video_id"]
            )
            processed_frames += 1
            
            # create checkpoint if needed
            if checkpoint_dir and processed_frames % checkpoint_interval == 0:
                try:
                    short_term_paths = memory_system.save_short_term_buffer(
                        temp_dir=str(checkpoint_dir / 'temp_short_term')
                    )
                    checkpoint_data = {
                        'timestamp': time.time(),
                        'processed_frames': processed_frames,
                        'short_term_paths': short_term_paths
                    }
                    checkpoint_file = checkpoint_dir / f'checkpoint_{int(time.time())}.json'
                    with open(checkpoint_file, 'w') as f:
                        json.dump(checkpoint_data, f, indent=2)
                        
                    # clean up old checkpoints (keep last 3)
                    old_checkpoints = sorted(checkpoint_dir.glob('checkpoint_*.json'))[:-3]
                    for old_cp in old_checkpoints:
                        old_cp.unlink()
                except Exception as e:
                    logger.error(f"failed to create checkpoint: {e}")
            
        elif item["type"] == "complete":
            # consolidate all memories for this video synchronously
            memory_system.consolidate_video_memories()
            
            # add video metadata
            memory_system.add_video(item["video_id"], item["metadata"])
            
            # process audio if available
            if item["audio_data"] is not None:
                memory_system.process_sequence(
                    audio_data=item["audio_data"],
                    audio_sample_rate=item["sample_rate"]
                )
            
            logger.info(f"completed processing video {item['video_id']}")
            
        elif item["type"] == "error":
            logger.error(f"error processing video {item['video_id']}: {item['error']}")

def main():
    """
    example usage of batch processing
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="batch process videos using hippocampalmemory")
    parser.add_argument("path", help="path to video file or folder containing videos")
    parser.add_argument("--config", help="path to config file", default="config/default_config.yaml")
    parser.add_argument("--skip-existing", action="store_true", help="skip already processed videos")
    parser.add_argument("--memory-store", help="path to store memory files (overrides config if provided)")
    parser.add_argument("--checkpoint-interval", type=int, help="save checkpoint every n videos/frames", default=5)
    parser.add_argument("--sort-by", choices=["name", "time", "size"], default="name",
                      help="how to sort videos: by name, modification time, or file size")
    args = parser.parse_args()
    
    # load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # update memory store location in config if provided
    if args.memory_store:
        config['storage']['base_dir'] = args.memory_store
        memory_store_path = Path(args.memory_store)
        memory_store_path.mkdir(parents=True, exist_ok=True)
        print(f"using memory store location: {args.memory_store}")
    else:
        memory_store_path = Path(config['storage']['base_dir'])
        memory_store_path.mkdir(parents=True, exist_ok=True)
        print(f"using default memory store location: {config['storage']['base_dir']}")
    
    # initialize memory system
    memory_system = HippocampalMemory(config)
    
    video_path = Path(args.path)
    if video_path.is_file():
        # process single video synchronously
        print(f"\nprocessing video synchronously: {video_path}")
        result = process_single_video((video_path, memory_store_path, config))
        if result["success"]:
            memory_system.add_video(result["video_id"], result["metadata"])
            memory_system.process_sequence(
                video_frames=result["frames"],
                frame_times=result["frame_times"],
                audio_data=result["audio_data"],
                audio_sample_rate=result["sample_rate"]
            )
            # consolidate memories for this video
            print(f"consolidating memories for video: {result['video_id']}")
            print("\nprocessing complete!")
            print(f"video id: {result['video_id']}")
            print(f"duration: {result['metadata']['duration']:.2f} seconds")
            print(f"frames extracted: {result['metadata']['frame_count']}")
            print(f"storage directory: {result['metadata']['storage_dir']}")
        else:
            print(f"\nfailed to process video: {result['error']}")
    else:
        # process folder
        print(f"\nprocessing video folder: {video_path}")
        print(f"sorting videos by: {args.sort_by}")
        stats = process_video_folder(
            args.path,
            memory_system,
            config=config,
            skip_existing=args.skip_existing,
            memory_store=args.memory_store,
            checkpoint_interval=args.checkpoint_interval,
            sort_by=args.sort_by
        )
        
        print("\nprocessing complete!")
        print(f"processed: {stats['processed']}")
        print(f"skipped: {stats['skipped']}")
        print(f"failed: {stats['failed']}")
        print(f"memory store location: {config['storage']['base_dir']}")

if __name__ == "__main__":
    main() 
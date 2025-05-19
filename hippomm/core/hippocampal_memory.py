import base64
import json
import logging
import multiprocessing
import os
import subprocess
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import librosa
import numpy as np
import soundfile as sf
import torch
from openai import OpenAI
from scipy import signal
from scipy.io import wavfile
from skimage.metrics import structural_similarity as ssim
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from ..models.foundation_models import ImageBind, QwenVL, Whisper
from ..utils.vector_ops import cosine_similarity, top_k_cosine_similarity
from token_count import TokenCount

# configure logger
logger = logging.getLogger(__name__)


@dataclass
class SequenceSegment:
    """Represents a segment of video/audio sequence."""
    start_time: float  # original video/audio start time
    end_time: float    # original video/audio end time
    frames: Optional[List[str]] = None
    audio_data: Optional[np.ndarray] = None
    frame_times: Optional[List[float]] = None  # original timestamps of frames in source video


@dataclass
class ShortTermMemory:
    """Represents a short-term memory entry with multimodal features."""
    features: Dict[str, np.ndarray]  # mapping of modality to features
    content: Dict[str, Any]  # raw content (frames, audio, text)
    timestamp: float  # processing timestamp
    source_time: float  # original time in source video/audio
    modalities: List[str]
    segment_info: SequenceSegment
    transcription: List[Dict[str, Any]]  # changed from List[Dict[str, Any, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory object to dictionary for serialization."""
        # convert features to list format
        features_dict = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                         for k, v in self.features.items()}
        
        # handle content dictionary
        content_dict = {}
        for k, v in self.content.items():
            if k == 'audio' and isinstance(v, dict):
                # handle audio data specially
                audio_dict = v.copy()
                if 'data' in audio_dict and isinstance(audio_dict['data'], np.ndarray):
                    audio_dict['data'] = audio_dict['data'].tolist()
                content_dict[k] = audio_dict
            else:
                content_dict[k] = v
        
        # convert segment info
        segment_dict = {
            'start_time': self.segment_info.start_time,
            'end_time': self.segment_info.end_time,
            'frames': self.segment_info.frames,
            'frame_times': self.segment_info.frame_times
        }
        if self.segment_info.audio_data is not None:
            segment_dict['audio_data'] = self.segment_info.audio_data.tolist()
        
        return {
            'features': features_dict,
            'content': content_dict,
            'timestamp': self.timestamp,
            'source_time': self.source_time,
            'modalities': self.modalities,
            'segment_info': segment_dict,
            'transcription': self.transcription
        }


@dataclass
class ThetaEvent:
    """Represents a consolidated event memory during theta rhythm."""
    features: Dict[str, np.ndarray]  # consolidated features by modality
    feature_times: Optional[Dict[str, np.ndarray]]  # consolidated feature times by modality
    frames: List[str]  # list of frame paths
    frame_times: List[float]  # timestamps of key frames
    frame_captions: List[str]  # captions for each frame
    audio_times: List[float]  # timestamps of audio segments
    audio_transcription: List[Dict[str, Any]]  # transcription for each audio segment
    holistic_audio_transcription: List[Dict[str, Any]]  # transcription for entire audio segment
    summary: str  # event summary from VLM
    start_time: float
    end_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event object to dictionary for serialization."""
        features_dict = {}
        times_dict = {}
        
        for modality, features in self.features.items():
            if modality.endswith('_times'):
                times_dict[modality] = features.tolist()
            else:
                features_dict[modality] = features.tolist()
        
        return {
            'features': features_dict,
            'feature_times': times_dict,
            'frames': self.frames,
            'frame_times': self.frame_times,
            'frame_captions': self.frame_captions,
            'audio_times': self.audio_times,
            'audio_transcription': self.audio_transcription,
            'holistic_audio_transcription': self.holistic_audio_transcription,
            'summary': self.summary,
            'start_time': self.start_time,
            'end_time': self.end_time,
        }


@dataclass
class QARecallResult:
    """Represents a result from the QA-based recall system."""
    answer: str  # the final answer to the question
    confidence: float  # confidence score (0-1)
    reasoning: str  # explanation of the reasoning process
    retrieved_segments: Optional[List[SequenceSegment]] = None  # retrieved segments if needed deeper analysis
    question_type: str = "unknown"  # type of question (VIDEO, AUDIO, VIDEO+AUDIO)
    used_direct_answer: bool = False  # whether answer was derived directly from summaries
    used_corner_case: bool = False  # whether corner case handling was used
    primary_modality: str = "unknown"  # primary modality used for processing
    segments_analyzed: int = 0  # number of segments analyzed
    used_reflection: bool = False  # whether reflection was used


# module level function for multiprocessing
def process_frame_with_api(frame, index, model_name, config: Dict[str, Any] = None):
    """Process a single frame with direct API calls."""
    try:
        # check if file exists
        if not os.path.exists(frame):
            return index, f"[Error: Image file not found: {frame}]"
        
        # proceed with base64 encoding
        with open(frame, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        
        # create message with a more specific prompt
        formatted_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe what you see in this image in a detailed, specific sentence. Focus on the main subjects, activities, and environment."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
                ]
            }
        ]
        
        # get API configuration
        if config is None:
            config = {}
        api_config = config.get('api', {}).get('frame_processing', {})
        base_urls = api_config.get('base_urls', ["http://localhost:8000/v1", "http://localhost:9000/v1"])
        api_key = api_config.get('api_key','')
        
        # choose API endpoint based on index (even/odd split)
        base_url = base_urls[index % len(base_urls)]
        
        # create a new client for each worker
        client = OpenAI(base_url=base_url, api_key=api_key)
        
        # call the API directly
        response = client.chat.completions.create(
            model=model_name,
            messages=formatted_messages,
            temperature=0
        )
        
        # extract the caption from the response
        caption = response.choices[0].message.content
        
        # return with index to maintain order
        return index, f"Frame {index+1}: {caption}"
        
    except Exception as e:
        logger.error(f"Error processing image {frame}: {e}")
        logger.exception("Full traceback:")
        return index, f"[Error processing image {frame}]"


class HippocampalMemory:
    """Hippocampus-inspired memory system with short-term and long-term components."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        imagebind_path: str = None,
        whisper_model: str = None,
        qwen_path: str = None,
    ):
        """Initialize the hippocampal memory system.
        
        Args:
            config: Configuration dictionary
            imagebind_path: Optional override for ImageBind model path
            whisper_model: Optional override for Whisper model name
            qwen_path: Optional override for Qwen VL model path
        """
        self.config = config
        
        # Get model paths from config, with optional overrides
        self.imagebind_path = imagebind_path or config['models']['imagebind_path']
        self.whisper_model = whisper_model or config['models']['whisper_model']
        self.qwen_path = qwen_path or config['models']['qwen_path']
        
        # initialize foundation models
        print("Initializing HippocampalMemory system...")
        print(f"Loading ImageBind from: {self.imagebind_path}")
        self.imagebind = ImageBind(model_path=self.imagebind_path)
        print("ImageBind loaded successfully")
        
        print(f"Loading Whisper model: {self.whisper_model}")
        self.whisper = Whisper(model_size=self.whisper_model)
        print("Whisper loaded successfully")
        
        print(f"Loading QwenVL from: {self.qwen_path}")
        self.qwen = QwenVL(model_name=self.qwen_path, config=config)
        print("QwenVL loaded successfully")
        
        # memory parameters
        self.max_short_term = config['memory'].get('max_short_term', 10)
        self.max_long_term = config['memory'].get('max_long_term', 100)
        
        # frame buffer parameters
        self.frame_buffer_size = config['processing'].get('frame_buffer_size', 32)  # process frames in batches
        self.frame_buffer: Dict[str, List[Tuple[str, float]]] = defaultdict(list)  # video_id -> [(frame_path, time)]
        self.frame_features: Dict[str, List[np.ndarray]] = defaultdict(list)  # video_id -> [features]
        
        # sequence processing parameters
        self.max_segment_duration = config['processing'].get('max_segment_duration', 10.0)
        self.min_segment_duration = config['processing'].get('min_segment_duration', 5.0)
        self.frame_similarity_threshold = config['processing'].get('frame_similarity_threshold', 0.95)
        self.audio_silence_threshold = config['processing'].get('audio_silence_threshold', -40)  # dB
        
        # initialize memory stores
        self.short_term_buffer: Dict[str, List[ShortTermMemory]] = defaultdict(list)  # video_id -> [memories]
        self.long_term_store: List[ThetaEvent] = []
        
        # enhanced storage configuration
        self.storage_dir = Path(config['storage'].get('base_dir', 'memory_store'))
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # create subdirectories for different types of data
        self.videos_dir = self.storage_dir / 'videos'
        self.events_dir = self.storage_dir / 'events'
        self.videos_dir.mkdir(exist_ok=True)
        self.events_dir.mkdir(exist_ok=True)
        
        # index files for quick lookup
        self.video_index_file = self.storage_dir / 'video_index.json'
        self.event_index_file = self.storage_dir / 'event_index.json'
        
        # load indices
        self.video_index = self._load_index(self.video_index_file)
        self.event_index = self._load_index(self.event_index_file)
        
        print("\nMemory System Configuration:")
        print(f"- Max Short-term Memories: {self.max_short_term}")
        print(f"- Max Long-term Memories: {self.max_long_term}")
        print(f"- Frame Buffer Size: {self.frame_buffer_size} frames")
        print("\nHippocampalMemory system initialized successfully!")
        
    def _load_index(self, index_file: Path) -> Dict[str, Dict[str, Any]]:
        """Load or create an index file."""
        if index_file.exists():
            with open(index_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_index(self, index_file: Path, index_data: Dict[str, Dict[str, Any]]) -> None:
        """Save index data to file."""
        with open(index_file, 'w') as f:
            json.dump(index_data, f, indent=2)

    def _numpy_to_base64(self, arr: np.ndarray) -> str:
        """Convert numpy array to base64 string."""
        return base64.b64encode(arr.tobytes()).decode('utf-8')

    def _base64_to_numpy(self, b64_str: str, dtype=np.float32, shape=None) -> np.ndarray:
        """Convert base64 string back to numpy array."""
        data = base64.b64decode(b64_str)
        if shape is None:
            # if shape is not provided, assume it's a 1D array
            return np.frombuffer(data, dtype=dtype)
        return np.frombuffer(data, dtype=dtype).reshape(shape)

    def save_theta_event(self, event: ThetaEvent, video_id: str) -> None:
        """Save a theta event to the JSON storage with video organization."""
        try:
            event_id = f"{video_id}_{int(event.start_time * 1000)}"
            
            # create video-specific directory
            video_dir = self.events_dir / video_id
            video_dir.mkdir(parents=True, exist_ok=True)
            
            # prepare event data
            event_data = event.to_dict()  # use modified to_dict method
            
            # save event to file
            event_file = video_dir / f"{event_id}.json"
            with open(event_file, 'w') as f:
                json.dump(event_data, f, indent=2)
            
            # update event index
            self.event_index[event_id] = {
                'video_id': video_id,
                'start_time': event.start_time,
                'end_time': event.end_time,
                'file_path': str(event_file)
            }
            
            # save updated index
            self._save_index(self.event_index_file, self.event_index)
            
            print(f"Saved event {event_id} to {event_file}")
            
        except Exception as e:
            logger.error(f"Error saving theta event for video {video_id}: {e}")
            logger.exception("Full traceback:")
            raise
        
    def load_theta_event(self, event_id: str) -> Optional[ThetaEvent]:
        """Load a theta event from its individual JSON file."""
        if event_id not in self.event_index:
            return None
        
        event_info = self.event_index[event_id]
        event_file = Path(event_info['file_path'])
        
        if not event_file.exists():
            logger.warning(f"Event file not found: {event_file}")
            return None

        # load event data
        with open(event_file, 'r') as f:
            event_data = json.load(f)
        
        try:
            # handle both old and new event formats
            features = {}
            feature_times = {}
            
            # first try new format with separate feature_times
            if 'feature_times' in event_data:
                # handle feature times
                for modality, times_list in event_data['feature_times'].items():
                    try:
                        feature_times[modality] = np.array(times_list)
                    except Exception as e:
                        logger.error(f"Error converting times for modality {modality}: {e}")
                        continue
                
                # handle main features
                for modality, feature_list in event_data['features'].items():
                    try:
                        if isinstance(feature_list, list):
                            if all(isinstance(x, list) for x in feature_list):
                                features[modality] = np.array(feature_list)
                            else:
                                features[modality] = np.array(feature_list)
                        else:
                            features[modality] = np.array(feature_list)
                    except Exception as e:
                        logger.error(f"Error converting features for modality {modality}: {e}")
                        continue
            else:
                # handle old format where times were mixed with features
                for modality, data in event_data['features'].items():
                    try:
                        if isinstance(data, dict):
                            # if it's a dict with 'features' and 'times', handle separately
                            if 'features' in data:
                                features[modality] = np.array(data['features'])
                            if 'times' in data:
                                feature_times[modality] = np.array(data['times'])
                        else:
                            # if it's just features, convert directly
                            features[modality] = np.array(data)
                    except Exception as e:
                        logger.error(f"Error converting data for modality {modality}: {e}")
                        continue
            
            # verify feature dimensions
            for modality, feature_data in features.items():
                if len(feature_data.shape) > 1 and feature_data.shape[1] != 1024:
                    logger.warning(f"Feature dimension mismatch for {modality}: {feature_data.shape[1]}, expected 1024")
                    # try to fix if possible
                    if len(feature_data.shape) > 1:
                        if feature_data.shape[0] == 1024:
                            features[modality] = feature_data.T
                        else:
                            logger.error(f"Cannot fix feature dimension for {modality}")
                            return None
            
            # create ThetaEvent object
            event = ThetaEvent(
                features=features,
                feature_times=feature_times,
                frames=event_data.get('frames', []),
                frame_times=event_data.get('frame_times', []),
                frame_captions=event_data.get('frame_captions', []),
                audio_times=event_data.get('audio_times', []),
                audio_transcription=event_data.get('audio_transcription', []),
                holistic_audio_transcription=event_data.get('holistic_audio_transcription', []),
                summary=event_data.get('summary', ''),
                start_time=event_data.get('start_time', 0.0),
                end_time=event_data.get('end_time', 0.0)
            )
            
            self.long_term_store.append(event)
            return event
            
        except Exception as e:
            logger.error(f"Error creating ThetaEvent: {e}")
            logger.exception("Full traceback:")
            return None

    def add_memory(
        self,
        video_frames: Optional[List[str]] = None,
        frame_times: Optional[List[float]] = None,  # original timestamps from source video
        audio_data: Optional[np.ndarray] = None,
        audio_sample_rate: Optional[int] = None,
        source_time: Optional[float] = None  # time in original video/audio
    ) -> None:
        """Add new memory to short-term buffer."""
        if source_time is None and frame_times:
            source_time = frame_times[0]
        elif source_time is None:
            source_time = time.time()
            
        features = {}
        content = {}
        modalities = []
        
        # process video frames with ImageBind
        if video_frames:
            if not frame_times:
                raise ValueError("frame_times must be provided when processing video frames")
                
            vision_features = self.imagebind.extract_features(
                {'vision': video_frames},
                ['vision']
            )['vision']
            # move tensor to CPU and convert to numpy
            if isinstance(vision_features, torch.Tensor):
                vision_features = vision_features.detach().cpu().numpy()
                
            # ensure features have correct dimension (1024)
            if len(vision_features.shape) > 1:
                if vision_features.shape[1] != 1024:
                    logger.error(f"Vision features have incorrect dimension: {vision_features.shape[1]}, expected 1024")
                    return
                features['vision'] = vision_features
                content['frames'] = video_frames
                content['frame_times'] = frame_times
                modalities.append('vision')
            
        # process audio with ImageBind and Whisper
        if audio_data is not None:
            # get audio features
            audio_features = self.imagebind.extract_features(
                {'audio': [audio_data]},
                ['audio']
            )['audio']
            # move tensor to CPU and convert to numpy
            if isinstance(audio_features, torch.Tensor):
                audio_features = audio_features.detach().cpu().numpy()
                
            # ensure features have correct dimension (1024)
            if len(audio_features.shape) > 1:
                if audio_features.shape[1] != 1024:
                    logger.error(f"Audio features have incorrect dimension: {audio_features.shape[1]}, expected 1024")
                    return
                features['audio'] = audio_features
                
                # get audio transcription
                transcription = self.whisper.transcribe(audio_data)
                content['audio'] = {
                    'data': audio_data,
                    'transcription': transcription,
                    'start_time': source_time
                }
                modalities.append('audio')
            
        # create short-term memory
        memory = ShortTermMemory(
            features=features,
            content=content,
            timestamp=time.time(),  # current processing time
            source_time=source_time,  # time in original video/audio
            modalities=modalities,
            segment_info=SequenceSegment(
                start_time=source_time,
                end_time=(frame_times[-1] if frame_times else 
                         source_time + (len(video_frames)/30.0 if video_frames else 
                         0.0 if audio_data is None else len(audio_data)/audio_sample_rate)),
                frames=video_frames,
                frame_times=frame_times
            ),
            transcription=[]  # initialize empty transcription
        )
        
        # add to buffer
        self.short_term_buffer[video_frames[0]].append(memory)

    def consolidate(self, memories: List[ShortTermMemory]) -> List[Dict[str, Any]]:
        """
        Consolidate short-term memories without clustering.
        This function combines all memories in the buffer into a single event.
        
        Args:
            memories: List of ShortTermMemory objects to consolidate
            
        Returns:
            List containing a single consolidated event dictionary
        """
        logger.info(f"Starting consolidation of {len(memories)} memories")
        if not memories:
            logger.warning("No memories provided for consolidation")
            return []
        
        try:
            # simply combine all memories
            consolidated_memory = self.consolidate_short_term_memory(memories)
            if consolidated_memory:
                logger.info("Successfully consolidated memory")
                # create event dictionary with proper structure
                event_dict = {
                    'features': consolidated_memory.features,
                    'content': {
                        'frames': consolidated_memory.content.get('frames', []),
                        'frame_times': consolidated_memory.content.get('frame_times', []),
                        'audio_times': consolidated_memory.content.get('audio_times', []),
                        'transcription': consolidated_memory.content.get('transcription', []),
                        'holistic_audio_transcription': consolidated_memory.content.get('holistic_audio_transcription', []),
                        'segment_info': {
                            'start_time': consolidated_memory.segment_info.start_time,
                            'end_time': consolidated_memory.segment_info.end_time,
                            'frames': consolidated_memory.segment_info.frames,
                            'frame_times': consolidated_memory.segment_info.frame_times
                        }
                    }
                }
                logger.info(f"Created event dict with {len(event_dict['content']['frames'])} frames")
                return [event_dict]  # return as a list with one event
            else:
                logger.warning("Failed to consolidate memory - no valid features found")
                return []
        except Exception as e:
            logger.error(f"Error during memory consolidation: {e}")
            logger.exception("Full traceback:")
            return []

    def replay(self, event: Union[Dict[str, Any], List[Dict[str, Any]]], video_id: str) -> None:
        """
        Replay consolidated events through VLM to generate semantic understanding
        and store them in long-term memory.
        
        Args:
            event: Consolidated event dictionary from consolidate() or list of events
            video_id: ID of the video being processed
        """
        logger.info(f"Starting replay for video {video_id}")
        
        # handle both single event and list of events
        if isinstance(event, list):
            if not event:
                logger.warning(f"Empty event list received for video {video_id}")
                return
            event = event[0]  # take the first event
            logger.info("Extracted first event from list")
            
        logger.debug(f"Event type: {type(event)}")
        logger.debug(f"Event content: {event}")
        
        if not event or not isinstance(event, dict):
            logger.error(f"Invalid event data received for video {video_id}")
            return
        
        frame_captions_results = []
        
        # step 1: process frames if available
        content = event.get('content', {})
        frames = content.get('frames', [])
        has_frames = bool(frames)
        has_audio = bool(content.get('audio_times', []))
        
        if not has_frames and not has_audio:
            logger.warning(f"No valid content found in event for video {video_id}")
            return
        
        if has_frames:
            # create list of arguments for each frame
            frame_index_model_triples = [
                (frame, i, self.qwen.model_name) 
                for i, frame in enumerate(frames)
            ]
            
            # initialize multiprocessing pool
            num_workers = min(multiprocessing.cpu_count(), 8)
            
            # process frames in parallel with progress bar
            with multiprocessing.Pool(processes=num_workers) as pool:
                frame_captions_results = list(tqdm(
                    pool.starmap(process_frame_with_api, frame_index_model_triples),
                    total=len(frame_index_model_triples),
                    desc="Processing frames",
                    unit="frame"
                ))
            
            # sort results by index and extract captions
            frame_captions_results.sort(key=lambda x: x[0])
            frame_captions = [caption for _, caption in frame_captions_results]
            
            # show summary of frame processing
            print(f"\nProcessed {len(frame_captions_results)} frames")
        
        # step 2: create formatted message based on available modalities
        print("\nPreparing message for VLM...")
        formatted_messages = [{"role": "user", "content": []}]
        
        # add frame captions if available
        if has_frames and frame_captions:
            formatted_messages[0]["content"].append({
                "type": "text",
                "text": f"Image descriptions: {' '.join(frame_captions)}"
            })
        
        # add audio transcription if available
        if has_audio:
            transcription_text = " ".join(
                [segment["text"] for segment in event['content']['transcription']]
            )
            formatted_messages[0]["content"].append({
                "type": "text",
                "text": f"Audio transcription: {transcription_text}"
            })
        
        # if no modalities found, raise error
        if not has_frames and not has_audio:
            raise ValueError("No frames or audio found in event")
            
        # add appropriate prompt based on available modalities
        if has_frames and has_audio:
            formatted_messages[0]["content"].append({
                "type": "text",
                "text": "Please provide a concise one sentence summary of this event based on the video frames descriptions and audio transcription. What is happening in this event?"
            })
        elif has_frames:
            formatted_messages[0]["content"].append({
                "type": "text",
                "text": "Please provide a concise one sentence summary of this event based on the video frames descriptions. What is happening in this event?"
            })
        else:  # audio only
            formatted_messages[0]["content"].append({
                "type": "text",
                "text": "Please provide a concise one sentence summary of this event based on the audio transcription. What is happening in this event?"
            })
        
        # step 3: generate summary through API client with progress indicator
        print("\nGenerating event summary...")
        try:
            response = self.qwen.client.chat.completions.create(
                model=self.qwen.model_name,
                messages=formatted_messages,
                temperature=0
            )
            print("Summary generated successfully")
        except Exception as e:
            if "maximum context length" in str(e):
                logger.error("Maximum context length exceeded. Truncating frame captions.")
                max_captions = 1000
                step = len(frame_captions) // max_captions
                reduced_captions = frame_captions[::step][:max_captions]
                logger.info(f"Reduced captions from {len(frame_captions)} to {len(reduced_captions)}")
                # update message with reduced captions
                formatted_messages[0]["content"][0]["text"] = f"Image descriptions: {' '.join(reduced_captions)}"
                response = self.qwen.client.chat.completions.create(
                    model=self.qwen.model_name,
                    messages=formatted_messages,
                    temperature=0
                )
                print("Summary generated successfully")
            else:
                raise e
        
        # step 4: create and store theta event
        summary = response.choices[0].message.content
        
        # create ThetaEvent object
        print("\nCreating theta event...")
        theta_event = ThetaEvent(
            features=event['features'],
            feature_times=None,
            frames=event['content']['frames'] if has_frames else [],
            frame_times=event['content']['frame_times'] if has_frames else [],
            frame_captions=frame_captions if has_frames else [],
            audio_times=event['content']['audio_times'] if has_audio else [],
            audio_transcription=event['content']['transcription'] if has_audio else [],
            holistic_audio_transcription=[],
            summary=summary,
            start_time=event['content']['segment_info']['start_time'],
            end_time=event['content']['segment_info']['end_time']
        )
        self.update_holistic_audio_transcription(theta_event, video_id)
        
        # save the theta event with progress indicator
        print("\nSaving theta event...")
        try:
            self.save_theta_event(theta_event, video_id)
            print(f"Successfully saved theta event for video {video_id}")
        except Exception as e:
            logger.error(f"Failed to save theta event: {e}")
            logger.exception("Full traceback:")
        
        # add to long-term store
        self.long_term_store.append(theta_event)
        print("\nReplay process completed successfully!")

    def consolidate_short_term_memory(self, memories: List[ShortTermMemory]) -> ShortTermMemory:
        """
        Combine all short-term memories into a single consolidated ShortTermMemory object.
        
        Args:
            memories: List of ShortTermMemory objects to combine
            
        Returns:
            A consolidated ShortTermMemory object
        """
        if not memories:
            logger.warning("No memories provided for consolidation")
            return None
        
        # sort memories by timestamp once at the beginning
        memories.sort(key=lambda x: x.segment_info.start_time)
        logger.info(f"Consolidating {len(memories)} memories spanning from {memories[0].segment_info.start_time:.2f}s to {memories[-1].segment_info.end_time:.2f}s")
        
        # initialize consolidated memory with basic info
        consolidated_memory = ShortTermMemory(
            features={},
            content={},
            timestamp=memories[0].timestamp,
            source_time=memories[0].source_time,
            modalities=set(),
            segment_info=SequenceSegment(
                start_time=memories[0].timestamp,
                end_time=memories[-1].timestamp
            ),
            transcription=[]
        )
        
        # collect all modalities in one pass
        consolidated_memory.modalities = list(set().union(*(memory.modalities for memory in memories)))
        logger.info(f"Processing modalities: {consolidated_memory.modalities}")
        
        # process features in parallel using multiprocessing
        with multiprocessing.Pool(processes=min(multiprocessing.cpu_count(), 4)) as pool:
            # process vision features
            if 'vision' in consolidated_memory.modalities:
                vision_results = self._process_vision_features(memories, pool)
                consolidated_memory.features.update(vision_results['features'])
                consolidated_memory.content.update(vision_results['content'])
            
            # process audio features
            if 'audio' in consolidated_memory.modalities:
                audio_results = self._process_audio_features(memories, pool)
                consolidated_memory.features.update(audio_results['features'])
                consolidated_memory.content.update(audio_results['content'])
        
        # log final consolidation results
        logger.info(f"Consolidation complete. Final memory contains:")
        if consolidated_memory.content.get('frames'):
            logger.info(f"- {len(consolidated_memory.content['frames'])} key frames")
        if consolidated_memory.content.get('audio_times'):
            logger.info(f"- {len(consolidated_memory.content['audio_times'])} audio segments")
        if consolidated_memory.content.get('transcription'):
            logger.info(f"- {len(consolidated_memory.content['transcription'])} transcription segments")

        return consolidated_memory
    
    def _process_vision_features(self, memories: List[ShortTermMemory], pool: multiprocessing.Pool) -> Dict[str, Any]:
        """Process vision features in parallel."""
        # collect all frames and features in one pass
        frames_data = []
        for memory in memories:
            if 'vision' in memory.modalities and 'frames' in memory.content:
                for idx, frame in enumerate(memory.content['frames']):
                    if idx < len(memory.content.get('frame_times', [])):
                        frame_time = memory.content['frame_times'][idx]
                        feature = self._extract_frame_feature(memory.features.get('vision'), idx)
                        if feature is not None:
                            # ensure feature is 1D array with correct dimension (1024)
                            if len(feature.shape) > 1:
                                feature = feature.flatten()
                            if feature.shape[0] != 1024:
                                logger.warning(f"Skipping frame feature with incorrect dimension: {feature.shape[0]}, expected 1024")
                                continue
                            frames_data.append((frame, feature, frame_time))
        
        if not frames_data:
            return {'features': {}, 'content': {}}
        
        # sort frames chronologically
        frames_data.sort(key=lambda x: x[2])
        
        # extract features and times for vectorized operations
        try:
            features = np.stack([f[1] for f in frames_data])
            times = np.array([f[2] for f in frames_data])
            
            # verify final feature dimensions
            if features.shape[1] != 1024:
                logger.error(f"Final vision features have incorrect dimension: {features.shape[1]}, expected 1024")
                return {'features': {}, 'content': {}}
        except ValueError as e:
            logger.error(f"Error stacking features: {e}")
            logger.error("Feature shapes: " + str([f[1].shape for f in frames_data]))
            return {'features': {}, 'content': {}}
        
        # select key frames using vectorized similarity computation
        key_indices = self._select_key_frames(features, times)
        
        # create results dictionary
        return {
            'features': {
                'vision': features,
                'vision_times': times
            },
            'content': {
                'frames': [frames_data[i][0] for i in key_indices],
                'frame_times': times[key_indices].tolist()
            }
        }
    
    def _process_audio_features(self, memories: List[ShortTermMemory], pool: multiprocessing.Pool) -> Dict[str, Any]:
        """Process audio features in parallel."""
        # collect all audio features and times in one pass
        audio_data = []
        transcriptions = []
        
        for memory in memories:
            if 'audio' in memory.modalities and 'audio' in memory.content:
                if 'audio' in memory.features:
                    audio_start = memory.content['audio'].get('start_time')
                    audio_end = memory.content['audio'].get('end_time', audio_start)
                    feature = memory.features['audio']
                    
                    # ensure feature has correct dimension (1024)
                    if isinstance(feature, torch.Tensor):
                        feature = feature.detach().cpu().numpy()
                    if len(feature.shape) > 1:
                        feature = feature.flatten()
                    if feature.shape[0] != 1024:
                        logger.warning(f"Skipping audio feature with incorrect dimension: {feature.shape[0]}, expected 1024")
                        continue
                        
                    audio_data.append((feature, audio_start, audio_end))
                if memory.transcription:
                    transcriptions.extend(memory.transcription)
        
        if not audio_data:
            return {'features': {}, 'content': {}}
        
        # process audio features in parallel
        processed_features = []
        processed_times = []
        
        for feature, start_time, _ in audio_data:
            if isinstance(feature, np.ndarray):
                processed_features.append(feature)
                processed_times.append(start_time)
        
        if processed_features:
            features = np.stack(processed_features)
            times = np.array(processed_times)
            
            # verify final feature dimensions
            if features.shape[1] != 1024:
                logger.error(f"Final audio features have incorrect dimension: {features.shape[1]}, expected 1024")
                return {'features': {}, 'content': {}}
            
            return {
                'features': {
                    'audio': features,
                    'audio_times': times
                },
                'content': {
                    'audio_times': times.tolist(),
                    'transcription': transcriptions if transcriptions else None
                }
            }
        
        return {'features': {}, 'content': {}}
    
    def _extract_frame_feature(self, features: Any, idx: int) -> Optional[np.ndarray]:
        """Extract feature for a specific frame index."""
        if features is None:
            return None
            
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
            
        if isinstance(features, np.ndarray):
            if len(features.shape) > 1 and features.shape[0] > 1 and idx < features.shape[0]:
                return features[idx]
            return features
            
        return None
    
    def _select_key_frames(self, features: np.ndarray, times: np.ndarray, 
                          similarity_threshold: float = 0.9) -> np.ndarray:
        """Select key frames using vectorized similarity computation."""
        if len(features) <= 2:
            return np.arange(len(features))
            
        # compute similarity matrix efficiently
        features_normalized = features / np.linalg.norm(features, axis=1, keepdims=True)
        similarity_matrix = np.dot(features_normalized, features_normalized.T)
        
        # initialize key frame indices
        key_indices = [0]  # always include first frame
        
        # select frames that are distinct enough from already selected frames
        for i in range(1, len(features)):
            similarities = similarity_matrix[i, key_indices]
            if np.all(similarities < similarity_threshold):
                key_indices.append(i)
        
        # always include last frame if distinct enough
        if len(features) > 1 and np.all(similarity_matrix[-1, key_indices] < similarity_threshold):
            key_indices.append(len(features) - 1)
        
        return np.array(key_indices)

    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        return {
            'short_term_size': sum(len(memories) for memories in self.short_term_buffer.values()),
            'long_term_size': len(self.long_term_store),
            'config': {
                'max_short_term': self.max_short_term,
                'max_long_term': self.max_long_term
            }
        }

    def _compute_frame_similarity(self, frame1_path: str, frame2_path: str) -> float:
        """Compute similarity between two frames using structural similarity."""
        frame1 = cv2.imread(frame1_path)
        frame2 = cv2.imread(frame2_path)
        
        # convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # compute SSIM using skimage
        score = ssim(gray1, gray2, data_range=gray1.max() - gray1.min())
        return score
        
    def _compute_audio_level(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Compute RMS level of audio segment in dB."""
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)  # convert to mono
        
        rms = np.sqrt(np.mean(np.square(audio_data)))
        db = 20 * np.log10(rms) if rms > 0 else -100
        return db
        
    def _segment_sequence(
        self,
        video_frames: Optional[List[str]] = None,
        frame_times: Optional[List[float]] = None,
        audio_data: Optional[np.ndarray] = None,
        audio_sample_rate: Optional[int] = None
    ) -> List[SequenceSegment]:
        """
        Intelligently segment video/audio sequence based on content changes.
        
        Args:
            video_frames: List of frame paths
            frame_times: List of frame timestamps
            audio_data: Audio waveform data
            audio_sample_rate: Audio sample rate in Hz
            
        Returns:
            List of SequenceSegment objects
        """
        segments = []
        
        if video_frames is None and audio_data is None:
            return segments
            
        # determine sequence duration
        if video_frames and frame_times:
            total_duration = frame_times[-1] - frame_times[0]
        elif audio_data is not None and audio_sample_rate:
            total_duration = len(audio_data) / audio_sample_rate
        else:
            return segments
            
        current_start = 0.0
        pbar = tqdm(total=int(total_duration), desc="Segmenting sequence", unit="sec")
        while current_start < total_duration:
            # initialize maximum possible end time
            current_end = min(current_start + self.max_segment_duration, total_duration)
            
            # find optimal segment end within the window
            optimal_end = current_end
            
            if video_frames and frame_times:
                # get frame indices within current window
                frame_indices = [
                    i for i, t in enumerate(frame_times)
                    if current_start <= t <= current_end
                ]
                
                if len(frame_indices) > 1:
                    # check frame similarities
                    for i in range(len(frame_indices)-1, 0, -1):
                        similarity = self._compute_frame_similarity(
                            video_frames[frame_indices[i]],
                            video_frames[frame_indices[i-1]]
                        )
                        if similarity < self.frame_similarity_threshold:
                            optimal_end = frame_times[frame_indices[i]]
                            break
                            
            if audio_data is not None and audio_sample_rate:
                # convert times to samples
                start_sample = int(current_start * audio_sample_rate)
                end_sample = int(current_end * audio_sample_rate)
                
                # analyze audio levels in small windows
                window_size = int(0.5 * audio_sample_rate)  # 500ms windows
                for i in range(end_sample - start_sample - window_size, 0, -window_size):
                    window_start = start_sample + i
                    window_end = window_start + window_size
                    level = self._compute_audio_level(
                        audio_data[window_start:window_end],
                        audio_sample_rate
                    )
                    if level < self.audio_silence_threshold:
                        optimal_end = (window_start / audio_sample_rate)
                        break
                        
            # ensure minimum duration
            if optimal_end - current_start < self.min_segment_duration:
                optimal_end = min(
                    current_start + self.min_segment_duration,
                    total_duration
                )
                
            # create segment
            segment = SequenceSegment(
                start_time=current_start,
                end_time=optimal_end
            )
            
            # add frames if available
            if video_frames and frame_times:
                segment.frames = [
                    f for f, t in zip(video_frames, frame_times)
                    if current_start <= t <= optimal_end
                ]
                segment.frame_times = [
                    t for t in frame_times
                    if current_start <= t <= optimal_end
                ]
                
            # add audio if available
            if audio_data is not None and audio_sample_rate:
                start_sample = int(current_start * audio_sample_rate)
                end_sample = int(optimal_end * audio_sample_rate)
                segment.audio_data = audio_data[start_sample:end_sample]
                
            segments.append(segment)
            pbar.update(int(optimal_end - current_start))
            current_start = optimal_end
            
        pbar.close()
        return segments
        
    def process_sequence(
        self,
        video_id: str,
        video_frames: Optional[List[str]] = None,
        frame_times: Optional[List[float]] = None,  # original timestamps from source video
        audio_data: Optional[np.ndarray] = None,
        audio_sample_rate: Optional[int] = None,
        base_time: float = 0.0  # base time offset in the original video
    ) -> None:
        """
        Process video/audio sequence by segmenting and extracting features.
        
        Args:
            video_id: Unique ID for the video
            video_frames: List of frame file paths
            frame_times: List of original timestamps for each frame in the source video
            audio_data: Audio waveform data
            audio_sample_rate: Audio sampling rate
            base_time: Base time offset in the original video (for handling video chunks)
        """
        # first try to load from checkpoint
        checkpoint_path = self._check_for_checkpoint(video_id)
        if checkpoint_path:
            logger.info(f"Found checkpoint for video {video_id}, attempting to load...")
            memories = self._load_checkpoint(checkpoint_path)
            if memories:
                logger.info(f"Successfully loaded {len(memories)} memories from checkpoint")
                self.short_term_buffer[video_id] = memories
                # proceed directly to consolidation and replay
                consolidated_memories = self.consolidate(memories)
                if consolidated_memories:
                    self.replay(consolidated_memories, video_id)
                return
            else:
                logger.warning("Failed to load checkpoint, proceeding with normal processing")
                
        # validate frame times if frames are provided
        if video_frames and not frame_times:
            raise ValueError("frame_times must be provided when processing video frames")
            
        if video_frames and len(video_frames) != len(frame_times):
            raise ValueError("Number of frames must match number of frame timestamps")
            
        # adjust frame times with base offset
        if frame_times:
            frame_times = [t + base_time for t in frame_times]
            
        # first segment the sequence
        segments = self._segment_sequence(
            video_frames=video_frames,
            frame_times=frame_times,
            audio_data=audio_data,
            audio_sample_rate=audio_sample_rate
        )
        
        # process each segment
        for segment in tqdm(segments, desc="Processing segments", unit="segment"):
            features = {}
            content = {}
            modalities = []
            transcription = []  # initialize transcription list
            
            # process video frames
            if segment.frames:
                vision_features = self.imagebind.extract_features(
                    {'vision': segment.frames},
                    ['vision']
                )['vision']
                # move tensor to CPU and convert to numpy
                if isinstance(vision_features, torch.Tensor):
                    vision_features = vision_features.detach().cpu().numpy()
                    
                # ensure features have correct dimension (1024)
                if len(vision_features.shape) > 1:
                    if vision_features.shape[1] != 1024:
                        logger.error(f"Vision features have incorrect dimension: {vision_features.shape[1]}, expected 1024")
                        continue
                    features['vision'] = vision_features
                    content['frames'] = segment.frames
                    content['frame_times'] = segment.frame_times  # original video timestamps
                    modalities.append('vision')
                    
            # process audio
            if segment.audio_data is not None and audio_sample_rate:
                # save audio segment to temporary file
                audio_file_path = f"/tmp/audio_segment_{int(time.time())}_{hash(str(segment.start_time))}.wav"
                
                try:
                    # ensure audio is mono and float32
                    if len(segment.audio_data.shape) > 1:
                        audio_data_mono = segment.audio_data.mean(axis=1)
                    else:
                        audio_data_mono = segment.audio_data
                    
                    # convert to float32 if needed
                    if audio_data_mono.dtype != np.float32:
                        audio_data_mono = audio_data_mono.astype(np.float32)
                    
                    # normalize audio
                    if np.abs(audio_data_mono).max() > 1.0:
                        audio_data_mono = audio_data_mono / np.abs(audio_data_mono).max()
                    
                    # save processed audio
                    wavfile.write(audio_file_path, audio_sample_rate, audio_data_mono)
                    
                    # get audio features using file path
                    audio_features = self.imagebind.extract_features(
                        {'audio': [audio_file_path]},
                        ['audio']
                    )['audio']
                    
                    # move tensor to CPU and convert to numpy
                    if isinstance(audio_features, torch.Tensor):
                        audio_features = audio_features.detach().cpu().numpy()
                        
                    # ensure features have correct dimension (1024)
                    if len(audio_features.shape) > 1:
                        if audio_features.shape[1] != 1024:
                            logger.error(f"Audio features have incorrect dimension: {audio_features.shape[1]}, expected 1024")
                            continue
                        features['audio'] = audio_features
                        
                        # process audio in chunks for transcription
                        max_chunk_duration = 600  # seconds
                        samples_per_chunk = max_chunk_duration * audio_sample_rate
                        
                        # process audio in chunks
                        for i in range(0, len(audio_data_mono), samples_per_chunk):
                            chunk = audio_data_mono[i:i + samples_per_chunk]
                            chunk_transcription = self.whisper.transcribe(chunk)
                            transcription.extend(chunk_transcription)
                            
                finally:
                    # clean up temporary file
                    if os.path.exists(audio_file_path):
                        os.remove(audio_file_path)
                        
            # create and add memory
            if features:  # only add if we have valid features
                memory = ShortTermMemory(
                    features=features,
                    content=content,
                    timestamp=time.time(),
                    source_time=segment.start_time,
                    modalities=modalities,
                    segment_info=segment,
                    transcription=transcription
                )
                
                # add to video-specific buffer
                self.short_term_buffer[video_id].append(memory)
                
        # save checkpoint after processing
        if self.short_term_buffer[video_id]:
            self._save_checkpoint(video_id, self.short_term_buffer[video_id])
            
        # proceed to consolidation and replay
        consolidated_memories = self.consolidate(self.short_term_buffer[video_id])
        if consolidated_memories:
            self.replay(consolidated_memories, video_id)

    def add_video(self, video_id: str, metadata: Dict[str, Any]) -> None:
        """Register a new video in the system.
        
        Args:
            video_id: Unique identifier for the video
            metadata: Video metadata (duration, source, etc.)
        """
        self.video_index[video_id] = {
            'metadata': metadata,
            'added_timestamp': time.time()
        }
        self._save_index(self.video_index_file, self.video_index) 

    def add_single_frame(
        self,
        frame_path: str,
        frame_time: float,
        video_id: str
    ) -> None:
        """Add a single frame to the memory system.
        
        Args:
            frame_path: Path to the frame image
            frame_time: Timestamp of the frame in the video
            video_id: ID of the video this frame belongs to
        """
        # add frame to buffer
        self.frame_buffer[video_id].append((frame_path, frame_time))
        
        # process batch if buffer is full
        if len(self.frame_buffer[video_id]) >= self.frame_buffer_size:
            self._process_frame_batch(video_id)
            
    def flush_frame_buffer(self, video_id: str) -> None:
        """Process any remaining frames in the buffer for a video."""
        if video_id in self.frame_buffer and self.frame_buffer[video_id]:
            self._process_frame_batch(video_id)
            
        # clean up any stored features
        if video_id in self.frame_features:
            del self.frame_features[video_id]

    def _process_frame_batch(self, video_id: str) -> None:
        """Process a batch of frames from the buffer."""
        if not self.frame_buffer[video_id]:
            return
            
        # get frames and times from buffer
        frames, times = zip(*self.frame_buffer[video_id])
        
        # extract features in batch
        vision_features = self.imagebind.extract_features(
            {'vision': list(frames)},
            ['vision']
        )['vision']
        
        # convert to numpy if needed
        if isinstance(vision_features, torch.Tensor):
            vision_features = vision_features.detach().cpu().numpy()
            
        # store features
        self.frame_features[video_id].extend([
            feat for feat in vision_features
        ])
        
        # create and add memory
        memory = ShortTermMemory(
            features={'vision': vision_features},
            content={
                'frames': list(frames),
                'frame_times': list(times)
            },
            timestamp=time.time(),
            source_time=times[0],  # use first frame's time
            modalities=['vision'],
            segment_info=SequenceSegment(
                start_time=times[0],
                end_time=times[-1],
                frames=list(frames),
                frame_times=list(times)
            ),
            transcription=[]  # no transcription for frame-only batch
        )
        
        # add to video-specific buffer
        self.short_term_buffer[video_id].append(memory)
        
        # clear processed frames
        self.frame_buffer[video_id].clear()

    def update_holistic_audio_transcription(self, event: ThetaEvent, video_id: str) -> Optional[ThetaEvent]:
        """Update holistic audio transcription for a theta event by processing the entire audio segment at once."""
        try:
            # get video path
            video_info = self.video_index.get(video_id)
            if not video_info or 'metadata' not in video_info:
                logger.error(f"Video ID {video_id} not found in video index")
                return None
                
            video_path = video_info['metadata'].get('path')
            if not video_path or not os.path.exists(video_path):
                logger.error(f"Video file not found: {video_path}")
                return None
                
            # extract audio using ffmpeg
            temp_audio_path = f"/tmp/audio_{video_id}_{int(time.time())}.wav"
            
            try:
                # extract audio at 16kHz, mono
                cmd = [
                    'ffmpeg', '-i', video_path,
                    '-vn',  # no video
                    '-acodec', 'pcm_s16le',  # PCM 16-bit
                    '-ar', '16000',  # 16kHz sampling
                    '-ac', '1',  # mono
                    '-y',  # overwrite output file
                    temp_audio_path
                ]
                
                subprocess.run(cmd, check=True, capture_output=True)
                
                # load the extracted audio
                audio_array, _ = sf.read(temp_audio_path)
            
                # transcribe
                transcription = self.whisper.transcribe(audio_array)
                event.holistic_audio_transcription = transcription
                return event
            except Exception as e:
                logger.error(f"Error updating audio transcription: {e}")
                return None
            finally:
                # clean up temporary file
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
                
        except Exception as e:
            logger.error(f"Error updating audio transcription: {e}")
            return None

    def _check_for_checkpoint(self, video_id: str) -> Optional[str]:
        """Check if a checkpoint exists for a video.
        
        Args:
            video_id: ID of the video to check
            
        Returns:
            Path to checkpoint file if exists, None otherwise
        """
        checkpoint_dir = self.storage_dir / 'checkpoints'
        if not checkpoint_dir.exists():
            return None
            
        # look for checkpoint file with video_id
        checkpoint_files = list(checkpoint_dir.glob(f'*_{video_id}_*.json'))
        if not checkpoint_files:
            return None
            
        # return most recent checkpoint
        return str(max(checkpoint_files, key=lambda x: x.stat().st_mtime))
        
    def _load_checkpoint(self, checkpoint_path: str) -> Optional[List[ShortTermMemory]]:
        """Load memories from a checkpoint file.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            List of ShortTermMemory objects if successful, None otherwise
        """
        try:
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
                
            memories = []
            for mem_dict in data['memories']:
                # convert base64 features back to numpy arrays
                for modality, b64_str in mem_dict['features'].items():
                    mem_dict['features'][modality] = self._base64_to_numpy(b64_str)
                
                # reconstruct segment info
                segment_info = SequenceSegment(
                    start_time=mem_dict['segment_info']['start_time'],
                    end_time=mem_dict['segment_info']['end_time'],
                    frames=mem_dict['segment_info'].get('frames'),
                    frame_times=mem_dict['segment_info'].get('frame_times'),
                    audio_data=None  # audio data not stored in checkpoint
                )
                
                # create ShortTermMemory object
                memory = ShortTermMemory(
                    features=mem_dict['features'],
                    content=mem_dict['content'],
                    timestamp=mem_dict['timestamp'],
                    source_time=mem_dict['source_time'],
                    modalities=mem_dict['modalities'],
                    segment_info=segment_info,
                    transcription=mem_dict['transcription']
                )
                memories.append(memory)
                
            logger.info(f"Successfully loaded {len(memories)} memories from checkpoint {checkpoint_path}")
            return memories
            
        except Exception as e:
            logger.error(f"Error loading checkpoint {checkpoint_path}: {e}")
            logger.exception("Full traceback:")
            return None
            
    def _save_checkpoint(self, video_id: str, memories: List[ShortTermMemory]) -> Optional[str]:
        """Save memories to a checkpoint file.
        
        Args:
            video_id: ID of the video
            memories: List of ShortTermMemory objects to save
            
        Returns:
            Path to saved checkpoint file if successful, None otherwise
        """
        try:
            checkpoint_dir = self.storage_dir / 'checkpoints'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # convert memories to serializable format
            serialized_memories = []
            for memory in memories:
                mem_dict = memory.to_dict()
                # convert numpy arrays to base64
                for modality, features in mem_dict['features'].items():
                    mem_dict['features'][modality] = self._numpy_to_base64(np.array(features))
                serialized_memories.append(mem_dict)
            
            # save to checkpoint file
            checkpoint_path = checkpoint_dir / f'checkpoint_{video_id}_{int(time.time())}.json'
            with open(checkpoint_path, 'w') as f:
                json.dump({
                    'video_id': video_id,
                    'memories': serialized_memories,
                    'timestamp': time.time()
                }, f, indent=2)
                
            logger.info(f"Successfully saved checkpoint to {checkpoint_path}")
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            logger.exception("Full traceback:")
            return None

    def save_short_term_buffer(self, temp_dir: Optional[str] = None) -> Dict[str, str]:
        """Save short-term memory buffer to temporary files.
        
        Args:
            temp_dir: Optional directory to save temp files. If None, uses system temp directory.
            
        Returns:
            Dictionary mapping video_ids to their temporary file paths
        """
        if temp_dir is None:
            temp_dir = os.path.join(self.storage_dir, 'temp_short_term')
        os.makedirs(temp_dir, exist_ok=True)
        
        file_paths = {}
        for video_id, memories in self.short_term_buffer.items():
            # convert memories to serializable format
            serialized_memories = []
            for memory in memories:
                mem_dict = memory.to_dict()
                # convert numpy arrays to base64
                for modality, features in mem_dict['features'].items():
                    mem_dict['features'][modality] = self._numpy_to_base64(np.array(features))
                serialized_memories.append(mem_dict)
            
            # save to temporary file
            temp_file = os.path.join(temp_dir, f'short_term_{video_id}_{int(time.time())}.json')
            with open(temp_file, 'w') as f:
                json.dump({
                    'video_id': video_id,
                    'memories': serialized_memories,
                    'timestamp': time.time()
                }, f, indent=2)
            file_paths[video_id] = temp_file
            
        return file_paths

    def load_short_term_buffer(self, file_paths: Dict[str, str]) -> None:
        """Load short-term memory buffer from temporary files.
        
        Args:
            file_paths: Dictionary mapping video_ids to their temporary file paths
        """
        for video_id, file_path in file_paths.items():
            if not os.path.exists(file_path):
                logger.warning(f"Temp file not found: {file_path}")
                continue
                
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # reconstruct memories
                memories = []
                for mem_dict in data['memories']:
                    # convert base64 features back to numpy arrays
                    for modality, b64_str in mem_dict['features'].items():
                        mem_dict['features'][modality] = self._base64_to_numpy(b64_str)
                    
                    # reconstruct segment info
                    segment_info = SequenceSegment(
                        start_time=mem_dict['segment_info']['start_time'],
                        end_time=mem_dict['segment_info']['end_time'],
                        frame_times=mem_dict['segment_info'].get('frame_times')
                    )
                    
                    # create ShortTermMemory object
                    memory = ShortTermMemory(
                        features=mem_dict['features'],
                        content=mem_dict['content'],
                        timestamp=mem_dict['timestamp'],
                        source_time=mem_dict['source_time'],
                        modalities=mem_dict['modalities'],
                        segment_info=segment_info,
                        transcription=mem_dict['transcription']
                    )
                    memories.append(memory)
                
                # update buffer
                self.short_term_buffer[video_id] = memories
                logger.info(f"Loaded {len(memories)} memories for video {video_id}")
                
                # clean up temp file
                os.remove(file_path)
                
            except Exception as e:
                logger.error(f"Error loading memories from {file_path}: {e}")
                logger.exception("Full traceback:")


class QARecallSystem:
    """QA-based recall system that uses LLMs for reasoning and retrieval."""
    
    def __init__(
        self,
        memory_system: HippocampalMemory,
        config: Dict[str, Any]
    ):
        """Initialize the QA recall system.
        
        Args:
            memory_system: Instance of HippocampalMemory
            config: Configuration dictionary
        """
        self.memory = memory_system
        self.config = config
        self.qwen = memory_system.qwen
        
        # initialize a separate client for reasoning with different model
        api_config = config.get('api', {}).get('reasoning', {})
        self.thinking_client = OpenAI(
            api_key=api_config.get('api_key','')
        )
        self.reasoning_model = api_config.get('model_name', "gpt-4o")  # use GPT-4 for reasoning
        
        self.imagebind = memory_system.imagebind  # reuse ImageBind for embeddings
        self.context_length = 120000
        self.tc = TokenCount(model_name=self.reasoning_model)
        
    def answer_question(self, question: str) -> QARecallResult:
        """Answer a question using the memory system.
        
        Args:
            question: The question to answer
            
        Returns:
            QARecallResult containing the answer and reasoning process
        """
        # step 0: classify question type
        self._current_question = question
        question_type = self._classify_question_type(question)
        logger.info(f"Question classified as type: {question_type}")
        
        direct_answer = None
        if question_type == "SUMMARY":
            # try direct answer first
            result = self._try_direct_answer(question, question_type)
            if result:
                result.question_type = question_type
                result.used_direct_answer = True
                return result
        elif question_type == "VIDEO+AUDIO":
            result = self._try_direct_answer(question, question_type)
            direct_answer = result.answer if result is not None else None
        else:
            result = self._try_direct_answer(question, question_type)
            if result:
                confidence = result.confidence
                if confidence > 0.7:
                    result.used_direct_answer = True
                    return result
                else:
                    direct_answer = result.answer
     
        if question_type == "VIDEO":
            result = self._process_video_query(question, find_video_segments=False)
            result.question_type = "VIDEO"
            result.primary_modality = "video"
        elif question_type == "AUDIO":
            primary_modality = self._determine_primary_modality(question)
            logger.info(f"Primary modality: {primary_modality}")
            result = self._process_audio_query(question, find_audio_segments=False, primary_modality=primary_modality)
            result.question_type = "AUDIO"
            result.primary_modality = primary_modality
        else:  # VIDEO+AUDIO
            result = self._process_multimodal_query(question)
            result.question_type = "VIDEO+AUDIO"
            result.primary_modality = "multimodal"
        
        # set segments analyzed count
        if result.retrieved_segments:
            result.segments_analyzed = len(result.retrieved_segments)

        if direct_answer and result.answer:
            # compare direct_answer with result.answer
            result = self._reflect_on_answer(question, direct_answer, result.answer)
            logger.info(f"Use reflection to answer question")
            
        return result
    
    def _reflect_on_answer(self, question: str, direct_answer: str, detailed_answer: str) -> QARecallResult:
        """Reflect on and reconcile two different answers to the same question.
        
        Args:
            question: The original question
            direct_answer: The answer derived directly from summaries
            detailed_answer: The answer derived from detailed segment analysis
            
        Returns:
            QARecallResult with the reconciled answer and reasoning
        """
        logger.info(f"Reflecting on two different answers for question: {question}")
        
        check_prompt = f"""You have two answers to the same question. The first was derived directly from summaries, while the second was derived from detailed segment analysis. Compare the two answers and determine if they are the same.

Question: {question}

Answer from summaries: {direct_answer}

Answer from detailed analysis: {detailed_answer}

Instructions:
1. Compare both answers for consistency, detail level, and confidence
2. If they agree, directly output yes
3. If they disagree, directly output no

Your output format must be structured as follows:
ANSWER: <yes or no>
"""
        check_response = self.thinking_client.chat.completions.create(
            model=self.reasoning_model,
            messages=[{"role": "user", "content": check_prompt}],
            temperature=0
        )
        check_answer = check_response.choices[0].message.content.strip().upper()
        if check_answer == "YES":
            return QARecallResult(
                answer=direct_answer,
                confidence=1.0,
                used_direct_answer=True,
                used_reflection=False,
                reasoning="The direct and detailed answers are the same"
            )
        else:
            # sample a few frame captions and transcriptions to provide context
            sample_frame_captions = []
            sample_transcriptions = []
            
            # get a sample of frame captions and transcriptions from long-term store
            for event in self.memory.long_term_store:
                # add a few frame captions
                if hasattr(event, 'frame_captions') and event.frame_captions:
                    # sample captions evenly across the event
                    num_captions = min(5, len(event.frame_captions))
                    if num_captions > 0:
                        indices = [int(i * len(event.frame_captions) / num_captions) for i in range(num_captions)]
                        for idx in indices:
                            if idx < len(event.frame_captions) and idx < len(event.frame_times):
                                sample_frame_captions.append(f"[{event.frame_times[idx]:.2f}s] {event.frame_captions[idx]}")
                
                # add a few transcriptions
                if hasattr(event, 'holistic_audio_transcription') and event.holistic_audio_transcription:
                    # sample transcriptions evenly across the event
                    transcriptions = event.holistic_audio_transcription
                    num_trans = min(5, len(transcriptions))
                    if num_trans > 0:
                        indices = [int(i * len(transcriptions) / num_trans) for i in range(num_trans)]
                        for idx in indices:
                            if idx < len(transcriptions):
                                trans = transcriptions[idx]
                                start_time = trans.get('start', 0)
                                sample_transcriptions.append(f"[{start_time:.2f}s] {trans['text']}")
            
            reflection_prompt = f"""You have two answers to the same question. The first was derived directly from summaries, while the second was derived from detailed segment analysis. Compare the two answers and provide a final answer that reflects the most accurate information.

Question: {question}

Answer from summaries: {direct_answer}

Answer from detailed analysis: {detailed_answer}

Here are some sample frame captions for context:
{chr(10).join(sample_frame_captions)}

Here are some sample transcriptions for context:
{chr(10).join(sample_transcriptions)}

Instructions:
1. Compare both answers for consistency, detail level, and confidence
2. If they agree, choose the more detailed and precise answer
3. If they disagree, evaluate which answer seems more reliable based on:
- Level of detail and specificity
- Logical coherence and completeness
- Relevance to the question
- Consistency with the provided frame captions and transcriptions
4. In ambiguous cases, prefer the answer from detailed analysis over the summary answer
5. For multiple choice questions, select only one answer option

Your output format must be structured as follows:
ANSWER: <reconciled final answer>
CONFIDENCE: <confidence score between 0.0-1.0>
REASONING: <brief explanation for why you chose this answer>
"""
            if self.tc.num_tokens_from_string(reflection_prompt) > self.context_length - 1000:
                logger.warning(f"Reflection prompt too long ({self.tc.num_tokens_from_string(reflection_prompt)} tokens), truncating")
                
                # limit samples to avoid token overflow
                if len(sample_frame_captions) > 1000:
                    step = len(sample_frame_captions) // 1000
                    sample_frame_captions = [sample_frame_captions[i] for i in range(0, len(sample_frame_captions), step)][:1000]
                
                if len(sample_transcriptions) > 1000:
                    step = len(sample_transcriptions) // 1000
                    sample_transcriptions = [sample_transcriptions[i] for i in range(0, len(sample_transcriptions), step)][:1000]
                
                reflection_prompt = f"""You have two answers to the same question. The first was derived directly from summaries, while the second was derived from detailed segment analysis. Compare the two answers and provide a final answer that reflects the most accurate information.

Question: {question}

Answer from summaries: {direct_answer}

Answer from detailed analysis: {detailed_answer}

Here are some sample frame captions for context:
{chr(10).join(sample_frame_captions)}

Here are some sample transcriptions for context:
{chr(10).join(sample_transcriptions)}

Instructions:
1. Compare both answers for consistency, detail level, and confidence
2. If they agree, choose the more detailed and precise answer
3. If they disagree, evaluate which answer seems more reliable based on:
- Level of detail and specificity
- Logical coherence and completeness
- Relevance to the question
- Consistency with the provided frame captions and transcriptions
4. In ambiguous cases, prefer the answer from detailed analysis over the summary answer
5. For multiple choice questions, select only one answer option

Your output format must be structured as follows:
ANSWER: <reconciled final answer>
CONFIDENCE: <confidence score between 0.0-1.0>
REASONING: <brief explanation for why you chose this answer>
"""

            response = self.thinking_client.chat.completions.create(
                model=self.reasoning_model,
                messages=[{"role": "user", "content": reflection_prompt}],
                temperature=0
            )
            
            # parse structured response
            response_lines = response.choices[0].message.content.strip().split('\n')
            response_dict = {}
            for line in response_lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    response_dict[key.strip()] = value.strip()
            
            try:
                confidence = float(response_dict.get('CONFIDENCE', '0.7'))  # default 0.7 if parsing fails
                confidence = max(0.0, min(1.0, confidence))  # ensure within bounds
            except ValueError:
                logger.warning(f"Could not parse confidence value, defaulting to 0.7")
                confidence = 0.7
            
            final_answer = response_dict.get('ANSWER', detailed_answer)
            reasoning = response_dict.get('REASONING', "Reconciled from both direct and detailed analysis")
            
            # return reconciled result
            return QARecallResult(
                answer=final_answer,
                confidence=confidence,
                reasoning=reasoning,
                used_direct_answer=False,  # we're using a combination, not just the direct answer
                used_reflection=True  # flag to indicate reflection was used
            )
    
    def _classify_question_type(self, question: str) -> str:
        """Classify the type of question based on what type of information is needed to answer it.
        
        Args:
            question: The question to classify
            
        Returns:
            One of: "VIDEO", "AUDIO", "VIDEO+AUDIO", "SUMMARY"
        """
        classification_prompt = f"""Classify this question into one of these categories based on what type of information is needed to answer it:

1. VIDEO - Questions specifically about visual elements, appearances, or actions that need frame-by-frame analysis, what is the main character holding?
2. AUDIO - Questions about sounds, speech, or audio content that need audio analysis, for example, what does the main character say/mention?
3. VIDEO+AUDIO - Questions that require both visual and audio information, for example, what is the main character doing while saying/mentioning something? or what is the main character saying/mentioning while doing something?
4. SUMMARY - Questions that only focus on the overall content of the video, for example, what is the main character doing?

Guidelines:
- If question is about visual details, appearances, or actions, classify as VIDEO
- If question is about sounds, speech, or audio content, classify as AUDIO
- If question requires both visual and audio information, classify as VIDEO+AUDIO
- If question is about the overall content of the video, classify as SUMMARY

Question: {question}

Return ONLY one of these exact words: VIDEO, AUDIO, VIDEO+AUDIO, SUMMARY"""

        response = self.thinking_client.chat.completions.create(
            model=self.reasoning_model,
            messages=[{"role": "user", "content": classification_prompt}],
            temperature=0
        )
        
        classification = response.choices[0].message.content.strip().upper()
        if classification not in ["VIDEO", "AUDIO", "VIDEO+AUDIO", "SUMMARY"]:
            logger.warning(f"Invalid classification '{classification}', defaulting to VIDEO+AUDIO")
            return "VIDEO+AUDIO"
            
        return classification
        
    def _try_direct_answer(self, question: str, question_type: str) -> Optional[QARecallResult]:
        """Try to answer question directly from event summaries or determine required modality."""
        logger.info(f"Attempting direct answer for question: {question}")
        logger.info(f"Number of events in long-term store: {len(self.memory.long_term_store)}")
        
        # format prompt for summary-based answering
        summary_prompt = f"""Given the following question and video event summaries, analyze whether the question can be answered directly or needs specific analysis.

Output format must be in one of these two structures:

1. If answerable from summaries and provided details:
ANSWER: <your detailed answer, if given a multiple choice question, output one letter>
CONFIDENCE: <score between 0.0-1.0>

2. If requiring specific analysis:
ANSWER: NONE
CONFIDENCE: 0.0

Guidelines for analysis:
- General questions about overall video content should be answered directly from summaries
- Questions about specific visual details (appearances, objects, actions) need VIDEO analysis
- Questions about specific sounds or speech need AUDIO analysis
- Questions combining visual and audio elements need VIDEO+AUDIO analysis

Question: {question}

Event Summaries:
"""
        total_tokens = self.tc.num_tokens_from_string(summary_prompt)
        # add event summaries and relevant details based on question type
        for event in self.memory.long_term_store:
            summary_prompt += f"- {event.summary}\n"
            total_tokens += self.tc.num_tokens_from_string(event.summary)
            if question_type == "VIDEO":
                if event.frame_captions:
                    frame_caption_text = f"  Frame details: {' '.join(event.frame_captions)}\n"
                    if self.tc.num_tokens_from_string(summary_prompt + frame_caption_text) > self.context_length:
                        # evenly distribute within frame captions
                        distributed_captions = self.evenly_distribute_captions(event.frame_captions, 
                                                                            self.context_length - self.tc.num_tokens_from_string(summary_prompt))
                        summary_prompt += f"  Frame details: {distributed_captions}\n"
                    else:
                        summary_prompt += frame_caption_text
            elif question_type == "AUDIO":
                if event.holistic_audio_transcription:
                    audio_texts = [t['text'] for t in event.holistic_audio_transcription]
                    audio_text = f"  Audio transcription: {' '.join(audio_texts)}\n"
                    if self.tc.num_tokens_from_string(summary_prompt + audio_text) > self.context_length:
                        # evenly distribute within audio transcription
                        distributed_transcription = self.evenly_distribute_transcription(audio_texts,
                                                                                        self.context_length - self.tc.num_tokens_from_string(summary_prompt))
                        summary_prompt += f"  Audio transcription: {distributed_transcription}\n"
                    else:
                        summary_prompt += audio_text
            else:  # VIDEO+AUDIO
                remaining_tokens = self.context_length - self.tc.num_tokens_from_string(summary_prompt)
                if event.frame_captions and event.holistic_audio_transcription:
                    # need to handle both frame captions and audio transcription
                    frame_caption_text = ' '.join(event.frame_captions)
                    audio_texts = [t['text'] for t in event.holistic_audio_transcription]
                    audio_text = ' '.join(audio_texts)
                    
                    # calculate tokens for each
                    caption_tokens = self.tc.num_tokens_from_string(frame_caption_text)
                    audio_tokens = self.tc.num_tokens_from_string(audio_text)
                    total_tokens = caption_tokens + audio_tokens
                    
                    if total_tokens > remaining_tokens:
                        # distribute tokens proportionally between captions and audio
                        caption_allocation = int(remaining_tokens * caption_tokens / total_tokens)
                        audio_allocation = remaining_tokens - caption_allocation
                        
                        # apply distributions
                        distributed_captions = self.evenly_distribute_captions(event.frame_captions, caption_allocation)
                        distributed_transcription = self.evenly_distribute_transcription(audio_texts, audio_allocation)
                        
                        summary_prompt += f"  Frame details: {distributed_captions}\n"
                        summary_prompt += f"  Audio transcription: {distributed_transcription}\n"
                    else:
                        # can fit both without distribution
                        summary_prompt += f"  Frame details: {frame_caption_text}\n"
                        summary_prompt += f"  Audio transcription: {audio_text}\n"
                
                elif event.frame_captions:
                    # only have frame captions
                    frame_caption_text = f"  Frame details: {' '.join(event.frame_captions)}\n"
                    if self.tc.num_tokens_from_string(summary_prompt + frame_caption_text) > self.context_length:
                        distributed_captions = self.evenly_distribute_captions(event.frame_captions, remaining_tokens)
                        summary_prompt += f"  Frame details: {distributed_captions}\n"
                    else:
                        summary_prompt += frame_caption_text
                
                elif event.holistic_audio_transcription:
                    # only have audio transcription
                    audio_texts = [t['text'] for t in event.holistic_audio_transcription]
                    audio_text = f"  Audio transcription: {' '.join(audio_texts)}\n"
                    if self.tc.num_tokens_from_string(summary_prompt + audio_text) > self.context_length:
                        distributed_transcription = self.evenly_distribute_transcription(audio_texts, remaining_tokens)
                        summary_prompt += f"  Audio transcription: {distributed_transcription}\n"
                    else:
                        summary_prompt += audio_text
                        
        logger.debug(f"Summary prompt: {summary_prompt}")
                
        response = self.thinking_client.chat.completions.create(
            model=self.reasoning_model,
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0
        )

        # parse structured response
        response_lines = response.choices[0].message.content.strip().split('\n')
        response_dict = {}
        for line in response_lines:
            if ':' in line:
                key, value = line.split(':', 1)
                response_dict[key.strip()] = value.strip()
                
        # if direct answer is provided
        if 'none' not in response_dict.get('ANSWER').lower():
            try:
                confidence = float(response_dict.get('CONFIDENCE', '0.0'))
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                logger.warning(f"Could not parse confidence value, defaulting to 0.5")
                confidence = 0.5
            if confidence > 0.5:
                return QARecallResult(
                    answer=response_dict['ANSWER'],
                    confidence=confidence,
                    reasoning=f"Answer derived from event summaries and {question_type.lower()} details"
                )
          
        # if specific analysis is needed
        if response_dict.get('ANSWER').lower() == 'none':
            return None
        
        # fallback for malformed responses
        logger.warning(f"Malformed model response: {response_dict}")
        return None
        
    def evenly_distribute_captions(self, captions, max_tokens):
        """Evenly distribute tokens among frame captions."""
        if not captions:
            return ""
        
        # if max_tokens is too small, return at least one caption
        if max_tokens < self.tc.num_tokens_from_string(captions[0]):
            return captions[0][:max(1, int(len(captions[0]) * max_tokens / self.tc.num_tokens_from_string(captions[0])))]
        
        total_captions = len(captions)
        
        # strategy: select evenly spaced captions
        if self.tc.num_tokens_from_string(' '.join(captions)) <= max_tokens:
            return ' '.join(captions)
        
        # calculate how many captions we can include
        max_caption_tokens = np.max([self.tc.num_tokens_from_string(cap) for cap in captions])
        estimated_captions = max(1, int(max_tokens / max_caption_tokens))
        
        # select evenly distributed indices
        if estimated_captions >= total_captions:
            return ' '.join(captions)
        
        # get evenly spaced indices
        indices = [int(i * total_captions / estimated_captions) for i in range(estimated_captions)]
        selected_captions = [captions[i] for i in indices]
        
        # if we still exceed tokens, trim the last caption
        result = ' '.join(selected_captions)
        if self.tc.num_tokens_from_string(result) > max_tokens and selected_captions:
            excess = self.tc.num_tokens_from_string(result) - max_tokens
            last_caption = selected_captions[-1]
            last_caption_tokens = self.tc.num_tokens_from_string(last_caption)
            
            if excess < last_caption_tokens:
                # trim the last caption proportionally
                trim_ratio = (last_caption_tokens - excess) / last_caption_tokens
                trimmed_last = last_caption[:int(len(last_caption) * trim_ratio)]
                selected_captions[-1] = trimmed_last
            else:
                # remove the last caption entirely
                selected_captions.pop()
        
        return ' '.join(selected_captions)

    def evenly_distribute_transcription(self, transcription_texts, max_tokens):
        """Evenly distribute tokens among audio transcription segments."""
        if not transcription_texts:
            return ""
        
        # similar logic to captions but for transcription segments
        total_segments = len(transcription_texts)
        
        # if max_tokens is too small, return at least the beginning of one segment
        if max_tokens < self.tc.num_tokens_from_string(transcription_texts[0]):
            return transcription_texts[0][:max(1, int(len(transcription_texts[0]) * max_tokens / self.tc.num_tokens_from_string(transcription_texts[0])))]
        
        # check if we can include everything
        if self.tc.num_tokens_from_string(' '.join(transcription_texts)) <= max_tokens:
            return ' '.join(transcription_texts)
        
        # calculate how many segments we can include
        max_segment_tokens = np.max([self.tc.num_tokens_from_string(seg) for seg in transcription_texts])
        estimated_segments = max(1, int(max_tokens / max_segment_tokens))
        
        # select evenly distributed segments
        if estimated_segments >= total_segments:
            return ' '.join(transcription_texts)
        
        # get evenly spaced indices
        indices = [int(i * total_segments / estimated_segments) for i in range(estimated_segments)]
        selected_segments = [transcription_texts[i] for i in indices]
        
        # if we still exceed tokens, trim the last segment
        result = ' '.join(selected_segments)
        if self.tc.num_tokens_from_string(result) > max_tokens and selected_segments:
            excess = self.tc.num_tokens_from_string(result) - max_tokens
            last_segment = selected_segments[-1]
            last_segment_tokens = self.tc.num_tokens_from_string(last_segment)
            
            if excess < last_segment_tokens:
                # trim the last segment proportionally
                trim_ratio = (last_segment_tokens - excess) / last_segment_tokens
                trimmed_last = last_segment[:int(len(last_segment) * trim_ratio)]
                selected_segments[-1] = trimmed_last
            else:
                # remove the last segment entirely
                selected_segments.pop()
        
        return ' '.join(selected_segments)
    
    def _process_video_query(self, question: str, find_video_segments: Optional[bool] = False) -> QARecallResult:
        """Process a question requiring video analysis."""
        # store question for use in frame selection
        self._current_question = question
        
        # generate search query for ImageBind
        search_query = self._format_search_query(question, "visual")
        search_response = self.thinking_client.chat.completions.create(
            model=self.reasoning_model,
            messages=[{
                "role": "user", 
                "content": search_query
            }],
            temperature=0
        )
        embedding_query = search_response.choices[0].message.content

        # get embeddings and find relevant segments
        query_features = self.imagebind.extract_features(
            {'text': [embedding_query]},
            ['text']
        )['text']
        relevant_segments = self._find_relevant_video_segments(query_features, search_query)
        if find_video_segments:
            return relevant_segments
        if not relevant_segments:
            return QARecallResult(
                answer="I could not find relevant visual information to answer this question.",
                confidence=0.0,
                reasoning="No relevant video segments found in memory."
            )
            
        # process frames with QwenVL
        segment_descriptions = []
        # first collect all frames from all segments
        all_segment_frames = []
        all_segment_times = []
                
        for segment in relevant_segments:
            if segment.frames:
                # get video path from video index
                frame_path = segment.frames[0]
                video_id = frame_path.split('/frames/')[-1].split('/')[0]
                
                if video_id not in self.memory.video_index:
                    logger.error(f"Video ID {video_id} not found in video index")
                    continue

                video_info = self.memory.video_index[video_id]
                video_path = video_info['metadata'].get('path')
                
                if not video_path or not os.path.exists(video_path):
                    logger.error(f"Video file not found: {video_path}")
                    continue

                # extract frames for all frame_times in this segment at once
                cap = cv2.VideoCapture(video_path)
                fps = 1
                frame_interval = 1.0 / fps

                for frame_time in segment.frame_times:
                    window_start = max(0, frame_time - 1)
                    window_end = frame_time + 1
                    
                    # set video position to window start
                    cap.set(cv2.CAP_PROP_POS_MSEC, window_start * 1000)
                    
                    # extract frames within the window
                    prev_frame_path = None
                    current_time = window_start
                    
                    while current_time <= window_end:
                        ret, frame = cap.read()
                        if not ret:
                            break
                            
                        temp_frame_path = f"/tmp/frame_{int(current_time * 1000)}.jpg"
                        frame_small = cv2.resize(frame, (320, 180))
                        cv2.imwrite(temp_frame_path, frame_small)
                        
                        # check similarity with previous frame if exists
                        if prev_frame_path:
                            similarity = self.memory._compute_frame_similarity(prev_frame_path, temp_frame_path)
                            if similarity > 0.3:
                                os.remove(temp_frame_path)
                                current_time += frame_interval
                                cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
                                continue
                        
                        all_segment_frames.append(temp_frame_path)
                        all_segment_times.append(current_time)
                        prev_frame_path = temp_frame_path
                        
                        current_time += frame_interval
                        cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
                
                cap.release()
        # now process all collected frames in one batch
        if all_segment_frames:
            try:
                frame_index_model_triples = [
                    (frame, idx, self.qwen.model_name) 
                    for idx, frame in enumerate(all_segment_frames)
                ]
                
                num_workers = multiprocessing.cpu_count()
                with multiprocessing.Pool(processes=num_workers) as pool:
                    frame_captions_results = list(tqdm(
                        pool.starmap(lambda f, i, m: process_frame_with_api(f, i, m, self.config), frame_index_model_triples),
                        total=len(frame_index_model_triples),
                        desc="Processing all frames",
                        unit="frame"
                    ))
                
                # sort results and generate description
                frame_captions_results.sort(key=lambda x: x[0])
                all_captions = [caption for _, caption in frame_captions_results]
                
                # try to summarize all captions first
                summarized_captions = None
                if len(all_captions) > 10:  # only summarize if we have many captions
                    summarized_captions = self._summarize_captions(all_captions, question)
                if summarized_captions:
                    caption_text = summarized_captions
                elif len(all_captions) > 200:  # if summarization failed and we have many captions, split and summarize
                    mid_point = len(all_captions) // 2
                    first_half = self._summarize_captions(all_captions[:mid_point], question) or chr(10).join(all_captions[:mid_point])
                    second_half = self._summarize_captions(all_captions[mid_point:], question) or chr(10).join(all_captions[mid_point:])
                    caption_text = f"First part:\n{first_half}\n\nSecond part:\n{second_half}"
                else:
                    caption_text = chr(10).join(all_captions)
                
                if caption_text == "":
                    segment_descriptions.append('None')
                else:
                    segment_descriptions.append(caption_text)
                # clean up temporary frame files
                for frame_path in all_segment_frames:
                    if os.path.exists(frame_path):
                        os.remove(frame_path)

            except Exception as e:
                logger.error(f"Error processing all segment frames: {e}")
                logger.exception("Full traceback:")
                # clean up any remaining temporary files
                for frame_path in all_segment_frames:
                    if os.path.exists(frame_path):
                        os.remove(frame_path)
        # generate final answer
        final_prompt = self._format_final_answer_prompt(
            question,
            segment_descriptions,
            "video"
        )
        
        final_response = self.thinking_client.chat.completions.create(
            model=self.reasoning_model,
            messages=[{
                "role": "user",
                "content": final_prompt
            }],
            temperature=0
        )
        
        return QARecallResult(
            answer=final_response.choices[0].message.content,
            confidence=0.8,
            retrieved_segments=relevant_segments,
            reasoning=f"Retrieved and analyzed {len(relevant_segments)} relevant video segments to answer the question.",
            segments_analyzed=len(relevant_segments) if relevant_segments else 0
        )

    def _process_audio_query(self, question: str, find_audio_segments: Optional[bool] = False, primary_modality: Optional[str] = None) -> QARecallResult:
        """Process a question requiring audio analysis."""

        if primary_modality == "speech":
            # collect all transcriptions with timestamps from long-term memory
            all_transcriptions = []
            for event in self.memory.long_term_store:
                if hasattr(event, 'holistic_audio_transcription') and event.holistic_audio_transcription:
                    for trans in event.holistic_audio_transcription:
                        # add timestamp information if available
                        if 'start' in trans:
                            all_transcriptions.append({
                                'text': trans['text'],
                                'start': trans['start'],
                                'end': trans.get('end', trans['start'] + 5)  # default 5s duration if no end
                            })
                elif hasattr(event, 'audio_transcription') and event.audio_transcription:
                    for trans in event.audio_transcription:
                        all_transcriptions.append({
                            'text': trans['text'],
                            'start': trans['start'],
                            'end': trans.get('end', trans['start'] + 5)  # default 5s duration if no end
                        })
            
            if not all_transcriptions:
                return QARecallResult(
                    answer="No speech transcriptions found in memory.",
                    confidence=0.0,
                    reasoning="No transcriptions available for analysis."
                )
            
            # create prompt for the thinking model to identify relevant time frames
            transcription_prompt = f"""Given this question and the transcriptions with timestamps, identify the most relevant time frames where the answer might be found.

Question: {question}

Transcriptions (with timestamps):
{chr(10).join(f"[{t['start']:.2f}s - {t['end']:.2f}s]: {t['text']}" for t in all_transcriptions)}

INSTRUCTIONS:
1. Analyze the transcriptions and identify segments that are most likely to contain the answer
2. Return a JSON array of time frames in this exact format:
[
    {{"start": START_TIME, "end": END_TIME}},
    {{"start": START_TIME, "end": END_TIME}}
]
3. Return at most 5 time frames
4. Include a small buffer around each time frame (±2 seconds)
5. If no relevant segments found, return "[]"

Example good responses:
[
    {{"start": 10.5, "end": 15.2}},
    {{"start": 45.8, "end": 52.3}}
]

or
[]

Your response (valid JSON only):"""

            # get time frames from thinking model
            response = self.thinking_client.chat.completions.create(
                model=self.reasoning_model,
                messages=[{"role": "user", "content": transcription_prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            # parse response to get time frames
            time_frames = []
            response_text = response.choices[0].message.content.strip()
            if response_text != "[]":
                try:
                    # parse JSON response
                    time_frame_data = json.loads(response_text)
                    if type(time_frame_data) is not list:
                        time_frame_data = [time_frame_data]
                    for frame in time_frame_data:
                        # add buffer and ensure start time is non-negative
                        time_frames.append((
                            max(0, frame['start'] - 2),
                            frame['end'] + 2
                        ))
                except Exception as e:
                    logger.error(f"Error parsing time frames JSON: {e}")
                    return self._handle_multimodal_corner_cases(question, primary_modality)
            
            if not time_frames:
                return self._handle_multimodal_corner_cases(question, primary_modality)
            
            # if just finding segments, return them

            relevant_segments = []
            for start, end in time_frames:
                segment = SequenceSegment(
                    start_time=start,
                    end_time=end,
                    audio_data=None
                )
                relevant_segments.append(segment)
            if find_audio_segments:
                return relevant_segments
                
        else:
            # audio query
            # generate search query for ImageBind
            search_query = self._format_search_query(question, "audio")
            search_response = self.thinking_client.chat.completions.create(
                model=self.reasoning_model,
                messages=[{
                    "role": "user", 
                    "content": search_query
                }],
                temperature=0
            )
            embedding_query = search_response.choices[0].message.content

            # get embeddings and find relevant segments
            query_features = self.imagebind.extract_features(
                {'text': [embedding_query]},
                ['text']
            )['text']
            relevant_segments = self._find_relevant_audio_segments(query_features)
        
        if not relevant_segments:
            return self._handle_multimodal_corner_cases(question, primary_modality)
        
        if find_audio_segments:
            return relevant_segments
        
        # process audio segments with Whisper
        segment_descriptions = []
        
        # first merge window
        time_windows = []
        for segment in relevant_segments:
            time_windows.append((segment.start_time, segment.end_time))

        video_id = self.memory.long_term_store[0].frames[0].split('/frames/')[-1].split('/')[0]
        # find audio segments in these time windows
        audio_segments = []

        # sort time windows by start time and merge overlapping intervals
        sorted_windows = sorted(time_windows, key=lambda x: x[0])
        merged_windows = []
        
        if sorted_windows:
            current_window = list(sorted_windows[0])  # convert to list for mutability
            
            for start, end in sorted_windows[1:]:
                if start <= current_window[1] + 2:  # allow 2 second gap for merging
                    current_window[1] = max(current_window[1], end)
                else:
                    merged_windows.append(tuple(current_window))
                    current_window = [start, end]
            merged_windows.append(tuple(current_window))
        
        # now process merged windows
        for start_time, end_time in merged_windows:
            segments = self._find_audio_segments_in_timeframe(video_id, start_time, end_time)
            audio_segments.extend(segments)
        
        if not audio_segments:
            return self._handle_multimodal_corner_cases(question, primary_modality)

        # process audio segments with Whisper
        segment_descriptions = []
        for segment in audio_segments:
            if segment.audio_data is not None:
                transcription = self.memory.whisper.transcribe(segment.audio_data)
                segment_descriptions.append(transcription)

        # generate final answer
        final_prompt = self._format_final_answer_prompt(
            question,
            segment_descriptions,
            "audio"
        )
        
        final_response = self.thinking_client.chat.completions.create(
            model=self.reasoning_model,
            messages=[{
                "role": "user",
                "content": final_prompt
            }],
            temperature=0
        )
        
        return QARecallResult(
            answer=final_response.choices[0].message.content,
            confidence=0.8,
            retrieved_segments=relevant_segments,
            reasoning=f"Retrieved and analyzed {len(relevant_segments)} relevant audio segments to answer the question.",
            segments_analyzed=len(relevant_segments) if relevant_segments else 0
        )
        
    # create prompt for the thinking model to identify relevant time frames
    def create_transcription_prompt(self, question, all_transcriptions, max_tokens):
        """Create prompt for identifying relevant time frames from transcriptions."""
        base_prompt = f"""Given this question and the transcriptions with timestamps, identify the most relevant time frames where the answer might be found.

Question: {question}

Transcriptions (with timestamps):
    """

        instruction_prompt = """
INSTRUCTIONS:
1. Analyze the transcriptions and identify segments that are most likely to contain the answer
2. Return a JSON array of time frames in this exact format:
[
    {"start": START_TIME, "end": END_TIME},
    {"start": START_TIME, "end": END_TIME}
]
3. Return at most 5 time frames
4. Include a small buffer around each time frame (±2 seconds)
5. If no relevant segments found, return "[]"

Example good responses:
[
    {"start": 10.5, "end": 15.2},
    {"start": 45.8, "end": 52.3}
]

or
[]

Your response (valid JSON only):"""

        # check if including all transcriptions would exceed token limit
        full_transcription_text = chr(10).join(f"[{t['start']:.2f}s - {t['end']:.2f}s]: {t['text']}" for t in all_transcriptions)
        full_prompt = base_prompt + full_transcription_text + instruction_prompt
        
        if self.tc.num_tokens_from_string(full_prompt) <= max_tokens:
            # can include all transcriptions
            transcription_prompt = full_prompt
        else:
            # need to distribute transcriptions evenly
            available_tokens = max_tokens - self.tc.num_tokens_from_string(base_prompt + instruction_prompt)
            
            # evenly distribute transcriptions
            distributed_transcriptions = self.evenly_distribute_transcriptions(all_transcriptions, available_tokens)
            
            transcription_prompt = base_prompt + distributed_transcriptions + instruction_prompt
        
        return transcription_prompt

    def evenly_distribute_transcriptions(self, transcriptions, max_tokens):
        """Evenly sample transcriptions to fit within token limit."""
        if not transcriptions:
            return ""
        
        total_transcriptions = len(transcriptions)
        
        # calculate token count for a representative sample
        sample_size = min(10, total_transcriptions)
        sample_indices = [int(i * total_transcriptions / sample_size) for i in range(sample_size)]
        sample_transcriptions = [transcriptions[i] for i in sample_indices]
        
        sample_text = chr(10).join(f"[{t['start']:.2f}s - {t['end']:.2f}s]: {t['text']}" for t in sample_transcriptions)
        sample_tokens = self.tc.num_tokens_from_string(sample_text)
        
        # estimate how many transcriptions we can include
        estimated_count = int((max_tokens / sample_tokens) * sample_size)
        
        if estimated_count >= total_transcriptions:
            # we can include all transcriptions
            return chr(10).join(f"[{t['start']:.2f}s - {t['end']:.2f}s]: {t['text']}" for t in transcriptions)
        
        # select evenly distributed transcriptions
        step = total_transcriptions / estimated_count
        indices = [int(i * step) for i in range(estimated_count)]
        
        # ensure we include the beginning, middle and end for context
        if 0 not in indices and total_transcriptions > 0:
            indices[0] = 0
        if total_transcriptions - 1 not in indices and total_transcriptions > 0:
            indices[-1] = total_transcriptions - 1
        
        # get middle index if not already included
        middle_idx = total_transcriptions // 2
        if middle_idx not in indices and total_transcriptions > 2:
            # replace the closest index with the middle index
            closest_idx = min(range(len(indices)), key=lambda i: abs(indices[i] - middle_idx))
            indices[closest_idx] = middle_idx
        
        selected_transcriptions = [transcriptions[i] for i in indices]
        
        # format the selected transcriptions
        result = chr(10).join(f"[{t['start']:.2f}s - {t['end']:.2f}s]: {t['text']}" for t in selected_transcriptions)
        
        # add indicator that transcriptions were sampled
        result += f"\n[Note: Showing {len(selected_transcriptions)} of {total_transcriptions} transcriptions due to length constraints]"
        
        return result

    def _handle_multimodal_corner_cases(self, question: str, primary_modality: str) -> Optional[QARecallResult]:
        """Handle corner cases in multimodal query processing.
        
        Args:
            question: The question to answer
            primary_modality: The primary modality being processed first
            
        Returns:
            Optional QARecallResult if a corner case is handled, None otherwise
        """
        # even if no specific visual segments found, try to use overall context
        video_context = []
        frame_descriptions = []
        audio_descriptions = []
        
        # collect all available visual context from long-term store
        for event in self.memory.long_term_store:
            video_context.append(event.summary)
            if hasattr(event, 'frame_captions') and event.frame_captions:
                # zip timestamps with captions for synchronized descriptions
                for time, caption in zip(event.frame_times, event.frame_captions):
                    frame_descriptions.append(f"[{time:.2f}s] {caption}")
                
                if hasattr(event, 'holistic_audio_transcription') and event.holistic_audio_transcription:
                    # handle audio transcriptions with their timestamps
                    for trans in event.holistic_audio_transcription:
                        start_time = trans.get('start', 0)
                        end_time = trans.get('end', start_time + 5)  # default 5s duration if no end
                        audio_descriptions.append(f"[{start_time:.2f}s - {end_time:.2f}s] {trans['text']}")
        
        # form final prompt using available context with chronological ordering
        final_prompt = f"""Based on the following overall video context, please answer this question:

Question: {question}

Overall Video Context:
{chr(10).join([f"- {summary}" for summary in video_context])}

Available Frame Descriptions (chronologically ordered):
{chr(10).join(frame_descriptions)}

Available Audio Transcriptions (chronologically ordered):
{chr(10).join(audio_descriptions)}

While specific visual segments matching the query weren't found, please analyze the available context to:
1. Identify any relevant visual information that might help answer the question
2. Explain what visual elements we would need to fully answer the question
3. Provide the best possible answer based on available information

For multiple choice questions, provide ONLY the letter of the best answer based on available context.
For open-ended questions, explain what can be determined from the available information.

Answer:"""

        if self.tc.num_tokens_from_string(final_prompt) > self.context_length:
            if primary_modality == "audio" or primary_modality == "speech":
                filtered_audio_descriptions = self.evenly_distribute_transcription(audio_descriptions, self.context_length/2)
                filtered_frame_descriptions = frame_descriptions
            else:
                filtered_frame_descriptions = self.evenly_distribute_captions(frame_descriptions, self.context_length)
                filtered_audio_descriptions = audio_descriptions
            
            final_prompt = f"""Based on the following overall video context, please answer this question:

Question: {question}

Overall Video Context:
{chr(10).join([f"- {summary}" for summary in video_context])}

Available Frame Descriptions (chronologically ordered):
{filtered_frame_descriptions}

Available Audio Transcriptions (chronologically ordered):
{filtered_audio_descriptions}

While specific visual segments matching the query weren't found, please analyze the available context to:
1. Identify any relevant visual information that might help answer the question
2. Explain what visual elements we would need to fully answer the question
3. Provide the best possible answer based on available information

For multiple choice questions, provide ONLY the letter of the best answer based on available context.
For open-ended questions, explain what can be determined from the available information.

Answer:"""

        final_response = self.thinking_client.chat.completions.create(
            model=self.reasoning_model,
            messages=[{
                "role": "user",
                "content": final_prompt
            }],
            temperature=0
        )
        
        return QARecallResult(
            answer=final_response.choices[0].message.content,
            confidence=0.3,  # lower confidence since using general context
            reasoning="No specific segments found, answer derived from overall information."
        )
    
        
    def _process_multimodal_query(self, question: str) -> QARecallResult:
        """Process a question requiring both video and audio analysis."""
        # first determine which modality to process first based on question structure
        primary_modality = self._determine_primary_modality(question)
        # log primary modality
        logger.info(f"Primary modality: {primary_modality}")
        
        if primary_modality == "audio" or primary_modality == "speech":
            # example: "What are they doing visually when they mention X?"
            # first find audio segments containing the mentioned content
            relevant_audio_segments = self._process_audio_query(question, find_audio_segments=True, primary_modality=primary_modality)
            if not relevant_audio_segments:
                return self._handle_multimodal_corner_cases(question, primary_modality)
            
            # then find visual content around those audio segment times
            time_windows = []
            for segment in relevant_audio_segments:
                time_windows.append((segment.start_time, segment.end_time))

            # modify video query to focus on these time windows
            video_segments = []
            for start_time, end_time in time_windows:
                segments = self._find_video_segments_in_timeframe(start_time, end_time)
                video_segments.extend(segments)
            
            if not video_segments:
                return self._handle_multimodal_corner_cases(question, primary_modality)
                
            # first collect all frames from all segments
            all_segment_frames = []
            all_segment_times = []
                    
            for segment in video_segments:
                if segment.frames:
                    # get video path from video index
                    frame_path = segment.frames[0]
                    video_id = frame_path.split('/frames/')[-1].split('/')[0]
                    
                    if video_id not in self.memory.video_index:
                        logger.error(f"Video ID {video_id} not found in video index")
                        continue

                    video_info = self.memory.video_index[video_id]
                    video_path = video_info['metadata'].get('path')
                    
                    if not video_path or not os.path.exists(video_path):
                        logger.error(f"Video file not found: {video_path}")
                        continue

                    # extract frames for all frame_times in this segment at once
                    cap = cv2.VideoCapture(video_path)
                    fps = 1
                    frame_interval = 1.0 / fps

                    for frame_time in segment.frame_times:
                        window_start = max(0, frame_time - 1)
                        window_end = frame_time + 1
                        
                        # set video position to window start
                        cap.set(cv2.CAP_PROP_POS_MSEC, window_start * 1000)
                        
                        # extract frames within the window
                        prev_frame_path = None
                        current_time = window_start
                        
                        while current_time <= window_end:
                            ret, frame = cap.read()
                            if not ret:
                                break
                                
                            temp_frame_path = f"/tmp/frame_{int(current_time * 1000)}.jpg"
                            frame_small = cv2.resize(frame, (320, 180))
                            cv2.imwrite(temp_frame_path, frame_small)
                            
                            # check similarity with previous frame if exists
                            if prev_frame_path:
                                similarity = self.memory._compute_frame_similarity(prev_frame_path, temp_frame_path)
                                if similarity > 0.4:
                                    os.remove(temp_frame_path)
                                    current_time += frame_interval
                                    cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
                                    continue
                            
                            all_segment_frames.append(temp_frame_path)
                            all_segment_times.append(current_time)
                            prev_frame_path = temp_frame_path
                            
                            current_time += frame_interval
                            cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
                    
                    cap.release()

            # now process all collected frames in one batch
            if all_segment_frames:
                try:
                    frame_index_model_triples = [
                        (frame, idx, self.qwen.model_name) 
                        for idx, frame in enumerate(all_segment_frames)
                    ]
                    
                    num_workers = multiprocessing.cpu_count()
                    with multiprocessing.Pool(processes=num_workers) as pool:
                        frame_captions_results = list(tqdm(
                            pool.starmap(lambda f, i, m: process_frame_with_api(f, i, m, self.config), frame_index_model_triples),
                            total=len(frame_index_model_triples),
                            desc="Processing all frames",
                            unit="frame"
                        ))
                    
                    # sort results and generate description
                    frame_captions_results.sort(key=lambda x: x[0])
                    all_captions = [caption for _, caption in frame_captions_results]
                
                except Exception as e:
                    logger.error(f"Error processing all segment frames: {e}")
                    logger.exception("Full traceback:")
                    # clean up any remaining temporary files
                    for frame_path in all_segment_frames:
                        if os.path.exists(frame_path):
                            os.remove(frame_path)
                            
            # generate final answer considering temporal relationships
            final_prompt = f"""Based on the following information, please answer this question:

Question: {question}

Overall Video Summary:
{chr(10).join([f"- {event.summary}" for event in self.memory.long_term_store])}

Audio-Guided Video Segment Analysis:
{chr(10).join(f"[{time}] {caption}" for time, caption in zip(all_segment_times, all_captions))}

Please analyze the audio-guided video segment captions to provide a comprehensive answer. Focus specifically on:
1. What is visually happening in the video segments identified by audio cues
2. Any new details revealed in the audio-guided segments that weren't in the original descriptions
3. The contextual relationship between what was heard and what was seen

For multiple choice questions, provide ONLY the letter of the correct answer.
For open-ended questions, provide a concise answer based on the evidence.
Answer:"""
            final_response = self.thinking_client.chat.completions.create(
                model=self.reasoning_model,
                messages=[{
                    "role": "user",
                    "content": final_prompt
                }],
                temperature=0
            )
        
            return QARecallResult(
                answer=final_response.choices[0].message.content,
                confidence=0.8,
                retrieved_segments=None,
                reasoning=f"Sequential analysis of {'audio then video' if primary_modality == 'audio' else 'video then audio'} segments with temporal alignment.",
                segments_analyzed=len(video_segments) if video_segments else 0
            )
                
        else:  # primary_modality == "video"
            # example: "What are they saying when they do X?"
            # first find visual segments containing the action
            relevant_video_segments = self._process_video_query(question, find_video_segments=True)
            if not relevant_video_segments:
                return self._handle_multimodal_corner_cases(question, primary_modality)
            
            # then find audio content around those video segment times
            time_windows = []
            for segment in relevant_video_segments:
                time_windows.append((segment.start_time, segment.end_time))
            
            video_id = relevant_video_segments[0].frames[0].split('/frames/')[-1].split('/')[0]
            # find audio segments in these time windows
            audio_segments = []

            # sort time windows by start time and merge overlapping intervals
            sorted_windows = sorted(time_windows, key=lambda x: x[0])
            merged_windows = []
            
            if sorted_windows:
                current_window = list(sorted_windows[0])  # convert to list for mutability
                
                for start, end in sorted_windows[1:]:
                    if start <= current_window[1] + 2:  # allow 2 second gap for merging
                        current_window[1] = max(current_window[1], end)
                    else:
                        merged_windows.append(tuple(current_window))
                        current_window = [start, end]
                merged_windows.append(tuple(current_window))
            
            # now process merged windows
            for start_time, end_time in merged_windows:
                segments = self._find_audio_segments_in_timeframe(video_id, start_time, end_time)
                audio_segments.extend(segments)
            
            if not audio_segments:
                return self._handle_multimodal_corner_cases(question, primary_modality)

            # process audio segments with whisper
            audio_transcriptions = []
            for segment in audio_segments:
                if segment.audio_data is not None:
                    transcription = self.memory.whisper.transcribe(segment.audio_data)
                    audio_transcriptions.extend(transcription)
                    
            # generate final answer considering temporal relationships
            final_prompt = f"""Based on the following information, please answer this question:

Question: {question}

Overall Video Summary:
{chr(10).join([f"- {event.summary}" for event in self.memory.long_term_store])}

Using the question's visual hint, here is the retrieved Visual-Guided Audio Segment Transcription:
{chr(10).join([f"{transcription['text']}" for transcription in audio_transcriptions])}

Please analyze the audio-guided video segment captions to provide a comprehensive answer. Focus specifically on:
1. What is being said during the identified visual actions
2. The temporal relationship between the visual actions and audio content
3. Any context from the overall video that helps understand this relationship

For multiple choice questions, provide ONLY the letter of the correct answer.
For open-ended questions, provide a concise answer based on the evidence.
Answer:"""

            final_response = self.thinking_client.chat.completions.create(
                model=self.reasoning_model,
                messages=[{
                    "role": "user",
                    "content": final_prompt
                }],
                temperature=0
            )
            
            return QARecallResult(
                answer=final_response.choices[0].message.content,
                confidence=0.8,
                retrieved_segments=None,
                reasoning=f"Sequential analysis of {'audio then video' if primary_modality == 'audio' else 'video then audio'} segments with temporal alignment.",
                segments_analyzed=len(relevant_video_segments) if relevant_video_segments else 0
            )

    def _determine_primary_modality(self, question: str) -> str:
        """Determine which modality should be processed first based on question structure."""
        prompt = f"""Analyze this question to determine which modality should be processed first.

Question: {question}

Guidelines:
1. If the question asks about "when they mention/say/talk about X", process SPEECH first
2. If the question asks about "what they say/mention when doing X", process VIDEO first
3. If the question asks about "what sound/noise is heard when X", process SOUND first

Examples:
- "What is the person doing visually when they mention X?" -> SPEECH (find the mention first)
- "What are they saying when they perform X action?" -> VIDEO (find the action first)
- "What sound is heard when they pick up X?" -> SOUND (find the sound first)
- "What is the main character doing when they talk about Y?" -> SPEECH (find the talk first)
- "What do they mention while picking up X?" -> VIDEO (find pickup first)

Key patterns for SPEECH first:
- "when they mention..."
- "when they talk about..."
- "when they say..."
- "while discussing..."
- "what they say..."

Key patterns for VIDEO first:
- "what they say while..."
- "what they mention during..."
- "what do they say as they..."
- "when they are doing..."

Key patterns for SOUND first:
- "what sound is heard..."
- "what noise is made..."
- "what sound when..."
- "what sound does it make..."

Return ONLY one word: "video", "speech", or "sound"

Answer:"""

        response = self.thinking_client.chat.completions.create(
            model=self.reasoning_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        result = response.choices[0].message.content.strip().lower()
        
        # validate the result
        if result not in ["video", "speech", "sound"]:
            logger.warning(f"Invalid modality '{result}', defaulting to 'video'")
            return "video"
            
        return result

    def _find_video_segments_in_timeframe(self, start_time: float, end_time: float) -> List[SequenceSegment]:
        """Find video segments within a specific time window."""
        relevant_segments = []
        for event in self.memory.long_term_store:
            if not hasattr(event, 'frame_times') or not event.frame_times:
                continue
                
            # find frames within the time window
            frame_indices = [
                i for i, t in enumerate(event.frame_times)
                if start_time - 2 <= t <= end_time + 2  # add small buffer
            ]
            
            if frame_indices:
                segment = SequenceSegment(
                    start_time=event.frame_times[frame_indices[0]],
                    end_time=event.frame_times[frame_indices[-1]],
                    frames=[event.frames[i] for i in frame_indices],
                    frame_times=[event.frame_times[i] for i in frame_indices]
                )
                relevant_segments.append(segment)
                
        return relevant_segments

    def _find_audio_segments_in_timeframe(self, video_id: str, start_time: float, end_time: float) -> List[SequenceSegment]:
        """Find audio segments within a specific time window by loading directly from video using ffmpeg."""
        relevant_segments = []

        # get video path from video index
        video_info = self.memory.video_index[video_id]
        video_path = video_info['metadata'].get('path')
        if not video_path or not os.path.exists(video_path):
            logger.warning(f"Video file not found: {video_path}")
            return []

        try:
            # calculate time window (with buffer)
            buffered_start = max(0, start_time - 2)  # 2 second buffer
            buffered_end = end_time + 2
            duration = buffered_end - buffered_start
            
            # create temporary audio file
            temp_audio_path = f"/tmp/audio_segment_{video_id}_{int(time.time())}.wav"
            
            # use ffmpeg to extract audio for specified time window
            cmd = [
                'ffmpeg',
                '-ss', str(buffered_start),  # start time
                '-t', str(duration),         # duration
                '-i', video_path,            # input video
                '-vn',                       # no video
                '-acodec', 'pcm_s16le',     # PCM 16-bit
                '-ar', '16000',             # 16kHz sample rate
                '-ac', '1',                 # mono channel
                '-y',                       # overwrite output file
                temp_audio_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            # load the extracted audio
            audio_array, _ = sf.read(temp_audio_path)
            
            # create sequence segment
            segment = SequenceSegment(
                start_time=buffered_start,
                end_time=buffered_end,
                audio_data=audio_array
            )
            relevant_segments.append(segment)
            
            # clean up temporary file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                
        except Exception as e:
            logger.error(f"Error extracting audio from video {video_path}: {e}")
            logger.exception("Full traceback:")
            return []

        return relevant_segments
        
    def _format_search_query(self, question: str, modality: str) -> str:
        """Format prompt to generate embedding search query for ImageBind.
        
        ImageBind is a multimodal embedding model that works best with concise, focused queries.
        This method generates a short, specific description optimized for ImageBind's embedding space.
        
        Args:
            question: The original question to answer
            modality: The type of content to search for ('visual' or 'audio')
            
        Returns:
            A concise description optimized for ImageBind embeddings
        """
        return f"""Given this question, generate a very short (2-5 words) description optimized for ImageBind embeddings.
ImageBind is a multimodal model that works best with concise, focused queries that capture key visual/audio elements.

Question: {question}

Generate a short description (2-5 words) that captures the key {modality} elements we need to find.
Focus on concrete, specific elements rather than abstract concepts.
Example for visual: "person riding bicycle" instead of "transportation activity"
Example for audio: "loud thunder sound" instead of "weather noise"

Short description:"""

    def _find_relevant_video_segments(self, query_features: torch.Tensor, optional_search_query: str = None) -> List[SequenceSegment]:
        """Find video segments relevant to the query."""
        relevant_segments = []
        query_features_np = query_features.cpu().numpy()
        
        # ensure query features are 1D and have correct dimension (1024)
        if len(query_features_np.shape) > 1:
            query_features_np = query_features_np.flatten()
        if query_features_np.shape[0] != 1024:
            logger.warning(f"Unexpected query feature dimension: {query_features_np.shape[0]}, expected 1024")
            return []
            
        # store similarities and segments for sorting
        similarity_segments = []
            
        # look through all events in long-term store
        for event in self.memory.long_term_store:
            if 'vision' not in event.features:
                continue
                
            # get event features and ensure they're 1D with correct dimension
            event_features = event.features['vision']
            if isinstance(event_features, torch.Tensor):
                event_features = event_features.cpu().numpy()
            
            # compute similarity
            top_k_indices, top_k_similarities = top_k_cosine_similarity(query_features_np, event_features, k=5)
            
            # if similarities are too low, try using frame descriptions
            if np.max(top_k_similarities) < 0.4 and hasattr(event, 'frame_captions') and event.frame_captions:
                # create prompt for LLM to find relevant frames
                frame_selection_prompt = f"""Given a question, frame descriptions and an optional element to search for, identify at most 5 relevant frames for answering the question.

Question: {self._current_question}
Element to search for: {optional_search_query}

Frame descriptions:
{chr(10).join(f"{i}: {desc}" for i, desc in enumerate(event.frame_captions))}

INSTRUCTIONS:
1. Return ONLY numbers separated by commas (e.g., "0,3,5,8,12")
2. Return at most 5 indices
3. Do not include any other text, explanations, or spaces
4. Use indices from 0 to {len(event.frame_captions)-1}
5. If fewer than 5 frames are relevant, return fewer indices

Example good responses:
"0,3,5,8,12"
"1,4,7"
"0,1,2"

Example bad responses:
"Frame indices: 0, 3, 5" (includes text)
"0, 3, 5, 8, 12" (includes spaces)
"1,2,3,4,5,6" (more than 5 indices)

Your response (numbers only, comma-separated):"""
                
                # count the tokens
                total_tokens = self.tc.num_tokens_from_string(frame_selection_prompt)
                if total_tokens > self.context_length:
                    filtered_frame_captions = self.evenly_distribute_captions(event.frame_captions, 
                                                                            self.context_length-1000)
                    frame_selection_prompt = f"""Given a question, frame descriptions and an optional element to search for, identify at most 5 relevant frames for answering the question.

Question: {self._current_question}
Element to search for: {optional_search_query}

Frame descriptions:
{filtered_frame_captions}

INSTRUCTIONS:
1. Return ONLY numbers separated by commas (e.g., "0,3,5,8,12")
2. Return at most 5 indices
3. Do not include any other text, explanations, or spaces
4. Use indices from 0 to {len(event.frame_captions)-1}
5. If fewer than 5 frames are relevant, return fewer indices

Example good responses:
"0,3,5,8,12"
"1,4,7"
"0,1,2"

Example bad responses:
"Frame indices: 0, 3, 5" (includes text)
"0, 3, 5, 8, 12" (includes spaces)
"1,2,3,4,5,6" (more than 5 indices)

Your response (numbers only, comma-separated):"""
                
                # get LLM response
                response = self.thinking_client.chat.completions.create(
                    model=self.reasoning_model,
                    messages=[{"role": "user", "content": frame_selection_prompt}],
                    temperature=0
                )
                
                try:
                    # parse frame indices from response and limit to top 5
                    frame_indices = [int(idx.strip()) for idx in response.choices[0].message.content.split(',')][:5]
                    
                    # get corresponding frame times
                    frame_times = [event.frame_times[idx - 1] for idx in frame_indices if idx < len(event.frame_times)]
                    
                    # create segments around selected frames
                    for time in frame_times:
                        segment = SequenceSegment(
                            start_time=max(0, time - 1),  # 1 seconds before
                            end_time=time + 1,  # 1 seconds after
                            frames=[event.frames[i] for i in range(len(event.frames)) 
                                   if event.frame_times[i] >= time - 1 and event.frame_times[i] <= time + 1],
                            frame_times=[t for t in event.frame_times 
                                       if t >= time - 1 and t <= time + 1]
                        )
                        similarity_segments.append((0.6, [segment]))  # use moderate similarity score
                
                except Exception as e:
                    logger.error(f"Error processing LLM frame selection: {e}")
                    # fall back to using top-k frames
                    for idx, similarity in zip(top_k_indices, top_k_similarities):
                        if idx < len(event.frame_times):
                            frame_time = event.frame_times[idx]
                            segment = SequenceSegment(
                                start_time=max(0, frame_time - 1),  # 1 seconds before
                                end_time=frame_time + 1,  # 1 seconds after
                                frames=[event.frames[i] for i in range(len(event.frames)) 
                                       if event.frame_times[i] >= frame_time - 1 and event.frame_times[i] <= frame_time + 1],
                                frame_times=[t for t in event.frame_times 
                                           if t >= frame_time - 1 and t <= frame_time + 1]
                            )
                            similarity_segments.append((similarity, [segment]))
            else:
                logger.info(f"Using frame features for detailed retrieval.")
                # use top-k frames for high-similarity cases
                for idx, similarity in zip(top_k_indices, top_k_similarities):
                    if idx < len(event.frame_times):
                        frame_time = event.frame_times[idx]
                        segment = SequenceSegment(
                            start_time=max(0, frame_time - 1),  # 1 seconds before
                            end_time=frame_time + 1,  # 1 seconds after
                            frames=[event.frames[i] for i in range(len(event.frames)) 
                                   if event.frame_times[i] >= frame_time - 1 and event.frame_times[i] <= frame_time + 1],
                            frame_times=[t for t in event.frame_times 
                                       if t >= frame_time - 1 and t <= frame_time + 1]
                        )
                        similarity_segments.append((similarity, [segment]))
                
        # sort by similarity and get top segments
        similarity_segments.sort(key=lambda x: x[0], reverse=True)
        for _, segments in similarity_segments[:5]:  # return top 5 most relevant segments
            relevant_segments.extend(segments)
            
        return relevant_segments
        
    def _find_relevant_audio_segments(self, query_features: torch.Tensor) -> List[SequenceSegment]:
        """Find audio segments relevant to the query."""
        relevant_segments = []
        query_features_np = query_features.cpu().numpy()
        
        # ensure query features are 1D and have correct dimension (1024)
        if len(query_features_np.shape) > 1:
            query_features_np = query_features_np.flatten()
            
        # store similarities and segments for sorting
        similarity_segments = []
            
        # look through all events in long-term store
        for event in self.memory.long_term_store:
            if 'audio' not in event.features:
                continue
                
            # get event features and ensure they're 1D with correct dimension (1024)
            event_features = event.features['audio']
            if isinstance(event_features, torch.Tensor):
                event_features = event_features.cpu().numpy()
                
            # compute similarity
            top_k_indices, top_k_similarities = top_k_cosine_similarity(query_features_np, event_features, k=5)
            
            # if similarities are too low, try using audio transcription
            if max(top_k_similarities) < 0.4 and hasattr(event, 'holistic_audio_transcription') and event.holistic_audio_transcription:
                logger.info(f"Using audio transcription for detailed retrieval.")
                audio_selection_prompt = self.create_transcription_prompt(self._current_question, event.holistic_audio_transcription, self.context_length)
                try:
                    # get LLM response
                    response = self.thinking_client.chat.completions.create(
                        model=self.reasoning_model,
                        messages=[{"role": "user", "content": audio_selection_prompt}],
                        temperature=0,
                        response_format={"type": "json_object"}
                    )
                except Exception as e:
                    audio_selection_prompt = self.create_transcription_prompt(self._current_question, event.holistic_audio_transcription, self.context_length/2)
                    response = self.thinking_client.chat.completions.create(
                        model=self.reasoning_model,
                        messages=[{"role": "user", "content": audio_selection_prompt}],
                        temperature=0,
                        response_format={"type": "json_object"}
                    )
                
                try:
                    # parse JSON response
                    try:
                        audio_times = json.loads(response.choices[0].message.content)
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON response: {response.choices[0].message.content}")
                        logger.info(f"Using audio features for detailed retrieval.")
                        # use top-k segments for high-similarity cases
                        for idx, similarity in zip(top_k_indices, top_k_similarities):
                            if idx < len(event.feature_times['audio_times']):
                                segment = SequenceSegment(
                                    start_time=max(0, event.feature_times['audio_times'][idx] - 1),
                                    end_time=event.feature_times['audio_times'][idx] + 1,
                                    audio_data=None
                                )
                                similarity_segments.append((similarity, [segment]))
                    if len(audio_times) == 2 and isinstance(audio_times, dict):
                        audio_times = [audio_times]
                    # create segments around selected audio times
                    for time in audio_times:
                        segment = SequenceSegment(
                            start_time=max(0, time['start'] - 1),  # 1 seconds before
                            end_time=time['end'] + 1,  # 1 seconds after
                            audio_data=None  # will be filled later if needed
                        )
                        similarity_segments.append((0.6, [segment]))  # use moderate similarity score
                
                except Exception as e:
                    logger.error(f"Error processing LLM audio selection: {e}")
                    logger.info(f"Using audio features for detailed retrieval.")
                    # use top-k segments for high-similarity cases
                    for idx, similarity in zip(top_k_indices, top_k_similarities):
                        if idx < len(event.feature_times['audio_times']):
                            segment = SequenceSegment(
                                start_time=max(0, event.feature_times['audio_times'][idx] - 1),
                                end_time=event.feature_times['audio_times'][idx] + 1,
                                audio_data=None
                            )
                            similarity_segments.append((similarity, [segment]))
            else:
                logger.info(f"Using audio features for detailed retrieval.")
                # use top-k segments for high-similarity cases
                for idx, similarity in zip(top_k_indices, top_k_similarities):
                    if idx < len(event.feature_times['audio_times']):
                        segment = SequenceSegment(
                            start_time=max(0, event.feature_times['audio_times'][idx] - 1),
                            end_time=event.feature_times['audio_times'][idx] + 1,
                            audio_data=None
                        )
                        similarity_segments.append((similarity, [segment]))
                
        # sort by similarity and get top-k
        similarity_segments.sort(key=lambda x: x[0], reverse=True)
        for _, segments in similarity_segments[:5]:  # return segments from top 5 most relevant events
            relevant_segments.extend(segments)

        return relevant_segments
        
    def _get_base64_image(self, image_path: str) -> str:
        """Convert image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
            
    def _format_final_answer_prompt(
        self,
        question: str,
        segment_descriptions: List[str],
        modality: str
    ) -> str:
        """Format prompt for generating final answer."""
        # get overall video summaries from long-term store
        video_context = []
        for event in self.memory.long_term_store:
            video_context.append(f"- {event.summary}\n")
            if event.frame_captions and modality == "video":
                if self.tc.num_tokens_from_string(f"  Frame details: {' '.join(event.frame_captions)}\n") > self.context_length:
                    frame_captions = self.evenly_distribute_captions(event.frame_captions, self.context_length - 2000)
                    video_context.append(f"  Frame details: {frame_captions}\n")
                else:
                    video_context.append(f"  Frame details: {' '.join(event.frame_captions)}\n")
            if event.holistic_audio_transcription and modality == "audio":
                if self.tc.num_tokens_from_string(f"  Audio transcription: {' '.join([t['text'] for t in event.holistic_audio_transcription])}\n") > self.context_length:
                    audio_transcription = self.evenly_distribute_transcription(event.holistic_audio_transcription, self.context_length - 2000)
                    video_context.append(f"  Audio transcription: {audio_transcription}\n")
                else:
                    video_context.append(f"  Audio transcription: {' '.join([t['text'] for t in event.holistic_audio_transcription])}\n")
            
        return f"""Based on the following video context and detailed descriptions from {modality} content, please answer this question:

Question: {question}

Overall Video Context: 
{chr(10).join(video_context)}

Relevant {modality.title()} Content:
{chr(10).join(f"- {desc}" for desc in segment_descriptions)}

Please provide a clear and specific answer based on both the overall context and the specific content above. If the content doesn't contain enough information to fully answer the question, please guess the answer based on the context.

Output should be one letter if given a multiple choice question.

Answer:"""

    def _summarize_captions(self, captions: List[str], question: str) -> str:
        """Helper method to summarize frame captions."""
        try:
            summarize_prompt = f"""Question: {question}

Please provide a concise summary of these frame descriptions, focusing on details relevant to the question:

{chr(10).join(captions)}

Summarize the key visual elements, actions, and temporal relationships while maintaining chronological order."""

            response = self.qwen.client.chat.completions.create(
                model=self.qwen.model_name,
                messages=[{"role": "user", "content": summarize_prompt}],
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # if summarization fails, return None
            return None
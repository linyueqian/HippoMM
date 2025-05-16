import torch
import torch.nn as nn
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from PIL import Image
import numpy as np
import logging
from typing import List, Union, Dict, Any, Optional
from pathlib import Path
from faster_whisper import WhisperModel
import decord
import requests
from io import BytesIO
import cv2
import base64
logger = logging.getLogger(__name__)
from qwen_vl_utils import process_vision_info
from openai import OpenAI

class ImageBind(nn.Module):
    """ImageBind model for multimodal feature extraction"""
    
    def __init__(self, model_path: str = "pretrained/imagebind"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load ImageBind model (placeholder - actual implementation would use Meta's ImageBind)
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        
    def _load_model(self, model_path: str) -> nn.Module:
        """Load ImageBind model from path"""
        model = imagebind_model.imagebind_huge(pretrained=True).cuda()
        model.eval()
        return model
    
    def _load_audio_file(self, audio_path: str) -> str:
        """Helper function to verify audio file exists"""
        try:
            audio_path = Path(audio_path)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            return str(audio_path)
        except Exception as e:
            logger.error(f"Failed to verify audio file {audio_path}: {str(e)}")
            raise

    def load_data(
        self,
        inputs: Dict[str, Union[List[str], List[Image.Image], np.ndarray]],
        modalities: List[ModalityType]
    ) -> Dict[ModalityType, torch.Tensor]:
        """
        Load and transform data for different modalities.
        
        Args:
            inputs: Dictionary containing input data for each modality
                - For text: List of strings
                - For vision: List of image paths or PIL Images
                - For audio: List of audio paths (numpy arrays not supported for audio)
            modalities: List of ModalityType to process
            
        Returns:
            Dictionary mapping ModalityType to transformed tensor data
        """
        transformed_data = {}
        
        for modality in modalities:
            if modality not in inputs:
                continue
                
            try:
                if modality == ModalityType.TEXT:
                    text_list = inputs[modality]
                    transformed_data[modality] = data.load_and_transform_text(
                        text_list, 
                        device=self.device
                    )
                    
                elif modality == ModalityType.VISION:
                    vision_inputs = inputs[modality]
                    # Convert PIL Images to paths if needed
                    image_paths = [
                        img if isinstance(img, str) else img.filename 
                        for img in vision_inputs
                    ]
                    transformed_data[modality] = data.load_and_transform_vision_data(
                        image_paths, 
                        device=self.device
                    )
                    
                elif modality == ModalityType.AUDIO:
                    audio_inputs = inputs[modality]
                    # Only support file paths for audio
                    if not all(isinstance(x, str) for x in audio_inputs):
                        raise ValueError(
                            "Audio inputs must be file paths. "
                            "Direct tensor/array inputs are not supported."
                        )
                    
                    audio_paths = [
                        self._load_audio_file(audio_path) 
                        for audio_path in audio_inputs
                    ]
                    
                    transformed_data[modality] = data.load_and_transform_audio_data(
                        audio_paths,
                        device=self.device
                    )
            except Exception as e:
                logger.error(f"Error processing {modality}: {str(e)}")
                continue
        
        return transformed_data

    def forward(
        self, 
        inputs: Dict[ModalityType, torch.Tensor]
    ) -> Dict[ModalityType, torch.Tensor]:
        """
        Forward pass through the ImageBind model.
        
        Args:
            inputs: Dictionary mapping ModalityType to transformed tensor data
                   (output from load_data method)
                   
        Returns:
            Dictionary mapping ModalityType to embedding tensors
        """
        with torch.no_grad():
            embeddings = self.model(inputs)
            
        return embeddings

    def extract_features(
        self,
        inputs: Dict[str, Union[List[str], List[Image.Image], np.ndarray]],
        modalities: List[ModalityType]
    ) -> Dict[ModalityType, torch.Tensor]:
        """
        Convenience method to load data and extract features in one step.
        
        Args:
            inputs: Raw input data dictionary
            modalities: List of modalities to process
            
        Returns:
            Dictionary of embeddings for each modality
        """
        transformed_inputs = self.load_data(inputs, modalities)
        return self.forward(transformed_inputs)

class Whisper:
    """Whisper model for audio processing using faster-whisper"""
    
    def __init__(self, model_size: str = "large-v3"):
        """
        Initialize Whisper model
        
        Args:
            model_size: Size of the model to use (e.g., "large-v3", "distil-large-v3", "medium", etc.)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = WhisperModel(
            model_size,
            device=self.device,
            compute_type="float16" if self.device == "cuda" else "float32"
        )
        
    def __call__(self, audio: np.ndarray) -> torch.Tensor:
        """
        Process audio and extract features
        Note: Feature extraction is not directly supported in faster-whisper,
        consider using the transcribe method instead
        """
        raise NotImplementedError(
            "Feature extraction is not supported in faster-whisper. "
            "Please use the transcribe method instead."
        )
        
    def transcribe(
        self, 
        audio: np.ndarray, 
        language: str = "en",
        beam_size: int = 5
    ) -> List[dict]:
        """
        Transcribe audio to text with timestamps
        
        Args:
            audio: Audio array
            language: Language code (e.g., "en" for English)
            beam_size: Beam size for decoding
            
        Returns:
            List of segments containing:
                - text: transcribed text
                - start: start time in seconds
                - end: end time in seconds
        """
        segments, info = self.model.transcribe(
            audio,
            language=language,
            beam_size=beam_size,
            condition_on_previous_text=False
        )
        
        return [
            {
                "text": segment.text,
                "start": segment.start,
                "end": segment.end
            }
            for segment in segments
        ]

class QwenVL:
    """Qwen VL model for video and image understanding"""
    
    def __init__(self, model_name: str = "/home/yl768/ckpt/Qwen/Qwen2.5-VL-7B-Instruct"):
        """
        Initialize OpenAI client for Qwen VL
        
        Args:
            model_name: Path to the model
        """
        self.client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-IrR7Bwxtin0haWagUnPrBgq5PurnUz86")
        models = self.client.models.list()
        model = models.data[0].id
        print(f"Using model: {model}")
        self.model_name = model

    def _load_video_frames(
        self, 
        video_path: str,
        fps: float = 1.0,
        max_pixels: Optional[int] = None
    ) -> List[str]:
        """Load video frames using decord and save them as images"""
        # Remove file:// prefix if present
        video_path = video_path.replace('file://', '')
        
        if video_path.startswith(('http://', 'https://')):
            # Download video to temporary file
            response = requests.get(video_path)
            with open('temp_video.mp4', 'wb') as f:
                f.write(response.content)
            video_path = 'temp_video.mp4'

        # Load video with decord
        video_reader = decord.VideoReader(video_path)
        video_fps = video_reader.get_avg_fps()
        
        # Calculate frame indices based on desired fps
        total_frames = len(video_reader)
        step = int(video_fps / fps)
        frame_indices = list(range(0, total_frames, step))
        
        # Read frames
        frames = video_reader.get_batch(frame_indices).asnumpy()
        
        # Process and save frames
        frame_paths = []
        for i, frame in enumerate(frames):
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Save frame at original resolution
            frame_path = f"/tmp/frame_{i:03d}.jpg"
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            frame_paths.append(frame_path)
            
        return frame_paths

    def generate(
        self, 
        messages: List[Dict],
        max_new_tokens: int = 128,
    ) -> str:
        """
        Generate response using OpenAI API format
        
        Args:
            messages: List of message dictionaries containing text and vision content
            max_new_tokens: Maximum number of tokens to generate (not used in API call)
            
        Returns:
            Generated text response
        """
        # Convert messages to OpenAI format
        formatted_messages = []
        for msg in messages:
            content = []
            
            for item in msg["content"]:
                if item["type"] == "video":
                    # Handle video input - convert to list of frames
                    frame_paths = []
                    if isinstance(item["video"], list):
                        frame_paths = item["video"]  # Already a list of frame paths
                    else:
                        # For single video file, load frames
                        frame_paths = self._load_video_frames(
                            item["video"],
                            fps=item.get("fps", 1.0),
                            max_pixels=item.get("max_pixels", None)
                        )
                    
                    # Read and encode the frames
                    for frame_path in frame_paths:
                        with open(frame_path, "rb") as image_file:
                            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                        
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        })
                        
                elif item["type"] == "image_url":
                    # Pass through image URLs as is
                    content.append(item)
                    
                elif item["type"] == "text":
                    content.append({
                        "type": "text",
                        "text": item["text"]
                    })

            formatted_messages.append({
                "role": msg["role"],
                "content": content
            })
        
        # Make API call
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=formatted_messages,
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in QwenVL generate: {str(e)}")
            raise

if __name__ == "__main__":
    # Test Whisper ASR
    try:
        # Initialize Whisper model
        whisper = Whisper(model_size="distil-large-v3")
        
        # Audio file path
        audio_path = "/home/yl768/proj/hippo/HippoLM/data/audio.wav"
        
        # Verify file exists
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        print(f"Testing Whisper ASR on: {audio_path}")
        
        # Transcribe audio
        segments = whisper.transcribe(
            audio_path,
            language="en",
            beam_size=5
        )
        
        # Print results
        print("\nTranscription results:")
        print("-" * 50)
        for segment in segments:
            print(f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}")
        
    except Exception as e:
        print(f"Error during ASR testing: {str(e)}")
        import pdb; pdb.set_trace()

    # # test the imagebind model
    # imagebind = ImageBind()
    
    # # Use absolute paths or ensure files exist
    # asset_dir = "/home/yl768/proj/hippo/HippoLM/pretrained/ImageBind/.assets/"
    
    # # Verify paths exist
    # text_list = ["A dog.", "A car", "A bird"]
    # image_paths = [
    #     str(asset_dir + "dog_image.jpg"),
    #     str(asset_dir + "car_image.jpg"),
    #     str(asset_dir + "bird_image.jpg")
    # ]
    # audio_paths = [
    #     str(asset_dir + "dog_audio.wav"),
    #     str(asset_dir + "car_audio.wav"),
    #     str(asset_dir + "bird_audio.wav")
    # ]
    
    # # Verify files exist before proceeding
    # for path in image_paths + audio_paths:
    #     if not Path(path).exists():
    #         print(f"Warning: File not found: {path}")
    
    # inputs = {
    #     ModalityType.TEXT: text_list,
    #     ModalityType.VISION: image_paths,
    #     ModalityType.AUDIO: audio_paths
    # }
    # modalities = [ModalityType.TEXT, ModalityType.VISION, ModalityType.AUDIO]
    
    # # Test with individual modalities first
    # try:
    #     # Test text only
    #     text_inputs = {ModalityType.TEXT: text_list}
    #     text_embeddings = imagebind.extract_features(text_inputs, [ModalityType.TEXT])
    #     print("Successfully extracted text features")

    #     # Test vision only
    #     vision_inputs = {ModalityType.VISION: image_paths}
    #     vision_embeddings = imagebind.extract_features(vision_inputs, [ModalityType.VISION])
    #     print("Successfully extracted vision features")

    #     # Test audio only
    #     audio_inputs = {ModalityType.AUDIO: audio_paths}
    #     audio_embeddings = imagebind.extract_features(audio_inputs, [ModalityType.AUDIO])
    #     print("Successfully extracted audio features")

    #     # Test all modalities together
    #     all_inputs = {
    #         ModalityType.TEXT: text_list,
    #         ModalityType.VISION: image_paths,
    #         ModalityType.AUDIO: audio_paths
    #     }
    #     all_embeddings = imagebind.extract_features(all_inputs, modalities)
    #     print("Successfully extracted features for all modalities")
        
    #     import pdb; pdb.set_trace()
    # except Exception as e:
    #     print(f"Error during feature extraction: {str(e)}")
    #     import pdb; pdb.set_trace()
    
    # Test QwenVL model
    # try:
    #     print("\nTesting QwenVL model...")
    #     print("-" * 50)
        
    #     # Initialize QwenVL model
    #     qwen = QwenVL()
        
    #     # Test with two images
    #     image_messages = [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {
    #                         "url": "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"
    #                     }
    #                 },
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {
    #                         "url": "https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png"
    #                     }
    #                 },
    #                 {
    #                     "type": "text",
    #                     "text": "I have two very different images. They are not related at all. "
    #                     "Please describe the first image in one sentence, and then describe the second image in another sentence."
    #                 }
    #             ]
    #         }
    #     ]
        
    #     print("\nTesting image input...")
    #     try:
    #         response = qwen.generate(image_messages)
    #         print("Image response:", response)
    #     except Exception as e:
    #         print(f"Error processing images: {str(e)}")

    # except Exception as e:
    #     print(f"Error during QwenVL testing: {str(e)}")
    #     import pdb; pdb.set_trace()

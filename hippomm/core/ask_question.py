#!/usr/bin/env python3

import os
import sys
import yaml
import argparse
from pathlib import Path
import logging
from typing import Optional, Dict, Any

# add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from hippomm.core.hippocampal_memory import HippocampalMemory, QARecallSystem

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config/default_config.yaml") -> Dict[str, Any]:
    """load configuration from yaml file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_event(memory: HippocampalMemory, event_id: str) -> None:
    """load a specific theta event by id"""
    event = memory.load_theta_event(event_id)
    if event:
        print(f"\nEvent {event_id}:")
        print(f"Summary: {event.summary}")
        print(f"Time range: {event.start_time:.2f}s - {event.end_time:.2f}s")
        print(f"Number of frames: {len(event.frames)}")
        print(f"Number of audio segments: {len(event.audio_times)}")
        
        if event.frame_captions:
            print("\nFrame captions:")
            for i, caption in enumerate(event.frame_captions, 1):
                print(f"{i}. {caption}")
                
        if event.audio_transcription:
            print("\nAudio transcription:")
            for segment in event.audio_transcription:
                print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}")
    else:
        print(f"Event {event_id} not found")

def ask_question(memory: HippocampalMemory, question: str) -> None:
    """ask a question using the qa recall system"""
    # initialize qa recall system
    qa_system = QARecallSystem(memory, memory.config)

    # get answer
    result = qa_system.answer_question(question)
    
    # print results
    print(f"\nQuestion: {question}")
    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Reasoning: {result.reasoning}")
    
    if result.retrieved_segments:
        print(f"\nRetrieved {len(result.retrieved_segments)} relevant segments")

def list_events(memory: HippocampalMemory) -> None:
    """list all available theta events"""
    print("\nAvailable events:")
    for event_id, event_info in memory.event_index.items():
        print(f"\nEvent ID: {event_id}")
        print(f"Video ID: {event_info['video_id']}")
        print(f"Time range: {event_info['start_time']:.2f}s - {event_info['end_time']:.2f}s")
        print(f"Summary: {event_info['summary']}")

def main():
    parser = argparse.ArgumentParser(description="ask questions about stored memories")
    parser.add_argument("--config", default="config/default_config.yaml", help="path to config file")
    parser.add_argument("--memory-store", default="memory_store", help="directory containing stored memories")
    parser.add_argument("--question", help="question to ask about the memories")
    parser.add_argument("--event", help="event id to load and display")
    parser.add_argument("--list", action="store_true", help="list all available events")
    
    args = parser.parse_args()
    
    # load configuration
    config = load_config(args.config)
    
    # update memory store directory
    config['storage']['base_dir'] = args.memory_store
    
    # initialize memory system
    memory = HippocampalMemory(config)
    memory.load_theta_event(event_id=args.event)
    # memory.update_holistic_audio_transcription(memory.long_term_store[0], args.event.split("_")[0])
    ask_question(memory, args.question)
    
if __name__ == "__main__":
    main()

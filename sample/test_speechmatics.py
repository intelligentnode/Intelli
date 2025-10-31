"""
Example: Using Speechmatics for speech recognition with Intelli.

This demonstrates the simplest way to transcribe audio using Speechmatics:
- Uses self-contained RemoteRecognitionModel
- Supports batch transcription
- Automatic speaker diarization
- Multiple audio formats supported

Required:
- pip install intelli[speech]
- SPEECHMATICS_API_KEY in .env or environment

Usage:
    python sample/test_speechmatics.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Import Intelli components
from intelli.controller.remote_recognition_model import RemoteRecognitionModel, SupportedRecognitionModels
from intelli.model.input.text_recognition_input import SpeechRecognitionInput


def main():
    """Example usage of Speechmatics with Intelli."""
    
    print("=" * 60)
    print("SPEECHMATICS INTEGRATION EXAMPLE")
    print("=" * 60)
    
    # Get API key from .env file
    api_key = os.getenv('SPEECHMATICS_API_KEY')
    
    if not api_key:
        print("\n❌ Error: SPEECHMATICS_API_KEY not found in .env")
        print("\nPlease add to your .env file:")
        print("SPEECHMATICS_API_KEY=your_key_here")
        return
    
    print(f"\n✓ API key loaded (length: {len(api_key)})")
    
    # Initialize the recognition model
    print("\nInitializing Speechmatics recognition model...")
    recognition_model = RemoteRecognitionModel(
        key_value=api_key,
        provider=SupportedRecognitionModels['SPEECHMATICS']
    )
    print("✓ Model initialized")
    
    # Check for audio file - try multiple locations/formats
    audio_files = ["./temp/test.mp3", "./temp/test.wav", "./temp/temp.mp3"]
    audio_file = None
    
    for af in audio_files:
        if os.path.exists(af):
            audio_file = af
            break
    
    if not audio_file or not os.path.exists(audio_file):
        print("\nℹ️  No audio file found.")
        print("\nTried looking for:")
        for af in audio_files:
            print(f"  - {af}")
        print("\nTo test, add an audio file to ./temp/")
        print("  cp your_audio.mp3 ./temp/test.mp3")
        return
    
    print(f"\n✓ Audio file found: {audio_file}")
    file_size = os.path.getsize(audio_file)
    print(f"  File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    
    # Create input parameters
    print("\n🔄 Starting transcription...")
    
    try:
        recognition_input = SpeechRecognitionInput(
            audio_file_path=audio_file,
            language="en"  # Language code
        )
        
        # Perform speech recognition
        result = recognition_model.recognize_speech(recognition_input)
        
        print("\n✅ Transcription successful!")
        
        # Show configuration
        output_format = os.getenv('SPEECHMATICS_OUTPUT_FORMAT', 'speakers')
        sensitivity = os.getenv('SPEECHMATICS_SPEAKER_SENSITIVITY', '0.6')
        
        print("\nConfiguration:")
        print(f"  Output format: {output_format}")
        print(f"  Speaker sensitivity: {sensitivity}")
        
        print("\n" + "=" * 60)
        print("TRANSCRIPTION RESULT")
        print("=" * 60)
        print(result)
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during transcription: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

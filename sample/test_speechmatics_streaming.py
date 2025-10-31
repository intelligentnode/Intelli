"""
Example: Using Speechmatics real-time streaming with Intelli.

This demonstrates advanced streaming capabilities:
- WebSocket-based real-time transcription
- Automatic audio format conversion
- Speaker diarization in streaming mode
- Suitable for live audio processing

Required:
- pip install intelli[speech]
- SPEECHMATICS_API_KEY in .env or environment

Usage:
    python sample/test_speechmatics_streaming.py
"""

import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

from intelli.controller.remote_recognition_model import RemoteRecognitionModel, SupportedRecognitionModels


async def streaming_example():
    """Example of real-time streaming transcription."""
    
    print("=" * 60)
    print("SPEECHMATICS REAL-TIME STREAMING EXAMPLE")
    print("=" * 60)
    
    # Get API key
    api_key = os.getenv('SPEECHMATICS_API_KEY')
    
    if not api_key:
        print("\n❌ Error: SPEECHMATICS_API_KEY not found")
        return
    
    print(f"\n✓ API key loaded")
    
    # Initialize the wrapper directly to access streaming methods
    from intelli.wrappers.speechmatics_wrapper import SpeechmaticsWrapper
    
    wrapper = SpeechmaticsWrapper(api_key)
    
    print("\n⚠️  Note: This example requires PCM F32LE audio at 16kHz")
    print("For sustainable use, convert your audio to this format first.\n")
    
    # Example: Start a streaming session
    print("Starting streaming session...")
    
    try:
        # Start the session
        session = await wrapper.start_streaming_session(
            language="en",
            sample_rate=16000,
            enable_partials=True
        )
        
        print("✓ Session started")
        
        # In a real application, you would:
        # 1. Capture audio chunks
        # 2. Stream them using: await wrapper.stream_audio(session, audio_chunk)
        # 3. Receive transcripts using: async for result in wrapper.receive_transcripts(session)
        
        print("\nExample usage:")
        print("-" * 60)
        print("""
# Stream audio chunk
await wrapper.stream_audio(session, audio_chunk_bytes)

# Receive transcripts
async for result in wrapper.receive_transcripts(session):
    if result['type'] == 'partial':
        print(f"Partial: {result['transcript']}")
    elif result['type'] == 'transcript':
        print(f"Final: {result['transcript']} (speaker: {result['speaker']})")
    elif result['type'] == 'error':
        print(f"Error: {result['message']}")
        """)
        print("-" * 60)
        
        # Close the session
        await session.close()
        print("\n✓ Session closed")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def streaming_with_file_example():
    """Example of streaming a pre-recorded file with automatic conversion."""
    
    print("\n" + "=" * 60)
    print("STREAMING FILE EXAMPLE WITH AUTO-CONVERSION")
    print("=" * 60)
    
    api_key = os.getenv('SPEECHMATICS_API_KEY')
    
    if not api_key:
        print("\n❌ Error: SPEECHMATICS_API_KEY not found")
        return
    
    from intelli.wrappers.speechmatics_wrapper import SpeechmaticsWrapper
    
    wrapper = SpeechmaticsWrapper(api_key)
    
    # Find audio file
    audio_files = ["./temp/temp.mp3", "./temp/test.mp3", "./temp/test.wav"]
    audio_file = None
    
    for af in audio_files:
        if os.path.exists(af):
            audio_file = af
            break
    
    if not audio_file:
        print("\n⚠️  No audio file found for streaming example")
        print("Audio files are auto-detected in: ./temp/")
        return
    
    print(f"\n✓ Audio file: {audio_file}")
    
    # Detect format
    format_info = wrapper.detect_audio_format(audio_file)
    if format_info:
        print(f"  Format: {format_info['format']}/{format_info['subtype']}")
        print(f"  Sample rate: {format_info['sample_rate']} Hz")
        print(f"  Channels: {format_info['channels']}")
        print(f"  Duration: {format_info['duration']:.2f}s")
    
    try:
        print("\n🔄 Starting streaming session...")
        session = await wrapper.start_streaming_session(language="en")
        print("✓ Session started")
        
        print("\n🔄 Converting audio to PCM F32LE...")
        # Convert audio to PCM F32LE format
        pcm_data = wrapper.convert_audio_to_pcm_f32le(audio_file)
        print(f"✓ Converted ({len(pcm_data)} bytes)")
        
        print("\n🔄 Streaming audio...")
        # Stream the audio in chunks
        chunk_size = 1024 * 100  # 100KB chunks
        for i in range(0, len(pcm_data), chunk_size):
            chunk = pcm_data[i:i+chunk_size]
            await wrapper.stream_audio(session, chunk)
            print(f"  Sent chunk {i//chunk_size + 1} ({len(chunk)} bytes)")
        
        print("✓ Audio streamed")
        
        print("\n📝 Receiving transcripts...")
        print("=" * 60)
        
        # Receive transcripts
        transcript_count = 0
        
        async for result in wrapper.receive_transcripts(session):
            if result['type'] == 'partial':
                # Skip partial transcripts
                continue
            elif result['type'] == 'transcript':
                transcript_count += 1
                speaker = result.get('speaker', 'unknown')
                text = result['transcript']
                print(f"Speaker {speaker}: {text}")
            elif result['type'] == 'error':
                print(f"\n❌ Error: {result['message']}")
                break
        
        print("=" * 60)
        print(f"\n✓ Received {transcript_count} transcripts")
        
        # Close the session
        await session.close()
        print("✓ Session closed")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run streaming examples."""
    asyncio.run(streaming_example())
    asyncio.run(streaming_with_file_example())


if __name__ == "__main__":
    main()


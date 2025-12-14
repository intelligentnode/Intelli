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
        tokens = result.get('tokens', [])
        confidence = result.get('confidence', [])
        if tokens and confidence and len(tokens) == len(confidence):
            formatted = " ".join([f"{token} [{conf:.2f}]" if conf is not None else f"{token} [N/A]" for token, conf in zip(tokens, confidence)])
            print(f"Partial: {formatted}")
        else:
            print(f"Partial: {result.get('transcript', '')}")
    elif result['type'] == 'final':
        tokens = result.get('tokens', [])
        speaker = result.get('speaker', 'unknown')
        confidence = result.get('confidence', [])
        if tokens and confidence and len(tokens) == len(confidence):
            formatted = " ".join([f"{token} [{conf:.2f}]" if conf is not None else f"{token} [N/A]" for token, conf in zip(tokens, confidence)])
            print(f"Final: {formatted} (speaker: {speaker})")
        else:
            print(f"Final: {result.get('transcript', '')} (speaker: {speaker})")
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
        session = await wrapper.start_streaming_session(language="en", enable_partials=True)
        print("✓ Session started with partials enabled")
        
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
        partial_count = 0
        final_confidences = []
        partial_confidences = []
        
        async for result in wrapper.receive_transcripts(session):
            if result['type'] == 'partial':
                partial_count += 1
                tokens = result.get('tokens', [])
                confidence = result.get('confidence', [])
                if tokens and confidence and len(tokens) == len(confidence):
                    formatted = " ".join([f"{token} [{conf:.2f}]" if conf is not None else f"{token} [N/A]" for token, conf in zip(tokens, confidence)])
                    partial_confidences.extend(confidence)
                    # Show partials (can be commented out to reduce noise)
                    print(f"[Partial #{partial_count}] {formatted}")
                else:
                    print(f"[Partial #{partial_count}] {result.get('transcript', '')}")
            elif result['type'] == 'final':
                transcript_count += 1
                speaker = result.get('speaker', 'unknown')
                tokens = result.get('tokens', [])
                confidence = result.get('confidence', [])
                if tokens and confidence and len(tokens) == len(confidence):
                    formatted = " ".join([f"{token} [{conf:.2f}]" if conf is not None else f"{token} [N/A]" for token, conf in zip(tokens, confidence)])
                    final_confidences.extend(confidence)
                    print(f"[Final #{transcript_count}] Speaker {speaker}: {formatted}")
                else:
                    print(f"[Final #{transcript_count}] Speaker {speaker}: {result.get('transcript', '')}")
            elif result['type'] == 'error':
                print(f"\n❌ Error: {result['message']}")
                break
        
        print("=" * 60)
        print(f"\n✓ Received {transcript_count} final transcripts and {partial_count} partial transcripts")
        
        # Show confidence statistics (filter out None values)
        valid_final_confidences = [c for c in final_confidences if c is not None]
        if valid_final_confidences:
            avg_final_conf = sum(valid_final_confidences) / len(valid_final_confidences)
            min_final_conf = min(valid_final_confidences)
            max_final_conf = max(valid_final_confidences)
            print(f"  Final confidence: avg={avg_final_conf:.2f}, min={min_final_conf:.2f}, max={max_final_conf:.2f}")
        
        valid_partial_confidences = [c for c in partial_confidences if c is not None]
        if valid_partial_confidences:
            avg_partial_conf = sum(valid_partial_confidences) / len(valid_partial_confidences)
            min_partial_conf = min(valid_partial_confidences)
            max_partial_conf = max(valid_partial_confidences)
            print(f"  Partial confidence: avg={avg_partial_conf:.2f}, min={min_partial_conf:.2f}, max={max_partial_conf:.2f}")
        
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


#!/usr/bin/env python3
"""
Test script for Azure Whisper Wrapper

This script tests all functionality of the Azure Whisper wrapper including:
- Basic transcription from file
- Transcription from bytes
- Error handling
- Different languages

Usage:
    python sample/test_azure_whisper.py

Environment variables required:
    AZURE_OPENAI_API_KEY - Azure OpenAI API key
    AZURE_OPENAI_ENDPOINT - Azure OpenAI endpoint URL
    AZURE_WHISPER_DEPLOYMENT_NAME - (optional) Whisper deployment name, defaults to 'whisper'
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from intelli.wrappers.azure_whisper_wrapper import AzureWhisperWrapper

load_dotenv()


def test_initialization():
    """Test wrapper initialization."""
    print("\n" + "="*60)
    print("TEST 1: Initialization")
    print("="*60)
    
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    deployment_name = os.getenv('AZURE_WHISPER_DEPLOYMENT_NAME', 'whisper')
    
    if not api_key or not endpoint:
        print("❌ FAILED: Missing required environment variables:")
        print("   - AZURE_OPENAI_API_KEY")
        print("   - AZURE_OPENAI_ENDPOINT")
        return None
    
    try:
        wrapper = AzureWhisperWrapper(
            api_key=api_key,
            endpoint=endpoint,
            deployment_name=deployment_name
        )
        print(f"✅ SUCCESS: Azure Whisper wrapper initialized")
        print(f"   Endpoint: {endpoint}")
        print(f"   Deployment: {deployment_name}")
        print(f"   Timeout: {wrapper.timeout}s")
        print(f"   Max retries: {wrapper.max_retries}")
        return wrapper
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return None


def test_invalid_initialization():
    """Test initialization with invalid parameters."""
    print("\n" + "="*60)
    print("TEST 2: Invalid Initialization")
    print("="*60)
    
    # Test with empty API key
    try:
        AzureWhisperWrapper(api_key="", endpoint="https://test.openai.azure.com")
        print("❌ FAILED: Should have raised ValueError for empty API key")
    except ValueError as e:
        print(f"✅ SUCCESS: Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"❌ FAILED: Raised unexpected exception: {e}")
    
    # Test with empty endpoint
    try:
        AzureWhisperWrapper(api_key="test-key", endpoint="")
        print("❌ FAILED: Should have raised ValueError for empty endpoint")
    except ValueError as e:
        print(f"✅ SUCCESS: Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"❌ FAILED: Raised unexpected exception: {e}")


def test_transcribe_from_file(wrapper):
    """Test transcription from audio file."""
    print("\n" + "="*60)
    print("TEST 3: Transcribe from File")
    print("="*60)
    
    test_audio_path = 'temp/temp.mp3'
    
    if not os.path.exists(test_audio_path):
        print(f"⚠️  SKIPPED: Test audio file not found: {test_audio_path}")
        print("   Please add an audio file to test transcription")
        return
    
    try:
        result = wrapper.transcribe(
            audio_file=test_audio_path,
            language='en',
            prompt_guide='medical terminology'
        )
        
        print(f"✅ SUCCESS: Transcription completed")
        print(f"   Audio file: {test_audio_path}")
        print(f"   Transcription length: {len(result)} characters")
        print(f"   Transcription preview: {result[:100]}...")
        print(f"\n   Full transcription:\n   {result}")
    except Exception as e:
        print(f"❌ FAILED: {e}")


def test_transcribe_from_bytes(wrapper):
    """Test transcription from audio bytes."""
    print("\n" + "="*60)
    print("TEST 4: Transcribe from Bytes")
    print("="*60)
    
    test_audio_path = 'temp/temp.mp3'
    
    if not os.path.exists(test_audio_path):
        print(f"⚠️  SKIPPED: Test audio file not found: {test_audio_path}")
        return
    
    try:
        # Read audio as bytes
        with open(test_audio_path, 'rb') as f:
            audio_bytes = f.read()
        
        result = wrapper.transcribe(
            audio_file=audio_bytes,
            file_name='test_audio.mp3',
            language='en'
        )
        
        print(f"✅ SUCCESS: Transcription from bytes completed")
        print(f"   Audio size: {len(audio_bytes)} bytes")
        print(f"   Transcription length: {len(result)} characters")
        print(f"   Transcription preview: {result[:100]}...")
    except Exception as e:
        print(f"❌ FAILED: {e}")


def test_invalid_audio_file(wrapper):
    """Test error handling for invalid audio file."""
    print("\n" + "="*60)
    print("TEST 6: Invalid Audio File Handling")
    print("="*60)
    
    try:
        wrapper.transcribe(audio_file="/nonexistent/file.mp3")
        print("❌ FAILED: Should have raised ValueError for nonexistent file")
    except ValueError as e:
        print(f"✅ SUCCESS: Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"⚠️  WARNING: Raised unexpected exception: {e}")
    
    try:
        wrapper.transcribe(audio_file=b"fake audio data")
        print("❌ FAILED: Should have raised ValueError for bytes without filename")
    except ValueError as e:
        print(f"✅ SUCCESS: Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"⚠️  WARNING: Raised unexpected exception: {e}")


def test_initialization_with_custom_timeout_retries():
    """Test initialization with custom timeout and retries."""
    print("\n" + "="*60)
    print("TEST 4: Custom Timeout and Retries Configuration")
    print("="*60)
    
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    deployment_name = os.getenv('AZURE_WHISPER_DEPLOYMENT_NAME', 'whisper')
    
    if not api_key or not endpoint:
        print("⚠️  SKIPPED: Missing Azure OpenAI credentials")
        return
    
    try:
        wrapper = AzureWhisperWrapper(
            api_key=api_key,
            endpoint=endpoint,
            deployment_name=deployment_name,
            timeout=120.0,
            max_retries=5
        )
        print(f"✅ SUCCESS: Wrapper initialized with custom settings")
        print(f"   Timeout: {wrapper.timeout}s")
        print(f"   Max retries: {wrapper.max_retries}")
    except Exception as e:
        print(f"❌ FAILED: {e}")


def test_different_languages(wrapper):
    """Test transcription with different languages."""
    print("\n" + "="*60)
    print("TEST 5: Different Languages")
    print("="*60)
    
    test_audio_path = 'temp/temp.mp3'
    
    if not os.path.exists(test_audio_path):
        print(f"⚠️  SKIPPED: Test audio file not found: {test_audio_path}")
        return
    
    languages = ['en', 'es', 'fr', 'de']
    
    for lang in languages:
        try:
            result = wrapper.transcribe(
                audio_file=test_audio_path,
                language=lang
            )
            print(f"✅ SUCCESS: Transcription in {lang} completed ({len(result)} chars)")
        except Exception as e:
            print(f"⚠️  WARNING: Failed for language {lang}: {e}")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("AZURE WHISPER WRAPPER - COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    # Initialize wrapper
    wrapper = test_initialization()
    
    if wrapper is None:
        print("\n❌ Cannot proceed without initialized wrapper. Please check your configuration.")
        return
    
    # Run all tests
    test_invalid_initialization()
    test_initialization_with_custom_timeout_retries()
    test_transcribe_from_file(wrapper)
    test_transcribe_from_bytes(wrapper)
    test_invalid_audio_file(wrapper)
    test_different_languages(wrapper)
    
    print("\n" + "="*60)
    print("TEST SUITE COMPLETED")
    print("="*60)


if __name__ == '__main__':
    main()


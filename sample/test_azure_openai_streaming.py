#!/usr/bin/env python3
"""
Dedicated test script for Azure OpenAI Streaming

This script tests streaming functionality of the Azure OpenAI wrapper including:
- Basic streaming
- Streaming with different parameters
- Streaming error handling
- Streaming with GPT-5 models
- Real-time output demonstration

Usage:
    python sample/test_azure_openai_streaming.py

Environment variables required:
    AZURE_OPENAI_API_KEY - Azure OpenAI API key
    AZURE_OPENAI_ENDPOINT - Azure OpenAI endpoint URL
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from intelli.wrappers.azure_openai_wrapper import AzureOpenAIWrapper

load_dotenv()


def test_initialization():
    """Test wrapper initialization."""
    print("\n" + "="*60)
    print("INITIALIZATION")
    print("="*60)
    
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    
    if not api_key or not endpoint:
        print("❌ FAILED: Missing required environment variables:")
        print("   - AZURE_OPENAI_API_KEY")
        print("   - AZURE_OPENAI_ENDPOINT")
        return None
    
    try:
        wrapper = AzureOpenAIWrapper(
            api_key=api_key,
            endpoint=endpoint
        )
        print(f"✅ SUCCESS: Azure OpenAI wrapper initialized")
        print(f"   Endpoint: {endpoint}")
        print(f"   Timeout: {wrapper.timeout}s")
        print(f"   Max retries: {wrapper.max_retries}")
        return wrapper
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return None


def test_basic_streaming(wrapper):
    """Test basic streaming functionality."""
    print("\n" + "="*60)
    print("TEST 1: Basic Streaming")
    print("="*60)
    
    model = os.getenv('AZURE_OPENAI_CHAT_MODEL', 'gpt-4o')
    
    try:
        print("   Streaming response (real-time):")
        print("   ", end="", flush=True)
        
        full_content = ""
        chunks_received = 0
        
        for chunk in wrapper.generate_chat_text_stream({
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Count from 1 to 5, one number per line."}
            ],
            "temperature": 0.7
        }):
            chunks_received += 1
            
            if "choices" in chunk and len(chunk["choices"]) > 0:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                
                if content:
                    print(content, end="", flush=True)
                    full_content += content
            
            if chunk["choices"][0].get("finish_reason"):
                print()  # New line after streaming
                if chunk.get("usage"):
                    usage = chunk["usage"]
                    print(f"\n   ✅ SUCCESS: Streaming completed")
                    print(f"   Chunks received: {chunks_received}")
                    print(f"   Prompt tokens: {usage.get('prompt_tokens', 0)}")
                    print(f"   Completion tokens: {usage.get('completion_tokens', 0)}")
                    print(f"   Total tokens: {usage.get('total_tokens', 0)}")
                    print(f"   Full content length: {len(full_content)} characters")
                    print(f"   Full content: {full_content}")
                break
        
        if chunks_received == 0:
            print("   ❌ FAILED: No chunks received")
        elif len(full_content) == 0:
            print("   ⚠️  WARNING: Received chunks but no content")
        
    except Exception as e:
        print(f"\n   ❌ FAILED: {e}")


def test_streaming_with_temperature(wrapper):
    """Test streaming with different temperature settings."""
    print("\n" + "="*60)
    print("TEST 2: Streaming with Different Temperatures")
    print("="*60)
    
    model = os.getenv('AZURE_OPENAI_CHAT_MODEL', 'gpt-4o')
    temperatures = [0.0, 0.7, 1.0]
    
    for temp in temperatures:
        try:
            print(f"\n   Testing with temperature={temp}:")
            print("   ", end="", flush=True)
            
            content_chunks = []
            for chunk in wrapper.generate_chat_text_stream({
                "model": model,
                "messages": [
                    {"role": "user", "content": "Say hello."}
                ],
                "temperature": temp
            }):
                if chunk["choices"][0]["delta"].get("content"):
                    content = chunk["choices"][0]["delta"]["content"]
                    print(content, end="", flush=True)
                    content_chunks.append(content)
                
                if chunk["choices"][0].get("finish_reason"):
                    print()
                    break
            
            print(f"   ✅ Temperature {temp} streaming completed")
        except Exception as e:
            print(f"\n   ❌ FAILED for temperature {temp}: {e}")


def test_streaming_with_max_tokens(wrapper):
    """Test streaming with max_tokens limit."""
    print("\n" + "="*60)
    print("TEST 3: Streaming with Max Tokens")
    print("="*60)
    
    model = os.getenv('AZURE_OPENAI_CHAT_MODEL', 'gpt-4o')
    
    try:
        print("   Streaming with max_tokens=30:")
        print("   ", end="", flush=True)
        
        full_content = ""
        chunks_received = 0
        
        for chunk in wrapper.generate_chat_text_stream({
            "model": model,
            "messages": [
                {"role": "user", "content": "Tell me a short story about a cat."}
            ],
            "max_tokens": 30
        }):
            chunks_received += 1
            
            if chunk["choices"][0]["delta"].get("content"):
                content = chunk["choices"][0]["delta"]["content"]
                print(content, end="", flush=True)
                full_content += content
            
            if chunk["choices"][0].get("finish_reason"):
                print()
                print(f"   ✅ SUCCESS: Streaming stopped at {chunks_received} chunks")
                print(f"   Content length: {len(full_content)} characters")
                print(f"   Finish reason: {chunk['choices'][0]['finish_reason']}")
                break
        
    except Exception as e:
        print(f"\n   ❌ FAILED: {e}")


def test_streaming_conversation(wrapper):
    """Test streaming in a multi-turn conversation."""
    print("\n" + "="*60)
    print("TEST 4: Streaming Multi-Turn Conversation")
    print("="*60)
    
    model = os.getenv('AZURE_OPENAI_CHAT_MODEL', 'gpt-4o')
    
    try:
        # First message
        print("   Turn 1 - User: What is Python?")
        print("   Assistant: ", end="", flush=True)
        
        assistant_response = ""
        for chunk in wrapper.generate_chat_text_stream({
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is Python?"}
            ]
        }):
            if chunk["choices"][0]["delta"].get("content"):
                content = chunk["choices"][0]["delta"]["content"]
                print(content, end="", flush=True)
                assistant_response += content
            
            if chunk["choices"][0].get("finish_reason"):
                print("\n")
                break
        
        # Second message
        print("   Turn 2 - User: What can you do with it?")
        print("   Assistant: ", end="", flush=True)
        
        for chunk in wrapper.generate_chat_text_stream({
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": assistant_response},
                {"role": "user", "content": "What can you do with it?"}
            ]
        }):
            if chunk["choices"][0]["delta"].get("content"):
                print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
            
            if chunk["choices"][0].get("finish_reason"):
                print("\n")
                print(f"   ✅ SUCCESS: Multi-turn streaming conversation completed")
                break
        
    except Exception as e:
        print(f"\n   ❌ FAILED: {e}")


def test_streaming_gpt5(wrapper):
    """Test streaming with GPT-5 model."""
    print("\n" + "="*60)
    print("TEST 5: Streaming with GPT-5 Model")
    print("="*60)
    
    model = 'gpt-5-mini'
    
    try:
        print("   Testing GPT-5-mini streaming (without temperature/max_tokens):")
        print("   ", end="", flush=True)
        
        chunks_received = 0
        for chunk in wrapper.generate_chat_text_stream({
            "model": model,
            "messages": [
                {"role": "user", "content": "Hello! Say hi back."}
            ]
        }):
            chunks_received += 1
            
            if chunk["choices"][0]["delta"].get("content"):
                print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
            
            if chunk["choices"][0].get("finish_reason"):
                print()
                print(f"   ✅ SUCCESS: GPT-5 streaming completed ({chunks_received} chunks)")
                break
            
            if chunks_received > 50:  # Safety limit
                print()
                print(f"   ⚠️  Reached safety limit at {chunks_received} chunks")
                break
        
    except Exception as e:
        error_msg = str(e).lower()
        if "gpt-5-mini" in error_msg or "not available" in error_msg:
            print(f"   ⚠️  SKIPPED: GPT-5-mini model not available: {e}")
        else:
            print(f"   ❌ FAILED: {e}")


def test_streaming_gpt5_rejects_parameters(wrapper):
    """Test that GPT-5 streaming rejects unsupported parameters."""
    print("\n" + "="*60)
    print("TEST 6: GPT-5 Streaming Parameter Validation")
    print("="*60)
    
    model = 'gpt-5-mini'
    
    # Test temperature rejection
    try:
        print("   Testing GPT-5-mini WITH temperature parameter (should fail)...")
        for chunk in wrapper.generate_chat_text_stream({
            "model": model,
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7
        }):
            pass
        print("   ❌ FAILED: Should have raised ValueError for temperature")
    except ValueError as e:
        if "temperature" in str(e).lower():
            print(f"   ✅ SUCCESS: Correctly rejected temperature parameter")
            print(f"   Error: {e}")
        else:
            print(f"   ⚠️  Unexpected ValueError: {e}")
    except Exception as e:
        if "gpt-5-mini" not in str(e).lower():
            print(f"   ⚠️  Unexpected error: {e}")
    
    # Test max_tokens rejection
    try:
        print("\n   Testing GPT-5-mini WITH max_tokens parameter (should fail)...")
        for chunk in wrapper.generate_chat_text_stream({
            "model": model,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100
        }):
            pass
        print("   ❌ FAILED: Should have raised ValueError for max_tokens")
    except ValueError as e:
        if "max_tokens" in str(e).lower():
            print(f"   ✅ SUCCESS: Correctly rejected max_tokens parameter")
            print(f"   Error: {e}")
        else:
            print(f"   ⚠️  Unexpected ValueError: {e}")
    except Exception as e:
        if "gpt-5-mini" not in str(e).lower():
            print(f"   ⚠️  Unexpected error: {e}")


def test_streaming_error_handling(wrapper):
    """Test streaming error handling."""
    print("\n" + "="*60)
    print("TEST 7: Streaming Error Handling")
    print("="*60)
    
    # Test with invalid model
    try:
        print("   Testing with invalid model...")
        chunks_received = 0
        for chunk in wrapper.generate_chat_text_stream({
            "model": "invalid-model-name-12345",
            "messages": [{"role": "user", "content": "Hello"}]
        }):
            chunks_received += 1
            if chunks_received > 1:
                break
        
        print("   ❌ FAILED: Should have raised an error for invalid model")
    except RuntimeError as e:
        print(f"   ✅ SUCCESS: Correctly raised RuntimeError: {e}")
    except Exception as e:
        print(f"   ⚠️  Raised unexpected exception: {e}")


def test_streaming_chunk_structure(wrapper):
    """Test that streaming chunks have correct structure."""
    print("\n" + "="*60)
    print("TEST 8: Streaming Chunk Structure Validation")
    print("="*60)
    
    model = os.getenv('AZURE_OPENAI_CHAT_MODEL', 'gpt-4o')
    
    try:
        chunks_validated = 0
        
        for chunk in wrapper.generate_chat_text_stream({
            "model": model,
            "messages": [
                {"role": "user", "content": "Say OK"}
            ]
        }):
            # Validate chunk structure
            assert "choices" in chunk, "Chunk must have 'choices' key"
            assert len(chunk["choices"]) > 0, "Chunk must have at least one choice"
            
            choice = chunk["choices"][0]
            assert "delta" in choice, "Choice must have 'delta' key"
            assert isinstance(choice["delta"], dict), "Delta must be a dictionary"
            
            chunks_validated += 1
            
            if choice.get("finish_reason"):
                assert choice["finish_reason"] is not None, "finish_reason should not be None"
                if chunk.get("usage"):
                    assert isinstance(chunk["usage"], dict), "Usage must be a dictionary"
                break
        
        print(f"   ✅ SUCCESS: Validated {chunks_validated} chunks with correct structure")
        
    except Exception as e:
        print(f"   ❌ FAILED: {e}")


def main():
    """Run all streaming tests."""
    print("\n" + "="*60)
    print("AZURE OPENAI WRAPPER - STREAMING TEST SUITE")
    print("="*60)
    
    # Initialize wrapper
    wrapper = test_initialization()
    
    if wrapper is None:
        print("\n❌ Cannot proceed without initialized wrapper. Please check your configuration.")
        return
    
    # Run all streaming tests
    test_basic_streaming(wrapper)
    test_streaming_with_temperature(wrapper)
    test_streaming_with_max_tokens(wrapper)
    test_streaming_conversation(wrapper)
    test_streaming_gpt5(wrapper)
    test_streaming_gpt5_rejects_parameters(wrapper)
    test_streaming_error_handling(wrapper)
    test_streaming_chunk_structure(wrapper)
    
    print("\n" + "="*60)
    print("STREAMING TEST SUITE COMPLETED")
    print("="*60)


if __name__ == '__main__':
    main()


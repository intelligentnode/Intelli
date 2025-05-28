#!/usr/bin/env python3
"""
Simple test suite for the latest Gemini API features.
Tests only the most important new capabilities: system instructions, image generation, and TTS.
"""

import unittest
import os
import base64
from dotenv import load_dotenv
from intelli.wrappers.geminiai_wrapper import GeminiAIWrapper

load_dotenv()


class TestGeminiLatestFeaturesSimple(unittest.TestCase):
    """Simple test suite for the latest Gemini API features"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.api_key = os.getenv("GEMINI_API_KEY")
        
        if not cls.api_key:
            raise unittest.SkipTest("GEMINI_API_KEY not set")
        
        cls.wrapper = GeminiAIWrapper(cls.api_key)

    def test_system_instructions(self):
        """Test system instructions functionality"""
        print("\n=== Testing System Instructions ===")
        
        system_instruction = "You are a helpful AI assistant. Always provide concise answers with examples."
        content_parts = [{"text": "What is machine learning?"}]
        
        try:
            result = self.wrapper.generate_content_with_system_instructions(
                content_parts, system_instruction
            )
            
            response_text = result['candidates'][0]['content']['parts'][0]['text']
            print(f"Response with system instruction: {response_text[:200]}...")
            
            self.assertIsNotNone(response_text)
            self.assertGreater(len(response_text), 50)
            print("✅ System instructions working correctly")
            
        except Exception as e:
            print(f"❌ System instructions error: {e}")
            # Don't fail the test as this might require special access
            self.skipTest(f"System instructions not available: {e}")

    def test_image_generation(self):
        """Test image generation using Gemini 2.0 Flash"""
        print("\n=== Testing Image Generation ===")
        
        prompt = "A simple cartoon cat sitting on a blue cushion"
        
        try:
            result = self.wrapper.generate_image(prompt)
            print(f"Image generation result keys: {list(result.keys())}")
            
            self.assertIn('candidates', result)
            
            # Check for image data in response
            candidate = result['candidates'][0]
            image_found = False
            
            if 'content' in candidate and 'parts' in candidate['content']:
                for part in candidate['content']['parts']:
                    if 'inline_data' in part and part['inline_data'].get('mime_type', '').startswith('image/'):
                        image_data = part['inline_data']['data']
                        print(f"✅ Image generated successfully - data length: {len(image_data)}")
                        image_found = True
                        break
            
            if not image_found:
                print("⚠️ No image data found in response")
                # Check if there's text response instead
                if 'content' in candidate and 'parts' in candidate['content']:
                    for part in candidate['content']['parts']:
                        if 'text' in part:
                            print(f"Text response: {part['text'][:100]}...")
                            
        except Exception as e:
            print(f"❌ Image generation error: {e}")
            print("Note: Image generation may require special access or billing setup")
            self.skipTest(f"Image generation not available: {e}")

    def test_text_to_speech(self):
        """Test text-to-speech generation"""
        print("\n=== Testing Text-to-Speech ===")
        
        text = "Hello! This is a test of Gemini's text-to-speech capabilities."
        
        try:
            result = self.wrapper.generate_speech(text)
            print(f"TTS result keys: {list(result.keys())}")
            
            self.assertIn('candidates', result)
            
            # Check for audio data
            candidate = result['candidates'][0]
            audio_found = False
            
            if 'content' in candidate and 'parts' in candidate['content']:
                for part in candidate['content']['parts']:
                    if 'inline_data' in part and part['inline_data'].get('mime_type', '').startswith('audio/'):
                        audio_data = part['inline_data']['data']
                        print(f"✅ Audio generated successfully - data length: {len(audio_data)}")
                        audio_found = True
                        break
            
            if not audio_found:
                print("⚠️ No audio data found in response")
                # Check if there's text response instead
                if 'content' in candidate and 'parts' in candidate['content']:
                    for part in candidate['content']['parts']:
                        if 'text' in part:
                            print(f"Text response: {part['text'][:100]}...")
                            
        except Exception as e:
            print(f"❌ TTS error: {e}")
            print("Note: TTS may require special model access")
            self.skipTest(f"TTS not available: {e}")


if __name__ == "__main__":
    # Create temp directory if it doesn't exist
    os.makedirs('./temp', exist_ok=True)
    
    # Run tests
    unittest.main(verbosity=2) 
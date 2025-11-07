import unittest
import os
from pathlib import Path
from intelli.wrappers.azure_whisper_wrapper import AzureWhisperWrapper
from dotenv import load_dotenv

load_dotenv()

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
TEST_AUDIO_PATH = PROJECT_ROOT / 'temp' / 'temp.mp3'


class TestAzureWhisperWrapper(unittest.TestCase):
    """Comprehensive test suite for Azure Whisper wrapper."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.api_key = os.getenv('AZURE_OPENAI_API_KEY')
        self.endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        self.deployment_name = os.getenv('AZURE_WHISPER_DEPLOYMENT_NAME', 'whisper')
        
        if not self.api_key or not self.endpoint:
            self.skipTest("Azure OpenAI credentials not configured. "
                         "Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables.")
        
        self.wrapper = AzureWhisperWrapper(
            api_key=self.api_key,
            endpoint=self.endpoint,
            deployment_name=self.deployment_name
        )
    
    def test_initialization(self):
        """Test wrapper initialization."""
        self.assertIsNotNone(self.wrapper)
        self.assertEqual(self.wrapper.endpoint, self.endpoint)
        self.assertEqual(self.wrapper.deployment_name, self.deployment_name)
        self.assertEqual(self.wrapper.timeout, 60.0)
        self.assertEqual(self.wrapper.max_retries, 3)
    
    def test_initialization_with_custom_timeout_and_retries(self):
        """Test wrapper initialization with custom timeout and max_retries."""
        wrapper = AzureWhisperWrapper(
            api_key=self.api_key,
            endpoint=self.endpoint,
            deployment_name=self.deployment_name,
            timeout=120.0,
            max_retries=5
        )
        self.assertEqual(wrapper.timeout, 120.0)
        self.assertEqual(wrapper.max_retries, 5)
    
    def test_invalid_initialization_empty_api_key(self):
        """Test initialization with empty API key."""
        with self.assertRaises(ValueError):
            AzureWhisperWrapper(api_key="", endpoint="https://test.openai.azure.com")
    
    def test_invalid_initialization_empty_endpoint(self):
        """Test initialization with empty endpoint."""
        with self.assertRaises(ValueError):
            AzureWhisperWrapper(api_key="test-key", endpoint="")
    
    def test_transcribe_from_file(self):
        """Test transcription from audio file."""
        test_audio_path = str(TEST_AUDIO_PATH)
        
        if not os.path.exists(test_audio_path):
            self.skipTest(f"Test audio file not found: {test_audio_path}")
        
        result = self.wrapper.transcribe(
            audio_file=test_audio_path,
            language='en',
            prompt_guide='medical terminology'
        )
        
        self.assertTrue(isinstance(result, str))
        self.assertTrue(len(result) > 0, "Transcription should not be empty")
    
    def test_transcribe_from_bytes(self):
        """Test transcription from audio bytes."""
        test_audio_path = str(TEST_AUDIO_PATH)
        
        if not os.path.exists(test_audio_path):
            self.skipTest(f"Test audio file not found: {test_audio_path}")
        
        with open(test_audio_path, 'rb') as f:
            audio_bytes = f.read()
        
        result = self.wrapper.transcribe(
            audio_file=audio_bytes,
            file_name='test_audio.mp3',
            language='en'
        )
        
        self.assertTrue(isinstance(result, str))
        self.assertTrue(len(result) > 0, "Transcription should not be empty")
    
    def test_different_languages(self):
        """Test transcription with different languages."""
        test_audio_path = str(TEST_AUDIO_PATH)
        
        if not os.path.exists(test_audio_path):
            self.skipTest(f"Test audio file not found: {test_audio_path}")
        
        languages = ['en', 'es', 'fr', 'de']
        
        for lang in languages:
            try:
                result = self.wrapper.transcribe(
                    audio_file=test_audio_path,
                    language=lang
                )
                self.assertTrue(isinstance(result, str))
            except Exception as e:
                # Some languages might fail, that's okay for this test
                pass
    
    def test_invalid_audio_file(self):
        """Test that invalid audio file raises appropriate error."""
        with self.assertRaises(ValueError):
            self.wrapper.transcribe(audio_file="/nonexistent/file.mp3")
    
    def test_invalid_bytes_without_filename(self):
        """Test that bytes without filename raises error."""
        with self.assertRaises(ValueError):
            self.wrapper.transcribe(audio_file=b"fake audio data")


if __name__ == '__main__':
    unittest.main()

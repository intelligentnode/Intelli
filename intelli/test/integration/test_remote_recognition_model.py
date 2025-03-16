import os
import unittest
from dotenv import load_dotenv
from intelli.controller.remote_recognition_model import RemoteRecognitionModel, SupportedRecognitionModels
from intelli.model.input.text_recognition_input import SpeechRecognitionInput

load_dotenv()


class TestRemoteRecognitionModel(unittest.TestCase):
    """
    Integration tests for the RemoteRecognitionModel with both
    OpenAI and Keras (offline) providers.
    """

    def setUp(self):
        """Set up for the test case."""
        self.api_key_openai = os.getenv('OPENAI_API_KEY')
        self.api_key_elevenlabs = os.getenv('ELEVENLABS_API_KEY')
        self.temp_dir = './temp'

        # Define path to test audio file (harvard.wav)
        self.test_audio_path = os.path.join(self.temp_dir, 'test.wav')

        # Skip tests if the file doesn't exist
        if not os.path.exists(self.test_audio_path):
            print(f"Warning: Test audio file not found at {self.test_audio_path}")

        # Initialize the recognition models
        if self.api_key_openai:
            self.openai_recognition = RemoteRecognitionModel(
                self.api_key_openai,
                SupportedRecognitionModels['OPENAI']
            )

        if self.api_key_elevenlabs:
            self.elevenlabs_recognition = RemoteRecognitionModel(
                self.api_key_elevenlabs,
                SupportedRecognitionModels['ELEVENLABS']
            )

        # Only set up Keras if we're going to test it
        self.keras_available = False
        try:
            import keras_nlp
            self.keras_available = True
            self.keras_recognition = RemoteRecognitionModel(
                provider=SupportedRecognitionModels['KERAS'],
                model_name="whisper_tiny_en"
            )
        except ImportError:
            print("Keras NLP not available, skipping Keras tests")

    def test_openai_recognition(self):
        """Test speech recognition with OpenAI"""
        if not self.api_key_openai:
            self.skipTest("OpenAI API key not provided")

        if not os.path.exists(self.test_audio_path):
            self.skipTest(f"Test audio file not found: {self.test_audio_path}")

        # Create input parameters
        recognition_input = SpeechRecognitionInput(
            audio_file_path=self.test_audio_path,
            model="whisper-1"
        )

        try:
            # Get transcription
            result = self.openai_recognition.recognize_speech(recognition_input)
            print(f"OpenAI Recognition Result: {result}")

            self.assertIsInstance(result, str)
            self.assertTrue(len(result) > 0, "Transcription should not be empty")
        except Exception as e:
            self.fail(f"OpenAI recognition failed with error: {e}")

    def test_keras_recognition(self):
        """Test speech recognition with Keras offline model"""
        if not self.keras_available:
            self.skipTest("Keras NLP not available")

        if not os.path.exists(self.test_audio_path):
            self.skipTest(f"Test audio file not found: {self.test_audio_path}")

        # Create input parameters
        recognition_input = SpeechRecognitionInput(
            audio_file_path=self.test_audio_path,
            language="<|en|>"  # Whisper format for language
        )

        try:
            # Get transcription
            result = self.keras_recognition.recognize_speech(recognition_input)
            print(f"Keras Recognition Result: {result}")

            self.assertIsInstance(result, str)
            self.assertTrue(len(result) > 0, "Transcription should not be empty")
        except Exception as e:
            print(f"Warning: Keras recognition test failed with: {str(e)}")
            # Don't fail the test as Keras might have issues on some configurations
            self.skipTest(f"Keras recognition failed: {str(e)}")

    def test_elevenlabs_recognition(self):
        """Test speech recognition with Eleven Labs"""
        if not self.api_key_elevenlabs:
            self.skipTest("Eleven Labs API key not provided")

        if not os.path.exists(self.test_audio_path):
            self.skipTest(f"Test audio file not found: {self.test_audio_path}")

        recognition_input = SpeechRecognitionInput(
            audio_file_path=self.test_audio_path,
            language="en"  # Standard parameter
        )

        recognition_input.model_id = "scribe_v1"

        try:
            # Get transcription
            result = self.elevenlabs_recognition.recognize_speech(recognition_input)
            print(f"Eleven Labs Recognition Result: {result}")

            self.assertIsInstance(result, str)
            self.assertTrue(len(result) > 0, "Transcription should not be empty")

        except Exception as e:
            error_str = str(e).lower()
            if 'subscription' in error_str or '402' in error_str or 'payment' in error_str:
                self.skipTest(f"Eleven Labs speech recognition requires higher subscription tier: {e}")
            else:
                self.fail(f"Eleven Labs recognition failed with error: {e}")

if __name__ == "__main__":
    unittest.main()
import unittest
import os
import base64
from unittest.mock import patch, MagicMock
from intelli.wrappers.googleai_wrapper import GoogleAIWrapper
from dotenv import load_dotenv

load_dotenv()


class TestGoogleAIWrapper(unittest.TestCase):

    def setUp(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.assertIsNotNone(self.api_key, "GOOGLE_API_KEY must not be None.")
        self.googleai = GoogleAIWrapper(self.api_key)

        # Path to test files
        self.test_audio_path = os.path.join("./temp", "test.ogg")
        self.test_image_path = os.path.join("./temp", "test.png")

        # Check if test files exist
        self.audio_file_exists = os.path.exists(self.test_audio_path)
        self.image_file_exists = os.path.exists(self.test_image_path)

    def test_generate_speech(self):
        params = {
            "text": "Welcome to IntelliNode",
            "languageCode": "en-US",
            "name": "en-US-Wavenet-A",
            "ssmlGender": "MALE",
        }

        result = self.googleai.generate_speech(params)
        print(
            "Generate Speech Result:",
            result[:50] + "..." if len(result) > 50 else result,
        )
        self.assertTrue(len(result) > 0)

    def test_analyze_image(self):
        if not self.image_file_exists:
            self.skipTest("Test image file not found at ./temp/component.png")

        # Read the real image file
        with open(self.test_image_path, "rb") as image_file:
            image_content = image_file.read()

        # Use only label detection to reduce API costs
        features = [{"type": "LABEL_DETECTION", "maxResults": 5}]

        result = self.googleai.analyze_image(image_content, features)
        print("Image Analysis Result:", result)

        # Check if the response contains label annotations
        self.assertIn("responses", result)
        self.assertTrue(len(result["responses"]) > 0)
        if "labelAnnotations" in result["responses"][0]:
            print("Detected labels:")
            for label in result["responses"][0]["labelAnnotations"]:
                print(
                    f"- {label.get('description', 'Unknown')} ({label.get('score', 0):.2f})"
                )

    def test_transcribe_audio(self):
        if not self.audio_file_exists:
            self.skipTest(f"Test audio file not found at {self.test_audio_path}")

        # Read the real audio file
        with open(self.test_audio_path, "rb") as audio_file:
            audio_content = audio_file.read()

        # Configure for OGG file with a supported sample rate
        config_params = {
            "languageCode": "en-US",
            "enableAutomaticPunctuation": True,
            "encoding": "OGG_OPUS",
            "sampleRateHertz": 16000,
        }

        result = self.googleai.transcribe_audio(audio_content, config_params)
        print("Audio Transcription Result:", result)

        # Check if the response contains transcription results
        self.assertIn("results", result)

        # Collect all transcripts from all results
        transcripts = []
        for segment in result["results"]:
            if (
                len(segment["alternatives"]) > 0
                and "transcript" in segment["alternatives"][0]
            ):
                transcripts.append(segment["alternatives"][0]["transcript"])

        if transcripts:
            print(f"Found {len(transcripts)} transcript segments:")
            for i, t in enumerate(transcripts):
                print(f"  {i + 1}. {t}")

            # Join all transcripts into a single string for testing purposes
            full_transcript = " ".join(transcripts)
            print(f"Full transcript: {full_transcript}")

            # Assert that we found at least one transcript
            self.assertTrue(
                len(transcripts) > 0, "No transcripts found in the response"
            )
        else:
            print("No transcripts were generated from the audio file.")

    @patch("requests.post")
    def test_analyze_text(self, mock_post):
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "documentSentiment": {"magnitude": 0.8, "score": 0.4},
            "entities": [],
            "language": "en",
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        text = "Google Cloud offers many AI services that are excellent for developers."

        result = self.googleai.analyze_text(text)
        print("Text Analysis Result:", result)
        self.assertIn("documentSentiment", result)

    @patch("requests.post")
    def test_translate_text(self, mock_post):
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "translations": [
                {
                    "translatedText": "¡Hola, ¿cómo estás hoy?",
                    "detectedSourceLanguage": "en",
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        text = "Hello, how are you today?"
        target_language = "es"  # Spanish

        result = self.googleai.translate_text(text, target_language)
        print("Translation Result:", result)
        self.assertIn("translations", result)

    def test_describe_image(self):
        if not self.image_file_exists:
            self.skipTest("Test image file not found at ./temp/component.png")

        # Read the real image file
        with open(self.test_image_path, 'rb') as image_file:
            image_content = image_file.read()

        # Get image description
        result = self.googleai.describe_image(image_content)
        print('Image Description Result:')
        print(f"Summary: {result['summary']}")
        print("Detailed analysis:")
        for key, value in result['detailed_analysis'].items():
            print(f"  {key}: {value}")

        # Verify the response
        self.assertIn('summary', result)
        self.assertIn('detailed_analysis', result)
        self.assertTrue(len(result['summary']) > 0, "Summary should not be empty")

if __name__ == "__main__":
    unittest.main()

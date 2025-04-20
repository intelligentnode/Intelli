import base64
import os
import unittest
from dotenv import load_dotenv
from intelli.controller.remote_speech_model import (
    RemoteSpeechModel,
    SupportedSpeechModels,
)
from intelli.model.input.text_speech_input import Text2SpeechInput

load_dotenv()


class TestRemoteSpeechModel(unittest.TestCase):

    def setUp(self):
        self.api_key_google = os.getenv("GOOGLE_API_KEY")
        self.api_key_openai = os.getenv("OPENAI_API_KEY")
        self.api_key_elevenlabs = os.getenv("ELEVENLABS_API_KEY")
        self.remote_speech_model_google = RemoteSpeechModel(
            self.api_key_google, SupportedSpeechModels["GOOGLE"]
        )
        self.remote_speech_model_openai = RemoteSpeechModel(
            self.api_key_openai, SupportedSpeechModels["OPENAI"]
        )
        if self.api_key_elevenlabs:
            self.remote_speech_model_elevenlabs = RemoteSpeechModel(
                self.api_key_elevenlabs, SupportedSpeechModels["ELEVENLABS"]
            )
        self.temp_dir = "../temp"

        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

    def test_generate_speech_google(self):
        input_params = Text2SpeechInput("Welcome to Intellinode.", "en-gb")
        audio_content = self.remote_speech_model_google.generate_speech(input_params)
        self.assertTrue(audio_content, "audio_content should not be None")

        # eecode the base64
        audio_data = base64.b64decode(audio_content)
        google_file_path = os.path.join(self.temp_dir, "google_speech.mp3")

        # save
        with open(google_file_path, "wb") as audio_file:
            audio_file.write(audio_data)

        self.assertTrue(
            os.path.exists(google_file_path), "Google TTS MP3 file should be saved"
        )

    def test_generate_speech_openai(self):
        input_params = Text2SpeechInput(
            "Welcome to Intellinode.", "en-US", "MALE", "alloy", "tts-1", True
        )
        result = self.remote_speech_model_openai.generate_speech(input_params)
        self.assertTrue(result, "result should not be None")

        # write the streaming audio
        openai_file_path = os.path.join(self.temp_dir, "openai_speech.mp3")
        with open(openai_file_path, "wb") as audio_file:
            for chunk in result:
                audio_file.write(chunk)

        self.assertTrue(
            os.path.exists(openai_file_path), "OpenAI TTS MP3 file should be saved"
        )

    def test_generate_speech_elevenlabs(self):
        """Test text-to-speech functionality with Eleven Labs"""
        if not self.api_key_elevenlabs:
            self.skipTest("Eleven Labs API key not provided")

        try:
            voices_result = self.remote_speech_model_elevenlabs.list_voices()
            self.assertIn("voices", voices_result, "Should return a list of voices")
            self.assertTrue(
                len(voices_result["voices"]) > 0, "Should have at least one voice"
            )

            # Get the first voice ID
            voice_id = voices_result["voices"][0]["voice_id"]
            print(
                f"Using Eleven Labs voice: {voices_result['voices'][0]['name']} ({voice_id})"
            )

            input_params = Text2SpeechInput(
                "Welcome to Intellinode with Eleven Labs.", language="en"
            )
            # Add the voice_id attribute dynamically
            input_params.voice_id = voice_id
            input_params.model_id = (
                "eleven_multilingual_v2"
            )

            # Generate speech
            audio_content = self.remote_speech_model_elevenlabs.generate_speech(
                input_params
            )
            self.assertTrue(audio_content, "audio_content should not be None")

            # Save the audio file
            elevenlabs_file_path = os.path.join(self.temp_dir, "elevenlabs_speech.mp3")
            with open(elevenlabs_file_path, "wb") as audio_file:
                audio_file.write(audio_content)

            self.assertTrue(
                os.path.exists(elevenlabs_file_path),
                "Eleven Labs TTS MP3 file should be saved",
            )
            print(f"Eleven Labs audio saved to: {elevenlabs_file_path}")

            # Test streaming
            input_params.text = "This is a streaming test with Eleven Labs."
            streaming_response = self.remote_speech_model_elevenlabs.stream_speech(
                input_params
            )

            # Save the streaming audio
            elevenlabs_stream_path = os.path.join(
                self.temp_dir, "elevenlabs_stream.mp3"
            )
            with open(elevenlabs_stream_path, "wb") as audio_file:
                for chunk in streaming_response.iter_content(chunk_size=1024):
                    if chunk:
                        audio_file.write(chunk)

            self.assertTrue(
                os.path.exists(elevenlabs_stream_path),
                "Eleven Labs streaming MP3 file should be saved",
            )
            print(f"Eleven Labs streaming audio saved to: {elevenlabs_stream_path}")

        except Exception as e:
            self.fail(f"Eleven Labs test failed with error: {str(e)}")


if __name__ == "__main__":
    unittest.main()

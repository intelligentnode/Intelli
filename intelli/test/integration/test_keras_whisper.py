import os
import unittest
import numpy as np
from intelli.wrappers.keras_wrapper import KerasWrapper


class TestKerasWhisper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # the tiny english model is open, no kaggle credentials needed
        cls.wrapper = KerasWrapper(model_name="whisper_tiny_en")

    def test_whisper_real_audio(self):
        import soundfile as sf

        test_files = ["temp/long_audio.ogg", "temp/test.wav", "../temp/test.wav"]
        test_file = next((path for path in test_files if os.path.exists(path)), None)
        if not test_file:
            self.skipTest("The file not found.")
        audio_data, sample_rate = sf.read(test_file)
        # one minute is enough to cover multiple chunks
        audio_data = audio_data[: sample_rate * 60]

        result = self.wrapper.transcript(
            audio_data,
            sample_rate=sample_rate,
            language="<|en|>",
            user_prompt="You are a medical expert responsible for transcribing notes from a doctor’s speech.",
            condition_on_previous_text=True,
        )
        print("Transcription output:", result)
        self.assertGreater(len(result.split()), 3)

    def test_whisper_language_formats(self):
        # low noise instead of speech, the test checks the call only
        audio_data = np.random.default_rng(0).normal(0, 0.05, 16000).astype("float32")

        for language in [None, "en", "<|en|>"]:
            result = self.wrapper.transcript(audio_data, language=language, max_steps=8)
            self.assertIsInstance(result, str)

    def test_whisper_silent_audio(self):
        result = self.wrapper.transcript(np.zeros(16000, dtype="float32"))
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()

import os
import numpy as np
import pytest
import soundfile as sf
from intelli.wrappers.keras_wrapper import KerasWrapper

def test_whisper_real_audio():
    if not os.path.exists("/Users/ahmad/Downloads/harvard.wav"):
        pytest.skip("harvard.wav not found.")
    audio_data, sr = sf.read("/Users/ahmad/Downloads/harvard.wav")
    wrapper = KerasWrapper(model_name="whisper_tiny_en", 
                           model_params = {"KAGGLE_USERNAME": "jaguar00", 
                                           "KAGGLE_KEY": ""})
    assert wrapper is not None, "Failed to initialize the Whisper model."
    result = wrapper.transcript(audio_data)
    assert result is not None, "Transcription result is None."
    print("Transcription output:", result)

test_whisper_real_audio()
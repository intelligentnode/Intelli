import os
import numpy as np
import pytest
from intelli.wrappers.keras_wrapper import KerasWrapper

def test_whisper_real_audio():
    if not os.path.exists("/Users/ahmad/Downloads/harvard.wav"):
        pytest.skip("harvard.wav not found.")
    audio_data, sample_rate = sf.read("/Users/ahmad/Downloads/harvard.wav")

    wrapper = KerasWrapper(model_name="whisper_tiny_en", 
                           model_params = {"KAGGLE_USERNAME": "jaguar00", 
                                           "KAGGLE_KEY": ""})
    
    result = wrapper.transcript(audio_data, sample_rate=sample_rate)
    assert result is not None, "Transcription result is None."
    print("Transcription output:", result)

test_whisper_real_audio()

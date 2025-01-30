import os
import numpy as np
import pytest
from intelli.wrappers.keras_wrapper import KerasWrapper


def test_whisper_real_audio():
    import soundfile as sf

    test_file = "temp/long_audio.ogg"
    if not os.path.exists(test_file):
        pytest.skip("The file not found.")
    audio_data, sample_rate = sf.read(test_file)

    # TODO : add your kaggle username and key
    wrapper = KerasWrapper(
        model_name="whisper_tiny_en",
        model_params={
            "KAGGLE_USERNAME": "",
            "KAGGLE_KEY": "",
        },
    )

    result = wrapper.transcript(
                audio_data,
                sample_rate=sample_rate,
                language="<|en|>",
                user_prompt="You are a medical expert responsible for transcribing notes from a doctor’s speech.",
                condition_on_previous_text=True
            )
    assert result is not None, "Transcription result is None."
    print("Transcription output:", result)


test_whisper_real_audio()

---
sidebar_position: 5
---
# Recognition

The Recognition controller converts **audio into text** (speech recognition). It provides one unified interface across multiple providers.

### Supported Providers

Common options:
- **openai** (cloud transcription)
- **keras** (offline Whisper models)
- **elevenlabs**
- **speechmatics** (batch / real-time via wrapper capabilities)

### Parameters

- **key_value**: API key for providers that require it (e.g., `openai`, `elevenlabs`, `speechmatics`). Not required for `keras`.
- **provider** (optional): Provider name (defaults to `openai`).
- **model_name** (optional): Used by `keras` provider (default is `whisper_tiny_en`).
- **model_params** (optional): Used by `keras` to configure the local model.
- **input_params**: An instance of `SpeechRecognitionInput` describing the audio source and options.

### Example (OpenAI)

```python
from intelli.controller.remote_recognition_model import RemoteRecognitionModel
from intelli.model.input.text_recognition_input import SpeechRecognitionInput

model = RemoteRecognitionModel(
    key_value="your_openai_api_key",
    provider="openai",
)

input_params = SpeechRecognitionInput(
    audio_file_path="path/to/audio.mp3",
    language="en",
    model="whisper-1",
)

text = model.recognize_speech(input_params)
print(text)
```

### Example (Offline / Keras Whisper)

Install offline extras:

```bash
pip install "intelli[offline]"
```

```python
from intelli.controller.remote_recognition_model import RemoteRecognitionModel
from intelli.model.input.text_recognition_input import SpeechRecognitionInput

model = RemoteRecognitionModel(
    provider="keras",
    model_name="whisper_tiny_en",
)

input_params = SpeechRecognitionInput(
    audio_file_path="path/to/audio.mp3",
    language="<|en|>",
)

text = model.recognize_speech(input_params)
print(text)
```

---

## Speechmatics (extra dependencies)

To use Speechmatics, install the **speech extra**:

```bash
pip install "intelli[speech]"
```

This extra corresponds to the `speech` section in `setup.py` and includes packages like:
`speechmatics-batch`, `speechmatics-rt`, `websockets`, `librosa`, `soundfile`, `numpy`, and `openai>=2.5.0`.

Minimal usage:

```python
from intelli.controller.remote_recognition_model import RemoteRecognitionModel
from intelli.model.input.text_recognition_input import SpeechRecognitionInput

model = RemoteRecognitionModel(
    key_value="your_speechmatics_api_key",
    provider="speechmatics",
)

input_params = SpeechRecognitionInput(
    audio_file_path="path/to/audio.wav",
    language="en",
)

text = model.recognize_speech(input_params)
print(text)
```



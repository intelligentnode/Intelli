# Gemini API Latest Features - Implementation Guide

This document outlines all the latest Gemini API features that have been implemented in the updated `GeminiAIWrapper` and `GoogleAIWrapper` classes.

## 🚀 New Features Overview

### Gemini 2.0/2.5 Models Support
- **Gemini 2.0 Flash**: Latest text and vision model
- **Gemini 2.5 Flash**: Enhanced capabilities with TTS support
- **Veo 2.0**: Advanced video generation
- **Text-Embedding-004**: Latest embedding model

### Image Generation
- Native image generation using Gemini 2.0 Flash
- Text-to-image capabilities
- Configurable generation parameters

### Video Generation
- Veo 2.0 integration for video creation
- Aspect ratio control
- Person generation settings
- Long-running operation support

### Text-to-Speech (TTS)
- Native Gemini TTS capabilities
- Single-speaker and multi-speaker support
- Voice configuration options
- Natural language style control

### Enhanced Vision Capabilities
- Bounding box detection
- Image segmentation
- Multiple image processing
- File API integration for large files

### System Instructions
- Custom system prompts
- Behavior steering
- Context-aware responses

## 📋 Detailed Feature Documentation

### 1. Gemini 2.0 Flash Text Generation

```python
from intelli.wrappers.geminiai_wrapper import GeminiAIWrapper

wrapper = GeminiAIWrapper(api_key)

# Basic text generation with Gemini 2.0 Flash
params = {
    "contents": [{
        "parts": [{
            "text": "Explain quantum computing in simple terms."
        }]
    }]
}

result = wrapper.generate_content(params)
```

### 2. System Instructions

```python
# Generate content with system instructions
system_instruction = "You are a helpful teacher. Always provide examples."
content_parts = [{"text": "What is machine learning?"}]

result = wrapper.generate_content_with_system_instructions(
    content_parts, 
    system_instruction
)
```

### 3. Image Generation

```python
# Generate images using Gemini 2.0 Flash
prompt = "A serene Japanese garden with cherry blossoms"
config_params = {
    "responseModalities": ["TEXT", "IMAGE"]
}

result = wrapper.generate_image(prompt, config_params)

# Extract image data
for part in result['candidates'][0]['content']['parts']:
    if 'inline_data' in part:
        image_data = part['inline_data']['data']
        # Save or process the base64 encoded image
```

### 4. Video Generation with Veo 2.0

```python
# Generate videos (requires Google Cloud project)
prompt = "A peaceful mountain lake at sunrise"
config_params = {
    "aspectRatio": "16:9",
    "personGeneration": "dont_allow"
}

result = wrapper.generate_video(
    prompt, 
    config_params, 
    project_id="your-project-id"
)

# Check operation status
operation_name = result['name']
status = wrapper.check_video_generation_status(operation_name, project_id)

# Wait for completion
final_result = wrapper.wait_for_video_completion(
    operation_name, 
    project_id, 
    max_wait_time=300
)
```

### 5. Text-to-Speech

```python
# Single-speaker TTS
text = "Hello, this is a test of Gemini's TTS capabilities."
voice_config = {
    "prebuilt_voice_config": {
        "voice_name": "Kore"
    }
}

result = wrapper.generate_speech(text, voice_config)

# Multi-speaker TTS
text = """TTS the following conversation:
Alice: Hello Bob!
Bob: Hi Alice, how are you?"""

speaker_configs = [
    {
        "speaker": "Alice",
        "voice_config": {
            "prebuilt_voice_config": {"voice_name": "Kore"}
        }
    },
    {
        "speaker": "Bob",
        "voice_config": {
            "prebuilt_voice_config": {"voice_name": "Puck"}
        }
    }
]

result = wrapper.generate_multi_speaker_speech(text, speaker_configs)
```

### 6. File Upload and Management

```python
# Upload a file
upload_result = wrapper.upload_file("./image.png", "my_image")
file_uri = upload_result['file']['uri']
mime_type = upload_result['file']['mimeType']

# Use uploaded file for analysis
result = wrapper.image_to_text_with_file_uri(
    "Describe this image", 
    file_uri, 
    mime_type
)

# List uploaded files
files = wrapper.list_files()

# Delete a file
wrapper.delete_file(file_name)
```

### 7. Bounding Box Detection

```python
# Detect objects with bounding boxes
with open("image.png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

result = wrapper.get_bounding_boxes(
    "Detect all objects and provide bounding boxes",
    image_data,
    'png'
)

# Response includes normalized coordinates [ymin, xmin, ymax, xmax]
```

### 8. Image Segmentation

```python
# Get segmentation masks
result = wrapper.get_image_segmentation(
    "Segment all objects in this image",
    image_data,
    'png'
)

# Response includes JSON with box_2d, mask, and label for each object
```

### 9. Multiple Image Processing

```python
# Process multiple images
images_data = [
    {'mime_type': 'image/png', 'data': image1_data},
    {'mime_type': 'image/jpeg', 'data': image2_data}
]

result = wrapper.multiple_images_to_text(
    "Compare these images",
    images_data
)
```

### 10. Latest Embedding Model

```python
# Use text-embedding-004 (latest model)
text = "Sample text for embedding"
params = {
    "content": {
        "parts": [{"text": text}]
    }
}

result = wrapper.get_embeddings(params)
embedding_values = result['embedding']['values']  # 768 dimensions
```

## 🔧 Google AI Wrapper Enhancements

### Enhanced Vision API Features

```python
from intelli.wrappers.googleai_wrapper import GoogleAIWrapper

wrapper = GoogleAIWrapper(api_key)

# Object detection with localization
result = wrapper.detect_objects_with_localization(image_content)

# Text detection including handwriting
result = wrapper.detect_text_with_handwriting(image_content)

# Face detection with emotions
result = wrapper.detect_faces_with_emotions(image_content)

# Safe search detection
result = wrapper.detect_safe_search(image_content)

# Crop hints
result = wrapper.crop_hints(image_content, [1.0, 1.5, 0.75])

# Document text extraction
result = wrapper.extract_document_text(image_content)
```

### Enhanced Speech Features

```python
# SSML-based speech generation
ssml_text = """
<speak>
    <p>Hello, <emphasis level="strong">this is important</emphasis>!</p>
    <break time="1s"/>
    <p>This is SSML speech.</p>
</speak>
"""

voice_params = {
    "languageCode": "en-US",
    "name": "en-US-Wavenet-A",
    "ssmlGender": "FEMALE"
}

result = wrapper.generate_speech_with_ssml(ssml_text, voice_params)

# Long-running audio transcription
result = wrapper.transcribe_audio_long_running("gs://bucket/audio.wav")
```

### Enhanced Language Features

```python
# Entity sentiment analysis
result = wrapper.analyze_entity_sentiment(text)

# Text classification
result = wrapper.classify_text(text)

# Language detection
result = wrapper.detect_language(text)

# Get supported languages
result = wrapper.get_supported_languages()
```

## 🧪 Testing

### Run Basic Tests
```bash
# Test Gemini wrapper
python -m pytest test/integration/test_geminiai_wrapper.py -v

# Test Google AI wrapper
python -m pytest test/integration/test_googleai_wrapper.py -v
```

### Run Comprehensive Feature Test
```bash
# Run the comprehensive test suite
python test/integration/test_gemini_latest_features.py
```

### Environment Setup
```bash
# Required environment variables
export GEMINI_API_KEY="your_gemini_api_key"
export GOOGLE_API_KEY="your_google_api_key"
export GOOGLE_PROJECT_ID="your_project_id"  # For video generation
```

## 📝 Configuration Updates

The configuration has been updated to support the latest models:

```python
"gemini": {
    "base": "https://generativelanguage.googleapis.com/v1beta/models",
    "upload_base": "https://generativelanguage.googleapis.com/upload/v1beta/files",
    "files_base": "https://generativelanguage.googleapis.com/v1beta/files",
    "vertex_base": "https://us-central1-aiplatform.googleapis.com/v1/projects",
    "models": {
        "text": "gemini-2.0-flash",
        "vision": "gemini-2.0-flash",
        "embedding": "text-embedding-004",
        "image_generation": "gemini-2.0-flash-preview-image-generation",
        "video_generation": "veo-2.0-generate-001",
        "tts": "gemini-2.5-flash-preview-tts",
        "tts_pro": "gemini-2.5-pro-preview-tts",
        "legacy_text": "gemini-1.5-pro",
        "legacy_vision": "gemini-1.5-pro"
    }
}
```

## 🚨 Important Notes

### Model Availability
- Some features (image generation, video generation, TTS) may require special access or billing setup
- Video generation requires a Google Cloud project with Vertex AI enabled
- TTS models are in preview and may have limited availability

### Rate Limits
- Be aware of API rate limits for different features
- Video generation has longer processing times
- File uploads have size limitations (20MB for inline, larger for File API)

### Error Handling
- All methods include comprehensive error handling
- Check response structure before accessing nested data
- Some features may return different response formats

### Backward Compatibility
- Legacy models (Gemini 1.5 Pro) are still supported
- Use `model_override` parameter to specify different models
- Existing code should continue to work with minimal changes

## 🔗 Related Documentation

- [Gemini API Documentation](https://ai.google.dev/docs)
- [Google Cloud Vision API](https://cloud.google.com/vision/docs)
- [Google Cloud Speech API](https://cloud.google.com/speech-to-text/docs)
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)

## 🤝 Contributing

When adding new features:
1. Update the wrapper classes with new methods
2. Add comprehensive tests
3. Update this documentation
4. Ensure backward compatibility
5. Add proper error handling

## 📄 License

This implementation follows the same license as the main project. 
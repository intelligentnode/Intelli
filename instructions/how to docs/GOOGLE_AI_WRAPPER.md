# GoogleAIWrapper Guide (Gemini + Vertex AI)

Read this before writing code that calls Gemini, Imagen, Veo, Lyria, Gemini TTS, the Live API or Agent Engine through Intelli.

## 1. Purpose and when to use it

`GoogleAIWrapper` (`intelli/wrappers/googleai_wrapper.py`) is the one entry point for Google AI in Intelli.

| You want | Use |
| --- | --- |
| Gemini text, chat, streaming, tools, media understanding, image/video/music/speech generation, embeddings, Live API, Agent Engine | `GoogleAIWrapper` (the Gemini and Vertex AI methods) |
| Google Cloud Text-to-Speech, Speech-to-Text, Vision, Natural Language, Translation (Cloud API key) | `GoogleAIWrapper` (the older Cloud methods at the top of the class; unchanged) |
| Old code that imports `GeminiAIWrapper` | Still works, but it is **deprecated** (emits `DeprecationWarning`) and only forwards to `GoogleAIWrapper`. Migrate (section 9). |
| Provider-agnostic chat / Flow agents | `Chatbot(..., "gemini", options)` and `Agent(..., provider="gemini", options=...)` build a `GoogleAIWrapper` for you (section 7) |

Importable names:

```python
from intelli.wrappers.googleai_wrapper import (
    GoogleAIWrapper, GoogleAIError, GoogleAIChatSession, GoogleAILiveSession)
```

## 2. 30-second quick start

```python
import os
from intelli.wrappers.googleai_wrapper import GoogleAIWrapper

w = GoogleAIWrapper(os.environ["VERTEX_API_KEY"], vertex=True)      # Vertex express mode
print(w.generate_text("Explain RAG in one sentence.", model="gemini-3.8-flash"))

chat = w.start_chat("gemini-3.8-flash", system_instruction="Be brief.")
print(chat.send_text("My name is Sam."))
for chunk in chat.stream("What is my name?"):
    print(chunk, end="", flush=True)
```

With an AI Studio key use `GoogleAIWrapper(os.environ["GEMINI_API_KEY"])` instead (Gemini Developer API).

## 3. Auth modes

Constructor: `GoogleAIWrapper(api_key=None, timeout=180, *, vertex=None, project_id=None, location=None, credentials=None, access_token=None, api_version=None, base_url=None, quota_project_id=None, session=None)`

| Mode | Constructor | Endpoint | Auth header |
| --- | --- | --- | --- |
| Gemini Developer API (AI Studio key, usually `AIza...`) | `GoogleAIWrapper(GEMINI_API_KEY)` | `https://generativelanguage.googleapis.com/v1beta/models/{m}:{method}` | `x-goog-api-key` |
| Vertex express (Agent Platform key, `AQ.` format) | `GoogleAIWrapper(VERTEX_API_KEY, vertex=True)` | `https://aiplatform.googleapis.com/v1beta1/publishers/google/models/{m}:{method}` | `x-goog-api-key` |
| Vertex project + key | `GoogleAIWrapper(VERTEX_API_KEY, vertex=True, project_id=PROJECT)` | `https://{host}/v1beta1/projects/{p}/locations/{loc}/publishers/google/models/{m}:{method}` | `x-goog-api-key` |
| Vertex ADC (no key) | `GoogleAIWrapper(project_id=PROJECT)` after `gcloud auth application-default login` | project-scoped, as above | `Authorization: Bearer` |
| Vertex OAuth token | `GoogleAIWrapper(access_token=TOKEN, project_id=PROJECT)` (`TOKEN` may be a string or a zero-arg callable) | project-scoped | `Authorization: Bearer` |
| Vertex credentials object | `GoogleAIWrapper(credentials=creds, project_id=PROJECT)` (a `google.auth` credentials object; refreshed when needed) | project-scoped | `Authorization: Bearer` |

Rules the code follows:

- `vertex=None` (default) means Vertex is selected when `project_id`, `credentials` or `access_token` is given, or when `GOOGLE_GENAI_USE_VERTEXAI` / `GOOGLE_GENAI_USE_ENTERPRISE` is `1`/`true`/`yes`. So `GoogleAIWrapper(VERTEX_API_KEY, project_id=PROJECT)` is also Vertex.
- If `api_key` is set it is always sent (`x-goog-api-key`); `access_token` / `credentials` are used only when there is no key.
- Host by location: `global` (or none) -> `aiplatform.googleapis.com`; `us` / `eu` -> `aiplatform.{loc}.rep.googleapis.com`; any other region -> `{loc}-aiplatform.googleapis.com`.
- Project mode without `location` uses `global`, and sends Veo, Live, Lyria, Imagen and Agent Engine to `us-central1` (`config['url']['gemini']['vertex']['locations']`). An explicit `location` (or `GOOGLE_CLOUD_LOCATION`) is used for every capability; see gotcha 2.
- Without a key, the project comes from `project_id`, else `GOOGLE_CLOUD_PROJECT`, else the ADC default project. With a key and no `project_id` you are in express mode (the env project is ignored).
- ADC needs `pip install google-auth`. The Live API needs `pip install websockets`.
- `quota_project_id` adds `x-goog-user-project` (Bearer modes only). `api_version` overrides `v1beta1` (Vertex) or `v1beta` (Developer API) for model calls (Files API URLs stay on `v1beta`). `base_url` replaces the API root for model calls (proxies, tests). `session` takes your own `requests.Session`.
- Some organizations block API keys. Then use ADC: `GoogleAIWrapper(project_id=PROJECT)`.

`GoogleAIWrapper.from_options(api_key=None, options=None, timeout=None)` builds a wrapper from a Chatbot / Agent / controller options dict. Keys read: `vertex`, `project_id` (or `vertex_project`), `location` (or `vertex_location`), `credentials`, `access_token`, `api_version`, `quota_project_id`, `timeout`.

## 4. Capability matrix

`yes` = supported, `no` = raises, `OAuth` = Vertex rejects API keys there (use ADC / `access_token`), `project` = needs `project_id`.

| Feature | Methods | Developer API | Vertex express | Vertex project |
| --- | --- | --- | --- | --- |
| Text, system prompt, thinking | `generate_text`, `generate_content` | yes | yes | yes |
| Multi-turn chat | `start_chat` -> `GoogleAIChatSession` | yes | yes | yes |
| Streaming | `stream_text`, `stream_generate_content`, `chat.stream` | yes | yes | yes |
| Structured JSON | `generate_structured_content`, or `generation_config` | yes | yes | yes |
| Function calling | `tools=[{"functionDeclarations": [...]}]`, `extract_function_calls`, `chat.send_function_response` | yes | yes | yes |
| Google Search grounding | `tools=[{"googleSearch": {}}]`, `extract_grounding` | yes | yes | yes |
| Code execution / URL context | `tools=[{"codeExecution": {}}]` / `[{"urlContext": {}}]` | yes | yes | yes |
| Image / audio / video / PDF understanding | `media_part`, `media_to_text`, `audio_to_text`, `video_to_text`, `image_to_text` | yes (inline, Files API URIs, YouTube) | yes (inline, `gs://`) | yes (inline, `gs://`) |
| Gemini image generation + editing | `generate_image`, `edit_image`, `extract_images` | yes | yes | yes |
| Imagen generate | `imagen_generate_images` | yes | yes (1) | yes (us-central1) |
| Imagen edit / upscale | `imagen_edit_image`, `imagen_upscale_image` | no (Vertex only) | yes (1) | yes (us-central1) |
| Veo video | `generate_video`, `wait_for_video_completion`, `extract_videos`, `download_media` | yes | no (`ValueError`) | yes (us-central1) |
| Lyria music | `generate_music`, `extract_audio`, `audio_to_wav` | Lyria 3 (`lyria-3.5`) | `lyria-002` (1) | `lyria-002` (us-central1) |
| Gemini TTS, multi-speaker | `generate_gemini_speech`, `generate_multi_speaker_speech` | yes | yes | yes |
| Embeddings | `embed_texts`, `get_embeddings`, `get_batch_embeddings` | yes | yes (`gemini-embedding-2`: OAuth) | yes (`gemini-embedding-2`: OAuth) |
| Token counting | `count_tokens` / `compute_tokens` | yes / no | yes / yes | yes / yes |
| Files API | `upload_file`, `get_file`, `list_files`, `delete_file` | yes | no (`NotImplementedError`) | no (`NotImplementedError`) |
| Context caching | `create_cached_content`, `get_cached_content`, `list_cached_contents`, `delete_cached_content` | yes | no (project) | OAuth |
| Model listing | `list_models` / `model_catalog()` (offline) | yes / yes | OAuth / yes | OAuth / yes |
| Live API (websocket) | `live_connect`, `live_generate`, `live_generate_async` | yes | no (project) | yes (us-central1 only) |
| Agent Engine | `list_agent_engines`, `get_agent_engine`, `query_agent_engine`, `stream_query_agent_engine` | no | no (project) | yes (us-central1) |

(1) Express mode sends these to the `us-central1` regional host without a project. If your key is rejected there, use project mode.

Tool availability (search, code execution, URL context) also depends on the model.

## 5. One example per capability

All examples assume:

```python
import base64, json, os, time
from intelli.wrappers.googleai_wrapper import GoogleAIWrapper, GoogleAIError

KEY, PROJECT = os.environ["VERTEX_API_KEY"], os.environ.get("VERTEX_PROJECT_ID")
w = GoogleAIWrapper(KEY, vertex=True)                       # express
wp = GoogleAIWrapper(KEY, vertex=True, project_id=PROJECT)  # project (Veo, Live, Agent Engine)
M = "gemini-3.8-flash"
```

Responses from `generate_content`, `generate_image`, TTS and Gemini music are JSON dicts with snake_case aliases added (`inlineData` and `inline_data` both present). Use the `extract_*` static helpers to read them.

### Text and system prompt

```python
text = w.generate_text("Write a haiku about rivers.", model=M,
                       system_instruction="You are a poet. Reply with the poem only.",
                       generation_config={"maxOutputTokens": 1024,
                                          "thinkingConfig": {"thinkingLevel": "low"}})  # Gemini 3
# Gemini 2.5: {"thinkingConfig": {"thinkingBudget": 0}}
raw = w.generate_content("Hello", model=M)   # full response; params may be a string or a request body
print(w.extract_text(raw), raw.get("usageMetadata"))
```

`generate_text(prompt, model=None, *, system_instruction=None, media=None, generation_config=None, tools=None, tool_config=None, safety_settings=None, history=None)` returns a string. `generate_content(params, vision=False, model_override=None, *, model=None)` returns the response dict.

### Chat with history

```python
chat = w.start_chat(M, system_instruction="You are a helpful tutor.")
chat.send_text("I am learning Spanish.")
reply = chat.send_text("Give me one practice sentence.")
saved = chat.history                                  # JSON-serializable list of contents
chat2 = w.start_chat(M, history=saved)                # resume later
```

`GoogleAIChatSession`: `send(message=None, media=None, *, parts=None)` -> response, `send_text(message=None, media=None)` -> str, `send_function_response(name, response, call_id=None)`, `stream(message=None, media=None)` -> text chunks, `reset()`, attributes `history`, `last_response` (set by `send` only), `model`. History keeps thought signatures, which Gemini 3 needs for multi-turn tool use and image editing.

### Streaming

```python
for chunk in w.stream_text("Tell a 5 line story.", model=M):
    print(chunk, end="", flush=True)

for event in w.stream_generate_content({"contents": "Hi"}, model=M):   # parsed dict per SSE event
    print(w.extract_text(event), end="")
```

`stream_generate_content(params, vision=False, model_override=None, *, model=None, raw=False)`. `raw=True` gives the old behavior (raw decoded lines, no `alt=sse`).

### Structured output

```python
schema = {"type": "OBJECT",
          "properties": {"city": {"type": "STRING"}, "population": {"type": "INTEGER"}},
          "required": ["city", "population"]}
r = w.generate_structured_content([{"text": "Largest city in Japan?"}], schema, model_override=M)
data = json.loads(w.extract_text(r))
```

Signature: `generate_structured_content(content_parts, response_schema, system_instruction=None, model_override=None, response_mime_type="application/json", generation_config=None, tools=None, tool_config=None)`. Same result: `generate_text(..., generation_config={"responseMimeType": "application/json", "responseSchema": schema})`.

### Function calling loop

```python
tools = [{"functionDeclarations": [{
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "OBJECT", "properties": {"city": {"type": "STRING"}}, "required": ["city"]}}]}]
handlers = {"get_weather": lambda city: {"temp_c": 21, "sky": "clear"}}

def run(call):                                          # call: {'name', 'args', 'id'?}
    result = {"name": call["name"], "response": handlers[call["name"]](**call.get("args", {}))}
    if call.get("id"):
        result["id"] = call["id"]
    return {"functionResponse": result}

chat = w.start_chat(M, tools=tools)
r = chat.send("What's the weather in Paris?")
while (calls := w.extract_function_calls(r)):
    r = chat.send(parts=[run(c) for c in calls])        # all results in one turn
print(w.extract_text(r))
```

For a single call, `chat.send_function_response(c["name"], result_dict, c.get("id"))` is enough. Use `tool_config={"functionCallingConfig": {"mode": "ANY"}}` to force a call.

### Google Search grounding

```python
r = w.generate_content({"contents": "Who won the most recent Ballon d'Or?",
                        "tools": [{"googleSearch": {}}]}, model=M)
print(w.extract_text(r))
meta = w.extract_grounding(r)          # groundingMetadata: webSearchQueries, groundingChunks, ...
sources = [c["web"]["uri"] for c in meta.get("groundingChunks", []) if "web" in c]
```

### Code execution

```python
r = w.generate_content({"contents": "Compute the sum of the first 50 primes with Python.",
                        "tools": [{"codeExecution": {}}]}, model=M)
for part in r["candidates"][0]["content"]["parts"]:
    if "executableCode" in part: print(part["executableCode"]["code"])
    if "codeExecutionResult" in part: print(part["codeExecutionResult"].get("output"))
print(w.extract_text(r))
```

### URL context

```python
r = w.generate_content({"contents": "Summarize https://example.com/article in 3 bullets.",
                        "tools": [{"urlContext": {}}]}, model=M)
print(w.extract_text(r), r["candidates"][0].get("urlContextMetadata"))
```

### Image, audio, video and PDF understanding

`media_part(source=None, mime_type=None, *, data=None, path=None, uri=None, video_metadata=None)` builds one part. `source` may be a part dict, bytes (needs `mime_type`), a `(bytes, mime_type)` tuple, a local path, or a `gs://` / `https://` / YouTube URI. In `media=[...]` lists, raw bytes must be `(bytes, mime)` tuples.

```python
w.media_to_text("Describe this photo.", ["photo.jpg"], model=M)
w.media_to_text("Compare these.", [(png_bytes, "image/png"), "gs://bucket/b.png"], model=M)
w.audio_to_text("meeting.mp3")                                   # default prompt: "Transcribe this audio."
w.video_to_text("gs://bucket/clip.mp4", "List the scenes.",
                video_metadata={"startOffset": "0s", "endOffset": "30s"})
w.video_to_text("https://www.youtube.com/watch?v=VIDEO_ID")      # default prompt: "Summarize this video."
w.media_to_text("Summarize this report.", ["report.pdf"], model=M)
w.generate_text("What is in the image?", model=M, media=[w.media_part(path="cat.png")])
```

Signatures: `media_to_text(prompt, media, model=None, **kwargs)` (kwargs go to `generate_text`), `audio_to_text(audio, prompt="Transcribe this audio.", mime_type=None, model=None, **kwargs)`, `video_to_text(video, prompt="Summarize this video.", mime_type=None, model=None, video_metadata=None, **kwargs)`. Older helpers keep working: `image_to_text(user_input, image_data_base64, extension, model_override=None)`, `image_to_text_params`, `image_to_text_with_file_uri`, `multiple_images_to_text`, `get_bounding_boxes`, `get_image_segmentation` (they return the raw response).

### Image generation and editing (Gemini image models)

```python
r = w.generate_image("A watercolor fox in a snowy forest", {"imageConfig": {"aspectRatio": "16:9"}},
                     model_override="gemini-3.1-flash-image")
img = w.extract_images(r)[0]                       # {'mime_type', 'data' (base64)}
open("fox.png", "wb").write(base64.b64decode(img["data"]))

r2 = w.edit_image("Make it night time, keep the fox.", ["fox.png"], model_override="gemini-3.1-flash-image")
```

`generate_image(prompt, config_params=None, model_override=None, *, images=None)` (config_params is merged into `generationConfig`; default `responseModalities` is `["TEXT", "IMAGE"]`), `edit_image(prompt, images, config_params=None, model_override=None)`. For iterative editing keep a chat: `w.start_chat("gemini-3.1-flash-image", generation_config={"responseModalities": ["TEXT", "IMAGE"]})`, then `w.extract_images(chat.send("Now add a moon"))`.

### Imagen (deprecated by Google; prefer Gemini image models)

```python
r = wp.imagen_generate_images("A red bicycle on a beach", number_of_images=2, aspect_ratio="1:1")
images = wp.extract_images(r)                      # works on Imagen predictions too
r = wp.imagen_edit_image("Replace the background with a city street", "bike.png",
                         mask_mode="MASK_MODE_BACKGROUND")
r = wp.imagen_upscale_image("bike.png", "x2")
```

Signatures: `imagen_generate_images(prompt, number_of_images=1, model=None, *, aspect_ratio=None, negative_prompt=None, parameters=None)`, `imagen_edit_image(prompt, image, mask=None, *, edit_mode=None, mask_mode=None, model=None, parameters=None)`, `imagen_upscale_image(image, upscale_factor='x2', model=None, *, parameters=None)`. A retired Imagen model fails with a `GoogleAIError` starting with `Imagen error` (Google's message says it is deprecated).

### Veo video (project mode)

```python
op = wp.generate_video("A slow drone shot over a misty pine forest",
                       {"durationSeconds": 4, "resolution": "720p", "generateAudio": False})
done = wp.wait_for_video_completion(op["name"], max_wait_time=900, poll_interval=10)
for i, v in enumerate(wp.extract_videos(done)):   # [{'mime_type', 'data' (base64) or None, 'uri' or None}]
    if v["data"]:
        open(f"video_{i}.mp4", "wb").write(base64.b64decode(v["data"]))
    elif v["uri"] and v["uri"].startswith("https://"):
        open(f"video_{i}.mp4", "wb").write(wp.download_media(v["uri"]))   # Developer API result
    else:
        print("stored at", v["uri"])               # gs:// when config_params has "storageUri"
```

`generate_video(prompt, config_params=None, project_id=None, *, model=None, image=None, last_frame=None, location=None)` returns the long-running operation. `image` / `last_frame` enable image-to-video (see gotcha 13). Poll manually with `get_video_operation(operation_name_or_dict)` (`check_video_generation_status` is an alias). `wait_for_video_completion(operation_name, project_id=None, max_wait_time=300, poll_interval=5)` raises `TimeoutError`. Default parameters: `{"aspectRatio": "16:9"}` plus your `config_params`. Veo is billed per second of video.

### Lyria music

```python
r = wp.generate_music("Calm lo-fi piano with soft rain", negative_prompt="drums", seed=42)  # lyria-002 on Vertex
track = wp.extract_audio(r)[0]
open("music.wav", "wb").write(wp.audio_to_wav(track))
```

`generate_music(prompt, model=None, *, negative_prompt=None, seed=None, sample_count=None, generation_config=None, location=None)`. `lyria-0xx` models use `:predict` (Vertex only; `GoogleAIError` on the Developer API). Other models (e.g. `lyria-3.5`, the Developer API default) use `generateContent` with `responseModalities ["AUDIO", "TEXT"]`.

### Speech (Gemini TTS) and multi-speaker

```python
r = w.generate_gemini_speech("Say cheerfully: have a wonderful day!", voice="Puck")
audio = w.extract_audio(r)[0]                      # 2.5 TTS: 'audio/L16;codec=pcm;rate=24000'
open("hello.wav", "wb").write(w.audio_to_wav(audio))   # wraps PCM; returns WAV unchanged

speakers = [{"speaker": "Joe", "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}},
            {"speaker": "Jane", "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Puck"}}}]
r = w.generate_multi_speaker_speech(
    "TTS the following conversation between Joe and Jane:\nJoe: Hi Jane!\nJane: Hi Joe, how are you?",
    speakers, model_override="gemini-2.5-flash-tts")
open("dialog.wav", "wb").write(w.audio_to_wav(w.extract_audio(r)[0]))
```

`generate_gemini_speech(text, voice_config=None, model_override=None, *, voice=None)` (default voice `Kore`), `generate_multi_speaker_speech(text, speaker_configs, model_override=None)`. `GoogleAIWrapper.generate_speech(params)` is **Google Cloud Text-to-Speech**, not Gemini. `pcm_to_wav(pcm, sample_rate=24000, channels=1, sample_width=2)` wraps raw PCM bytes or base64.

### Embeddings

```python
vectors = w.embed_texts(["first doc", "second doc"], task_type="RETRIEVAL_DOCUMENT", output_dimensionality=768)
one = w.get_embeddings({"content": {"parts": [{"text": "hello"}]}})["embedding"]["values"]
batch = w.get_batch_embeddings({"requests": [{"content": {"parts": [{"text": "a"}]}}]})   # [{'values': [...]}]
```

`embed_texts(texts, model=None, *, task_type=None, title=None, output_dimensionality=None)` -> list of vectors. `get_embeddings` honors `params["model"]`; `get_batch_embeddings` always uses the default embedding model. Vertex default is `gemini-embedding-001`; Vertex takes one text per request for `gemini-embedding-*`, and `embed_texts` loops for you.

### Tokens

```python
w.count_tokens("How many tokens is this?", model=M)                              # {'totalTokens': ...}
w.count_tokens({"contents": "Hi", "system_instruction": "Be brief."}, model=M)
w.compute_tokens("hello world", model=M)                                          # Vertex only: {'tokensInfo': [...]}
```

### Files API (Developer API only)

```python
d = GoogleAIWrapper(os.environ["GEMINI_API_KEY"])
f = d.upload_file("lecture.mp4")["file"]                  # {'name', 'uri', 'mimeType', 'state', ...}
while d.get_file(f["name"]).get("state") == "PROCESSING":
    time.sleep(5)
print(d.media_to_text("Summarize the lecture.", [d.media_part(f["uri"], f["mimeType"])], model=M))
d.delete_file(f["name"])
```

On Vertex these raise `NotImplementedError`; send files inline (path / bytes) or as `gs://` URIs.

### Context caching

```python
c = d.create_cached_content(M, [{"role": "user", "parts": [d.media_part("big_report.pdf")]}],
                            system_instruction="Answer questions about the report.", ttl="600s")
r = d.generate_content({"cachedContent": c["name"], "contents": "List the key risks."}, model=M)
d.delete_cached_content(c["name"])
```

`create_cached_content(model, contents, *, system_instruction=None, ttl='3600s', display_name=None, tools=None, tool_config=None, location=None)`, `get_cached_content(name)`, `list_cached_contents(page_size=None, page_token=None, location=None)`, `delete_cached_content(name)`. Vertex needs `project_id` and OAuth (ADC / `access_token`). Google enforces a minimum cached size, and the generate call must use the same model as the cache.

### Live API (project mode on Vertex, `pip install websockets`)

```python
turn = wp.live_generate("Tell me a short joke.")          # blocking; uses asyncio.run
print(turn["transcription"])                              # text of the spoken reply
open("joke.wav", "wb").write(wp.pcm_to_wav(turn["audio"]))   # 24 kHz PCM

import asyncio
async def main():
    async with wp.live_connect(config={"systemInstruction": {"parts": [{"text": "You are a voice assistant."}]},
                                      "outputAudioTranscription": {}}) as session:
        await session.send_text("Hello!")
        t = await session.receive_turn()
        if t["tool_calls"]:
            await session.send_tool_response([{"id": c["id"], "name": c["name"], "response": {"ok": True}}
                                              for c in t["tool_calls"]])
            t = await session.receive_turn()
        print(t["transcription"])
asyncio.run(main())
```

`live_connect(model=None, config=None, location=None)` is an async context manager yielding `GoogleAILiveSession`: `send(message)`, `send_text(text, turn_complete=True)`, `send_audio(data, mime_type='audio/pcm;rate=16000')`, `send_audio_stream_end()`, `send_tool_response(function_responses)`, `receive()` (async iterator), `receive_turn()` -> `{'text', 'transcription', 'input_transcription', 'audio' (bytes), 'audio_mime_type', 'tool_calls', 'usage', 'messages'}`, `close()`. `config` is the setup message without `model`; `responseModalities` defaults to `["AUDIO"]`. Inside a running event loop use `await wp.live_generate_async(text)` (same arguments as `live_generate(text, model=None, config=None, location=None)`).

### Agent Engine (project mode)

```python
engines = wp.list_agent_engines().get("reasoningEngines", [])
name = engines[0]["name"]                                  # full resource name or just the id
print(wp.query_agent_engine(name, {"input": "Hello"}))     # {'output': ...}
for event in wp.stream_query_agent_engine(name, {"message": "Hello", "user_id": "u1"}):
    print(event)                                           # dict, or text when not JSON
```

Signatures: `list_agent_engines(location=None, page_size=None, page_token=None, filter=None)`, `get_agent_engine(name, location=None)`, `query_agent_engine(name, input=None, class_method=None, location=None)`, `stream_query_agent_engine(name, input=None, class_method='stream_query', location=None)`. Input keys depend on the deployed agent (an ADK app takes `message` and `user_id`).

## 6. Response helpers (static)

| Helper | Returns |
| --- | --- |
| `extract_text(response, include_thoughts=False)` | joined text of the first candidate (thought parts skipped) |
| `extract_function_calls(response)` | `[{'name', 'args', 'id'?}]` |
| `extract_grounding(response)` | `groundingMetadata` dict or `{}` |
| `extract_images(response)` | `[{'mime_type', 'data'}]` from Gemini parts or Imagen predictions |
| `extract_audio(response)` | `[{'mime_type', 'data'}]` from Gemini TTS / music parts or Lyria predictions |
| `extract_videos(operation)` | `[{'mime_type', 'data', 'uri'}]` from a finished Veo operation (both backends) |
| `audio_to_wav(item)` / `pcm_to_wav(pcm, sample_rate=24000, channels=1, sample_width=2)` | WAV bytes |
| `model_catalog()` | known Vertex model ids by capability (copy of config) |

## 7. Chatbot and Flow Agent usage

`Chatbot` and the gemini controllers (`RemoteImageModel`, `RemoteVisionModel`, `RemoteSpeechModel`, `RemoteEmbedModel`) call `GoogleAIWrapper.from_options(api_key, options)`. No options = Gemini Developer API.

```python
from intelli.function.chatbot import Chatbot
from intelli.model.input.chatbot_input import ChatModelInput

bot = Chatbot(KEY, "gemini", {"vertex": True, "project_id": PROJECT, "location": "global"})
# express: {"vertex": True}   ADC: Chatbot(None, "gemini", {"project_id": PROJECT})
inp = ChatModelInput("You are a helpful assistant.", model="gemini-3.8-flash", max_tokens=2048)
inp.add_user_message("What is Vertex AI express mode?")
print(bot.chat(inp)[0])                 # list of strings, one per candidate
for chunk in bot.stream(inp):           # text chunks (Chatbot._stream_gemini)
    print(chunk, end="")
```

Extra `ChatModelInput` kwargs are merged into the request body, so `ChatModelInput(..., generationConfig={"maxOutputTokens": 2048, "thinkingConfig": {"thinkingLevel": "low"}})` replaces the generated `generationConfig`, and `tools=[{"googleSearch": {}}]` adds tools.

Flow agents pass `options` through to the Chatbot and to the image / vision / speech / embed controllers:

```python
import asyncio
from intelli.flow import Agent, Task, Flow, TextTaskInput, AgentTypes
from intelli.wrappers.googleai_wrapper import GoogleAIWrapper

opts = {"vertex": True, "project_id": PROJECT}      # or {"vertex": True} for express
writer = Agent(AgentTypes.TEXT.value, "gemini", "Write a one-line slogan",
               {"key": KEY, "model": "gemini-3.8-flash"}, options=opts)
painter = Agent(AgentTypes.IMAGE.value, "gemini", "Create a poster",
                {"key": KEY, "model": "gemini-3.1-flash-image"}, options=opts)
voice = Agent(AgentTypes.SPEECH.value, "gemini", "Read the slogan",
              {"key": KEY, "model": "gemini-3.8-flash-tts"}, options=opts)
tasks = {"slogan": Task(TextTaskInput("Reusable water bottle"), writer),
         "poster": Task(TextTaskInput("Poster for the slogan"), painter),
         "voice": Task(TextTaskInput("Read it aloud"), voice)}
out = asyncio.run(Flow(tasks=tasks, map_paths={"slogan": ["poster", "voice"]}).start())
png = base64.b64decode(out["poster"]["output"])            # image agents return base64
wav = base64.b64decode(out["voice"]["output"])             # 3.8 TTS returns WAV
# with a 2.5 TTS model the output is raw PCM: wav = GoogleAIWrapper.pcm_to_wav(out["voice"]["output"])
```

Flow notes: text and speech agents require `model_params["key"]`; vision agents require `model_params["model"]` (and `extension`, default `png`); the speech controller maps OpenAI voices / `tts-1` defaults to Gemini voices (`Kore`, or `Puck` for `gender="MALE"`) and the default TTS model.

## 8. Building a "Gemini app"-style UI

| UI feature | Backend call |
| --- | --- |
| New conversation | `chat = w.start_chat(model, system_instruction=..., tools=...)`; store `chat.history` (JSON) per conversation |
| Resume conversation | `w.start_chat(model, history=saved_history)` |
| Streaming reply | `for t in chat.stream(message, media=attachments)` -> push chunks over SSE / websocket |
| Attachments | `media=["file.pdf", (bytes, mime), "gs://..."]`; large files: Files API (Developer API) or `gs://` (Vertex) |
| Regenerate last answer | `chat.history = chat.history[:-2]`, then send the same message again |
| Search toggle + citations | `tools=[{"googleSearch": {}}]`, use `chat.send` and `w.extract_grounding(chat.last_response)` (stream does not keep the response) |
| Show thinking | `generation_config={"thinkingConfig": {"includeThoughts": True}}` + `w.extract_text(r, include_thoughts=True)` |
| Chat title | `w.generate_text("Title for: " + first_message, model="gemini-3.5-flash-lite")` |
| Image tab | `generate_image` / `edit_image`, or an image chat (`start_chat("gemini-3.1-flash-image", generation_config={"responseModalities": ["TEXT", "IMAGE"]})`) + `extract_images` |
| Video tab | `op = generate_video(...)`; store `op["name"]`; poll `get_video_operation(name)` in a background job until `done`; `extract_videos` |
| Music tab | `generate_music` -> `extract_audio` -> `audio_to_wav` |
| Speech tab | `generate_gemini_speech` / `generate_multi_speaker_speech` -> `audio_to_wav` |
| Voice mode | `live_connect` + `send_audio` (16 kHz PCM in) + `receive()` (24 kHz PCM out) |
| Token meter | `count_tokens` before sending; `response["usageMetadata"]` after |
| Error banner / retry | catch `GoogleAIError`; back off on `status_code` 429 / 5xx; show `str(e)` (secrets are redacted) |

Minimal backend sketch:

```python
class GeminiApp:
    def __init__(self, wrapper, model="gemini-3.8-flash"):
        self.w, self.model, self.chats = wrapper, model, {}

    def chat_stream(self, conv_id, message, files=()):
        chat = self.chats.setdefault(conv_id, self.w.start_chat(self.model))
        yield from chat.stream(message, media=list(files) or None)

    def image(self, prompt, edit_from=None):
        r = self.w.edit_image(prompt, [edit_from]) if edit_from else self.w.generate_image(prompt)
        return self.w.extract_images(r)

    def speech(self, text, voice="Kore"):
        return self.w.audio_to_wav(self.w.extract_audio(self.w.generate_gemini_speech(text, voice=voice))[0])
```

## 9. Migration: GeminiAIWrapper -> GoogleAIWrapper

`GeminiAIWrapper(api_key, timeout=180, **google_options)` is deprecated. It defaults to the Developer API (`vertex=False`) unless you pass Vertex options, and keeps the old signatures, defaults, return shapes and error prefixes.

| GeminiAIWrapper (old) | GoogleAIWrapper (new) | Difference |
| --- | --- | --- |
| `GeminiAIWrapper(key)` | `GoogleAIWrapper(key)` | same Developer API config |
| `generate_content(params, vision=False, model_override=None)` | same | also `model=`; params may be a string |
| `generate_content_with_system_instructions(content_parts, system_instruction=None, model_override=None)` | same | or `generate_text(prompt, system_instruction=...)` |
| `generate_structured_content(...)` | same | |
| `stream_generate_content(params, vision, model_override)` (raw lines) | `stream_generate_content(..., raw=True)` | default now yields parsed dict chunks |
| `image_to_text`, `image_to_text_with_file_uri`, `multiple_images_to_text`, `get_bounding_boxes`, `get_image_segmentation` | same, + `model_override` | |
| `image_to_text_params(params, model_override=None)` | same | |
| `generate_image(prompt, config_params=None, model_override=None)` | same, + `images=` | |
| `generate_video(prompt, config_params=None, project_id=None)` | `generate_video(prompt, config_params=None, project_id=None, *, model, image, last_frame, location)` | old one requires a project and adds `personGeneration: "dont_allow"`; new one does neither |
| `check_video_generation_status(op, project_id)` / `wait_for_video_completion(op, project_id, ...)` | same; `project_id` optional | the operation name carries the project |
| `generate_speech(text, voice_config=None, model_override=None)` | `generate_gemini_speech(text, voice_config=None, model_override=None, *, voice=None)` | `GoogleAIWrapper.generate_speech` is Cloud TTS |
| `generate_multi_speaker_speech(text, speaker_configs)` | same, + `model_override` | |
| `upload_file`, `list_files`, `delete_file` | same, + `get_file` | Developer API only |
| `get_embeddings(params)`, `get_batch_embeddings(params)` | same | or `embed_texts(texts)` |
| `_get_mime_type(path)` | `media_part(...)` | |

## 10. Errors and hints

```python
try:
    w.generate_text("hi", model="gemini-does-not-exist")
except GoogleAIError as e:
    print(e.status_code)   # HTTP status (400, 403, 404, 429, ...) or None
    print(e.details)       # parsed JSON error body (or text), secrets redacted
    print(str(e))          # "<prefix>: <HTTP error> - Details: {...} (hint: ...)"
```

- `GoogleAIError(message, status_code=None, details=None)` subclasses `Exception`, so old `except Exception` code still works.
- API keys, access tokens, credential tokens and `?key=` values are replaced by `<redacted>` in the message and `details`.
- Prefixes: `Gemini API error`, `Gemini stream error`, `Gemini Image Generation error`, `Imagen error`, `Imagen edit error`, `Imagen upscale error`, `Veo Video Generation error`, `Video status check error`, `Download error`, `Lyria music error`, `Gemini TTS error`, `Gemini Multi-Speaker TTS error`, `Gemini countTokens error`, `Gemini computeTokens error`, `File upload error`, `List files error`, `Get file error`, `Delete file error`, `List models error`, `Context cache error`, `Agent Engine error`, `Live API connection error`, `Live API setup error`, `Live API setup failed`.

| Error text | Hint appended | Fix |
| --- | --- | --- |
| `API_KEY_SERVICE_BLOCKED` on the Developer API | key not enabled for the Developer API | it is an Agent Platform key: add `vertex=True` |
| `API keys are not supported by this API` on Vertex | endpoint needs OAuth | ADC (`gcloud auth application-default login`) or `access_token=` |
| `RESOURCE_PROJECT_INVALID` on Vertex | endpoint needs a project | add `project_id=` |

Other exceptions: `ValueError` (no default model for a kind, empty model, bytes without `mime_type`, unknown path / URI, Veo in express mode, a Vertex operation name not starting with `projects/`), `NotImplementedError` (Files API on Vertex), `TimeoutError` (`wait_for_video_completion`), `FileNotFoundError` (`upload_file`), and `GoogleAIError` without status for local checks (missing project, Vertex-only features on the Developer API, missing `google-auth` / `websockets`, ADC failures).

## 11. Gotchas

1. **Developer API defaults are 2.5-era** (`config['url']['gemini']['models']`: text `gemini-2.5-flash`, TTS `gemini-2.5-flash-preview-tts`, video `veo-2.0-generate-001`). Gemini 2.5 text models retire on **2026-10-20**. Pass `model=` explicitly on the Developer API. Vertex defaults are Gemini 3.x (section 12).
2. **Explicit location wins everywhere.** With `location="global"` (or `GOOGLE_CLOUD_LOCATION`), Veo and Live also go to `global`, where the Live models are not served. Either omit `location`, or pass `location="us-central1"` per call (`generate_video`, `generate_music`, `live_connect` / `live_generate`, Agent Engine methods). Imagen methods have no `location` argument.
3. The Live models (`gemini-3.8-live`) are served only from `us-central1`. `gemini-3.8-flash`, `gemini-3.5-flash` and `gemini-3.1-flash-image` are served from `global` and express mode.
4. Veo, the Live API, Agent Engine and Vertex context caching need `project_id`.
5. On Vertex, model listing, context caching and `gemini-embedding-2` need OAuth; an API key is rejected. Use `model_catalog()` for an offline list.
6. Thinking models can spend a small `maxOutputTokens` entirely on thoughts and return empty text (Chatbot then returns `""`). Raise the limit or set `thinkingConfig`: `{"thinkingLevel": "low"}` for Gemini 3, `{"thinkingBudget": 0}` for 2.5 Flash.
7. Gemini 3.6+ ignores `temperature` / `topP` / `topK` and rejects `presencePenalty` / `frequencyPenalty` (400).
8. Gemini 2.5 TTS returns raw PCM (`audio/L16;rate=24000`); Gemini 3.8 TTS returns WAV. Always save with `audio_to_wav`. Flow speech agents return only the base64 data, so wrap 2.5 output with `pcm_to_wav`.
9. `generate_speech` is Cloud TTS (params dict with `text`, `languageCode`, `name`, `ssmlGender`). Gemini TTS is `generate_gemini_speech`.
10. Vertex rejects `contents` without `role`. The wrapper adds it (`user`, or `model` for `functionCall` parts) for every body it sends.
11. Responses carry snake_case aliases next to camelCase keys. Do not paste raw response parts back into a request; `start_chat` strips the aliases for you.
12. A Files API URI has no file extension: pass the mime type (`media_part(f["uri"], f["mimeType"])`), and wait for `state == "ACTIVE"` before using videos.
13. Veo `image` / `last_frame` given as a path or bytes are sent with `mimeType: image/png` whatever the real format. Use PNG files, or pass a dict `{"bytesBase64Encoded": b64, "mimeType": "image/jpeg"}`.
14. `chat.stream` does not set `chat.last_response`; use `chat.send` when you need grounding metadata or function calls. Parallel function calls go back in one turn: `chat.send(parts=[...])`.
15. `live_generate` calls `asyncio.run`; inside Jupyter / FastAPI use `await live_generate_async(...)`.
16. Imagen is deprecated by Google in favor of Gemini image models; a retired Imagen model fails with a deprecation message.
17. Some organizations disallow API keys. Use ADC: `gcloud auth application-default login`, then `GoogleAIWrapper(project_id=PROJECT)`.
18. `GOOGLE_GENAI_USE_VERTEXAI=true` (or `GOOGLE_GENAI_USE_ENTERPRISE`) in the environment switches every `GoogleAIWrapper` built without an explicit `vertex=` to Vertex, including the ones `Chatbot`, the controllers and Flow agents build. Pass `{"vertex": False}` in options to stay on the Developer API. `GeminiAIWrapper` always defaults to `vertex=False`.
19. Never print `wrapper.api_key`, and never commit keys. Keys go only in the gitignored `intelli/.env`.

## 12. Model catalog

`GoogleAIWrapper.model_catalog()` returns the Vertex model ids known to work, by capability (from `config['url']['gemini']['vertex']['catalog']`). `w.models` shows the defaults of a given wrapper.

| Kind | Developer API default | Vertex default |
| --- | --- | --- |
| `text` / `vision` | `gemini-2.5-flash` | `gemini-3.8-flash` |
| `image_generation` | `gemini-2.5-flash-image` | `gemini-3.1-flash-image` |
| `imagen` / `imagen_edit` / `imagen_upscale` | `imagen-4.0-generate-001` / none / none | `imagen-4.0-generate-001` / `imagen-3.0-capability-001` / `imagen-4.0-upscale-preview` |
| `video_generation` | `veo-2.0-generate-001` | `veo-3.1-fast-generate-001` |
| `tts` / `tts_pro` | `gemini-2.5-flash-preview-tts` / `gemini-2.5-pro-preview-tts` | `gemini-2.5-flash-tts` / `gemini-2.5-pro-tts` |
| `music` | `lyria-3.5` | `lyria-002` |
| `live` | `gemini-3.8-live` | `gemini-3.8-live` |
| `embedding` | `gemini-embedding-001` | `gemini-embedding-001` |

Catalog highlights: text `gemini-3.8-flash`, `gemini-3.7-flash`, `gemini-3.6-flash`, `gemini-3.5-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-pro-preview`; image `gemini-3.1-flash-image`, `gemini-3-pro-image`; video `veo-3.1-generate-001`, `veo-3.1-fast-generate-001`, `veo-3.1-lite-generate-001`; TTS `gemini-3.8-flash-tts`, `gemini-3.8-flash-lite-tts`; live `gemini-3.8-live`, `gemini-live-2.5-flash-native-audio`; embedding `gemini-embedding-001`, `text-embedding-005`, `text-multilingual-embedding-002`. To change a default, edit `config['url']['gemini']['models']` (Developer API) or `config['url']['gemini']['vertex']['models']` (Vertex).

## 13. Running the tests

Keys live only in the gitignored `intelli/.env` (template: `intelli/.example.env`):

| Variable | Used for |
| --- | --- |
| `GEMINI_API_KEY` | Developer API tests and the old `GeminiAIWrapper` tests |
| `GOOGLE_API_KEY` | Google Cloud API tests (`test_googleai_wrapper.py`: Cloud TTS, Vision, Language, Translation) |
| `VERTEX_API_KEY` | Vertex express and project tests (every Vertex class skips without it) |
| `VERTEX_PROJECT_ID` | project-scoped tests: Live API, Agent Engine, Veo, Chatbot / Flow on Vertex |
| `INTELLI_RUN_VEO=1` | opt in to the slow, billed Veo test |

```bash
# Unit tests (mocked HTTP, no keys, no network), from the repo root
PYTHONPATH=. python3 -m pytest -q intelli/test/unit/test_googleai_genai_core.py \
    intelli/test/unit/test_googleai_genai_features.py intelli/test/unit/test_gemini_facade_compat.py

# Live tests (real API calls; run from the inner intelli/ dir so load_dotenv finds intelli/.env)
cd <repo>/intelli
PYTHONPATH=<repo> python3 -m pytest test/integration/test_googleai_vertex_live.py -q
PYTHONPATH=<repo> python3 -m pytest test/integration/test_googleai_gemini_dev_live.py -q
INTELLI_RUN_VEO=1 PYTHONPATH=<repo> python3 -m pytest test/integration/test_googleai_vertex_live.py -q -k Veo

# Old Gemini tests, now running through the deprecated facade (Developer API key)
PYTHONPATH=<repo> python3 -m pytest test/integration/test_geminiai_wrapper.py \
    test/integration/test_geminiai_latest_wrapper.py test/integration/test_gemini_structured_params.py -q
```

| File | Covers |
| --- | --- |
| `unit/test_googleai_genai_core.py` | backend selection, URLs and capability locations, auth headers, errors and redaction, body normalization, SSE streaming, text, tokens, chat sessions, extractors, audio helpers, `media_part`, `from_options` |
| `unit/test_googleai_genai_features.py` | feature methods on each backend (exact URL, method, headers, body): Gemini images, Imagen, Veo, Lyria, TTS, embeddings, Files API, models, caching, Agent Engine, Live API (fake websocket) |
| `unit/test_gemini_facade_compat.py` | the deprecated `GeminiAIWrapper` contract, plus Chatbot, controllers and Flow agents building `GoogleAIWrapper` |
| `integration/test_googleai_vertex_live.py` | Vertex express + project live calls (text, tools, media, embeddings, image, Imagen, TTS, Lyria, Live, Agent Engine, Veo, Chatbot / Flow) |
| `integration/test_googleai_gemini_dev_live.py` | Developer API live calls, and the key-blocked hint |

Live tests save generated media under `temp/` and retry 429 / 5xx errors. Do not paste keys into commands, test names or assertion messages.

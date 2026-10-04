"""
Live integration tests for GoogleAIWrapper on Vertex AI (Gemini Enterprise Agent Platform).

These tests call Google and are billed. Prompts are tiny, thinking is set low and media
generation is limited to a few images, one music clip and (opt-in) one short Veo video.

Environment:
    VERTEX_API_KEY           Agent Platform API key. Every class is skipped without it.
    VERTEX_PROJECT_ID        Google Cloud project for project-scoped features (Live API,
                             Agent Engine, Veo, Intelli Chatbot / Flow integration).
    INTELLI_RUN_VEO=1        Also run the Veo video test (slow: about 1-3 minutes).
    INTELLI_LIVE_OUTPUT_DIR  Folder for generated media (default: temp/vertex_live).

Run:
    python3 -m pytest intelli/test/integration/test_googleai_vertex_live.py -q
"""
import asyncio
import base64
import io
import json
import math
import os
import time
import unittest
import wave

from dotenv import load_dotenv

from intelli.controller.remote_embed_model import RemoteEmbedModel
from intelli.controller.remote_speech_model import RemoteSpeechModel
from intelli.flow.agents.agent import Agent
from intelli.flow.flow import Flow
from intelli.flow.input.task_input import TextTaskInput
from intelli.flow.tasks.task import Task
from intelli.function.chatbot import Chatbot
from intelli.model.input.chatbot_input import ChatModelInput
from intelli.model.input.embed_input import EmbedInput
from intelli.model.input.text_speech_input import Text2SpeechInput
from intelli.wrappers.googleai_wrapper import GoogleAIWrapper, GoogleAIError

load_dotenv()

TEXT_MODEL = "gemini-3.8-flash"
TEXT_MODEL_3_5 = "gemini-3.5-flash"
IMAGE_MODEL = "gemini-3.1-flash-image"
IMAGEN_MODEL = "imagen-4.0-fast-generate-001"
VEO_MODEL = "veo-3.1-fast-generate-001"
LOW_THINKING = {"thinkingConfig": {"thinkingLevel": "LOW"}}

SCONES_IMAGE = "gs://cloud-samples-data/generative-ai/image/scones.jpg"
PIXEL_AUDIO = "gs://cloud-samples-data/generative-ai/audio/pixel.mp3"
ANIMALS_VIDEO = "gs://cloud-samples-data/video/animals.mp4"

# Rate limits and overloads are retried so a busy minute does not fail the suite.
TRANSIENT_STATUS = {429, 500, 502, 503, 504}


def output_path(name):
    folder = os.getenv("INTELLI_LIVE_OUTPUT_DIR") or os.path.join("temp", "vertex_live")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, name)


def save_bytes(name, data):
    path = output_path(name)
    with open(path, "wb") as file:
        file.write(data)
    return path


def image_extension(mime_type):
    return "jpg" if "jpeg" in (mime_type or "") else (mime_type or "image/png").split("/")[-1]


def is_image(data):
    return data[:8] == b"\x89PNG\r\n\x1a\n" or data[:3] == b"\xff\xd8\xff" or data[8:12] == b"WEBP"


def wav_seconds(wav_bytes):
    """Duration of the audio data actually present (Lyria WAV headers can overstate the data size)."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        frame_size = wav_file.getsampwidth() * wav_file.getnchannels()
        data = wav_file.readframes(wav_file.getnframes())
        return len(data) / float(frame_size * wav_file.getframerate())


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def evidence(feature, detail):
    print(f"\n[vertex-live] {feature}: {str(detail)[:200]}")


class VertexLiveTestCase(unittest.TestCase):
    """Shared setup: an express-mode wrapper and, when VERTEX_PROJECT_ID is set, a project wrapper."""

    needs_project = False

    @classmethod
    def setUpClass(cls):
        cls.api_key = os.getenv("VERTEX_API_KEY")
        cls.project_id = os.getenv("VERTEX_PROJECT_ID")
        if not cls.api_key:
            raise unittest.SkipTest("VERTEX_API_KEY not set")
        if cls.needs_project and not cls.project_id:
            raise unittest.SkipTest("VERTEX_PROJECT_ID not set")
        cls.express = GoogleAIWrapper(cls.api_key, vertex=True)
        cls.project = (GoogleAIWrapper(cls.api_key, vertex=True, project_id=cls.project_id)
                       if cls.project_id else None)

    def call(self, fn, *args, attempts=3, **kwargs):
        """Call fn, retrying transient API errors (429 / 5xx) with a short backoff."""
        for attempt in range(attempts):
            try:
                return fn(*args, **kwargs)
            except GoogleAIError as error:
                if error.status_code not in TRANSIENT_STATUS or attempt == attempts - 1:
                    raise
                time.sleep(10 * (attempt + 1))

    def project_wrapper(self):
        if not self.project:
            self.skipTest("VERTEX_PROJECT_ID not set")
        return self.project

    def assertKeyNotLeaked(self, text):
        # Never put the key itself in an assertion message.
        self.assertFalse(self.api_key in str(text), "the API key leaked into the error message")


class TestVertexTextLive(VertexLiveTestCase):
    """Text, chat, streaming, structured output, function calling and token counting."""

    def test_express_text_default_model(self):
        self.assertEqual(self.express.models["text"], TEXT_MODEL)
        text = self.call(self.express.generate_text, "What is the capital of France? Answer with one word.",
                         generation_config=LOW_THINKING)
        evidence("express text (default gemini-3.8-flash)", text)
        self.assertIn("paris", text.lower())

    def test_express_text_gemini_3_5_flash(self):
        text = self.call(self.express.generate_text, "What is the capital of Italy? Answer with one word.",
                         model=TEXT_MODEL_3_5, generation_config=LOW_THINKING)
        evidence("express text gemini-3.5-flash", text)
        self.assertIn("rome", text.lower())

    def test_project_text_default_model(self):
        project = self.project_wrapper()
        text = self.call(project.generate_text, "What is the capital of Spain? Answer with one word.",
                         generation_config=LOW_THINKING)
        evidence("project text (default gemini-3.8-flash, global)", text)
        self.assertIn("madrid", text.lower())

    def test_project_text_gemini_3_5_flash(self):
        project = self.project_wrapper()
        text = self.call(project.generate_text, "What is the capital of Germany? Answer with one word.",
                         model=TEXT_MODEL_3_5, generation_config=LOW_THINKING)
        evidence("project text gemini-3.5-flash", text)
        self.assertIn("berlin", text.lower())

    def test_system_instruction(self):
        text = self.call(self.express.generate_text, "What color is a clear daytime sky? One word.",
                         system_instruction="Always answer in UPPERCASE letters only.",
                         generation_config=LOW_THINKING)
        evidence("system instruction", text)
        self.assertIn("BLUE", text)
        self.assertEqual(text.strip(), text.strip().upper())

    def test_snake_case_generation_config(self):
        response = self.call(self.express.generate_content, {
            "contents": [{"parts": [{"text": "What is 2 + 3? Reply with the number only."}]}],
            "system_instruction": {"parts": [{"text": "Reply with digits only."}]},
            "generation_config": {"max_output_tokens": 1024, "temperature": 0.2,
                                  "thinking_config": {"thinking_level": "LOW"}},
        }, model=TEXT_MODEL)
        text = GoogleAIWrapper.extract_text(response)
        evidence("snake_case generation_config", text)
        self.assertIn("5", text)

    def test_multi_turn_chat_remembers_fact(self):
        chat = self.express.start_chat(model=TEXT_MODEL, system_instruction="Be very brief.",
                                       generation_config=LOW_THINKING)
        self.call(chat.send_text, "My favorite color is teal. Reply with OK.")
        answer = self.call(chat.send_text, "What is my favorite color? One word.")
        evidence("multi-turn chat", f"{answer!r}, history={len(chat.history)}")
        self.assertIn("teal", answer.lower())
        self.assertEqual(len(chat.history), 4)

    def test_chat_stream_adds_turn_to_history(self):
        chat = self.express.start_chat(model=TEXT_MODEL, generation_config=LOW_THINKING)
        chunks = self.call(lambda: list(chat.stream("Name the largest planet in the solar system. One word.")))
        evidence("chat stream", f"chunks={len(chunks)} text={''.join(chunks)!r}")
        self.assertIn("jupiter", "".join(chunks).lower())
        self.assertEqual([turn["role"] for turn in chat.history], ["user", "model"])

    def test_stream_text_yields_several_chunks(self):
        chunks = self.call(lambda: list(self.express.stream_text(
            "Count from 1 to 60, separated by commas. Output only the numbers.", generation_config=LOW_THINKING)))
        text = "".join(chunks)
        evidence("stream_text", f"chunks={len(chunks)} tail={text[-40:]!r}")
        self.assertGreater(len(chunks), 1)
        self.assertIn("60", text)

    def test_structured_json_output(self):
        schema = {
            "type": "OBJECT",
            "properties": {"city": {"type": "STRING"}, "country": {"type": "STRING"}},
            "required": ["city", "country"],
        }
        response = self.call(self.express.generate_structured_content,
                             [{"text": "Give the capital city of Japan and its country."}], schema,
                             model_override=TEXT_MODEL, generation_config=LOW_THINKING)
        data = json.loads(GoogleAIWrapper.extract_text(response))
        evidence("structured JSON", data)
        self.assertIn("tokyo", data["city"].lower())
        self.assertIn("japan", data["country"].lower())

    def test_function_calling_round_trip(self):
        tools = [{"functionDeclarations": [{
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {"type": "OBJECT", "properties": {"city": {"type": "STRING"}}, "required": ["city"]},
        }]}]
        chat = self.express.start_chat(model=TEXT_MODEL, tools=tools, generation_config=LOW_THINKING)
        first = self.call(chat.send, "What is the weather in Paris right now? Use the tool.")
        calls = GoogleAIWrapper.extract_function_calls(first)
        self.assertTrue(calls, "expected a functionCall part")
        self.assertEqual(calls[0]["name"], "get_weather")
        self.assertIn("paris", json.dumps(calls[0].get("args", {})).lower())

        final = self.call(chat.send_function_response, "get_weather",
                          {"temperature_c": 21, "condition": "sunny"}, call_id=calls[0].get("id"))
        text = GoogleAIWrapper.extract_text(final)
        evidence("function calling", f"call={calls[0]} final={text!r}")
        self.assertIn("21", text)

    def test_count_tokens(self):
        result = self.call(self.express.count_tokens, "Hello world, how are you today?")
        evidence("count_tokens", result)
        self.assertGreater(result.get("totalTokens", 0), 0)

    def test_compute_tokens(self):
        result = self.call(self.express.compute_tokens, "Hello world")
        info = result["tokensInfo"][0]
        evidence("compute_tokens", {"tokenIds": info.get("tokenIds"), "tokens": len(info.get("tokens", []))})
        self.assertTrue(info.get("tokenIds"))
        self.assertEqual(len(info["tokenIds"]), len(info.get("tokens", [])))


class TestVertexToolsLive(VertexLiveTestCase):
    """Built-in tools: Google Search grounding, code execution and URL context."""

    def test_google_search_grounding(self):
        response = self.call(self.express.generate_content, {
            "contents": "Which country won the most recent FIFA World Cup? Answer in one short sentence.",
            "tools": [{"googleSearch": {}}],
            "generationConfig": LOW_THINKING,
        }, model=TEXT_MODEL)
        grounding = GoogleAIWrapper.extract_grounding(response)
        evidence("google search grounding",
                 f"queries={grounding.get('webSearchQueries')} chunks={len(grounding.get('groundingChunks') or [])}")
        self.assertTrue(grounding, "expected groundingMetadata")
        self.assertTrue(grounding.get("webSearchQueries") or grounding.get("groundingChunks"))
        self.assertTrue(GoogleAIWrapper.extract_text(response).strip())

    def test_code_execution(self):
        response = self.call(self.express.generate_content, {
            "contents": "Write and run Python code that prints the sum of the first 20 prime numbers.",
            "tools": [{"codeExecution": {}}],
            "generationConfig": LOW_THINKING,
        }, model=TEXT_MODEL)
        parts = list(GoogleAIWrapper._iter_parts(response))
        code = [p["executableCode"] for p in parts if "executableCode" in p]
        results = [p["codeExecutionResult"] for p in parts if "codeExecutionResult" in p]
        evidence("code execution", f"code_parts={len(code)} result={results[:1]}")
        self.assertTrue(code, "expected an executableCode part")
        self.assertTrue(results, "expected a codeExecutionResult part")
        self.assertIn("639", json.dumps(results) + GoogleAIWrapper.extract_text(response))

    def test_url_context(self):
        response = self.call(self.express.generate_content, {
            "contents": "In one sentence, what does https://www.intellinode.ai offer?",
            "tools": [{"urlContext": {}}],
            "generationConfig": LOW_THINKING,
        }, model=TEXT_MODEL)
        candidate = response["candidates"][0]
        url_metadata = (candidate.get("urlContextMetadata") or {}).get("urlMetadata") or []
        text = GoogleAIWrapper.extract_text(response)
        evidence("url context", f"urls={[(u.get('retrievedUrl'), u.get('urlRetrievalStatus')) for u in url_metadata]} "
                                f"text={text!r}")
        self.assertTrue(url_metadata, "expected urlContextMetadata.urlMetadata")
        self.assertIn("intellinode", json.dumps(url_metadata).lower())
        self.assertTrue(text.strip())


class TestVertexMediaUnderstandingLive(VertexLiveTestCase):
    """Image, audio and video understanding from public gs:// samples."""

    def test_image_understanding_from_gcs(self):
        text = self.call(self.express.media_to_text, "What baked food is shown? Answer in one word.",
                         SCONES_IMAGE, generation_config=LOW_THINKING)
        evidence("image understanding (scones.jpg)", text)
        self.assertIn("scone", text.lower())

    def test_audio_understanding_from_gcs(self):
        text = self.call(self.express.audio_to_text, PIXEL_AUDIO,
                         prompt="Which product line is discussed in this audio? Answer in a few words.",
                         generation_config=LOW_THINKING)
        evidence("audio understanding (pixel.mp3)", text)
        self.assertIn("pixel", text.lower())

    def test_video_understanding_from_gcs(self):
        text = self.call(self.express.video_to_text, ANIMALS_VIDEO,
                         prompt="Name one animal you can see. Answer with one word.",
                         video_metadata={"endOffset": "10s"},
                         generation_config={**LOW_THINKING, "mediaResolution": "MEDIA_RESOLUTION_LOW"})
        evidence("video understanding (animals.mp4, first 10 s)", text)
        self.assertTrue(text.strip())
        self.assertLessEqual(len(text.split()), 5)


class TestVertexEmbeddingsLive(VertexLiveTestCase):

    def test_embed_texts_with_output_dimensionality(self):
        vectors = self.call(self.express.embed_texts,
                            ["The cat sat on the mat.", "A kitten rests on a rug.", "Quarterly revenue grew 5%."],
                            output_dimensionality=64)
        similar, different = cosine(vectors[0], vectors[1]), cosine(vectors[0], vectors[2])
        evidence("embed_texts", f"count={len(vectors)} dims={[len(v) for v in vectors]} "
                                f"cos(similar)={similar:.3f} cos(different)={different:.3f}")
        self.assertEqual([len(v) for v in vectors], [64, 64, 64])
        self.assertGreater(similar, different)

    def test_get_embeddings_compat_shape(self):
        result = self.call(self.express.get_embeddings, {"content": {"parts": [{"text": "hello world"}]}})
        values = result["embedding"]["values"]
        evidence("get_embeddings compat", f"keys={list(result)} dims={len(values)}")
        self.assertGreater(len(values), 100)
        self.assertTrue(all(isinstance(v, float) for v in values[:10]))


class TestVertexMediaGenerationLive(VertexLiveTestCase):
    """Gemini images, Imagen, Gemini TTS and Lyria music (express mode)."""

    def test_gemini_image_generate_then_edit(self):
        response = self.call(self.express.generate_image, "A flat icon of a single red circle on a white background",
                             {"imageConfig": {"aspectRatio": "1:1"}}, model_override=IMAGE_MODEL)
        images = GoogleAIWrapper.extract_images(response)
        self.assertTrue(images, "expected an image part")
        original = base64.b64decode(images[0]["data"])
        self.assertTrue(is_image(original))
        path = save_bytes(f"gemini_image.{image_extension(images[0]['mime_type'])}", original)

        edited_response = self.call(self.express.edit_image, "Change the circle color to blue. Keep the rest.",
                                    [(original, images[0]["mime_type"])], model_override=IMAGE_MODEL)
        edited_images = GoogleAIWrapper.extract_images(edited_response)
        self.assertTrue(edited_images, "expected an edited image part")
        edited = base64.b64decode(edited_images[0]["data"])
        self.assertTrue(is_image(edited))
        self.assertNotEqual(edited, original)
        edited_path = save_bytes(f"gemini_image_edited.{image_extension(edited_images[0]['mime_type'])}", edited)
        evidence("gemini image generate + edit",
                 f"{images[0]['mime_type']} {len(original)} B -> {edited_images[0]['mime_type']} {len(edited)} B "
                 f"({os.path.basename(path)}, {os.path.basename(edited_path)})")

    def test_imagen_generate(self):
        try:
            response = self.call(self.express.imagen_generate_images, "A simple flat icon of a blue square",
                                 number_of_images=1, model=IMAGEN_MODEL)
        except GoogleAIError as error:
            # Google deprecated the Imagen endpoints (discontinued after 2026-06-30, replaced by Gemini
            # image models). A retired model answers 404 "Publisher model ... was not found".
            message = str(error)
            retired = "deprecat" in message.lower() or (error.status_code == 404 and IMAGEN_MODEL in message)
            if not retired:
                raise
            # The error must still be clear: which call, which model, the HTTP status, and no key.
            self.assertTrue(message.startswith("Imagen error"))
            self.assertIn(IMAGEN_MODEL, message)
            self.assertKeyNotLeaked(message)
            evidence("imagen generate (retired by Google)", f"status={error.status_code} {message[:160]}")
            self.skipTest(f"{IMAGEN_MODEL} is no longer served by Google (status {error.status_code})")
        images = GoogleAIWrapper.extract_images(response)
        self.assertEqual(len(images), 1)
        data = base64.b64decode(images[0]["data"])
        self.assertTrue(is_image(data))
        save_bytes(f"imagen.{image_extension(images[0]['mime_type'])}", data)
        evidence("imagen generate", f"{images[0]['mime_type']} {len(data)} B")

    def _check_tts(self, model):
        response = self.call(self.express.generate_gemini_speech, "Hello from Intelli.", model_override=model)
        audio = GoogleAIWrapper.extract_audio(response)
        self.assertTrue(audio, "expected an audio part")
        wav = GoogleAIWrapper.audio_to_wav(audio[0])
        self.assertEqual(wav[:4], b"RIFF")
        seconds = wav_seconds(wav)
        self.assertGreater(seconds, 0.3)
        save_bytes(f"{model}.wav", wav)
        evidence(f"gemini tts {model}", f"{audio[0]['mime_type']} -> WAV {len(wav)} B, {seconds:.1f} s")

    def test_gemini_tts_2_5_flash(self):
        self._check_tts("gemini-2.5-flash-tts")

    def test_gemini_tts_3_8_flash(self):
        self._check_tts("gemini-3.8-flash-tts")

    def test_multi_speaker_tts(self):
        speakers = [
            {"speaker": "Joe", "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}},
            {"speaker": "Jane", "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Puck"}}},
        ]
        response = self.call(self.express.generate_multi_speaker_speech,
                             "TTS the following conversation between Joe and Jane:\nJoe: Hi Jane.\nJane: Hi Joe.",
                             speakers)
        audio = GoogleAIWrapper.extract_audio(response)
        self.assertTrue(audio, "expected an audio part")
        wav = GoogleAIWrapper.audio_to_wav(audio[0])
        self.assertEqual(wav[:4], b"RIFF")
        seconds = wav_seconds(wav)
        self.assertGreater(seconds, 0.5)
        save_bytes("multi_speaker.wav", wav)
        evidence("multi-speaker tts", f"{audio[0]['mime_type']} -> WAV {len(wav)} B, {seconds:.1f} s")

    def test_lyria_music(self):
        response = self.call(self.express.generate_music, "A short calm piano melody", model="lyria-002")
        audio = GoogleAIWrapper.extract_audio(response)
        self.assertEqual(len(audio), 1)
        wav = GoogleAIWrapper.audio_to_wav(audio[0])
        self.assertEqual(wav[:4], b"RIFF")
        seconds = wav_seconds(wav)
        self.assertGreater(seconds, 5)
        save_bytes("lyria-002.wav", wav)
        evidence("lyria-002 music", f"{audio[0]['mime_type']} {len(wav)} B, {seconds:.1f} s")


class TestVertexProjectLive(VertexLiveTestCase):
    """Project-scoped features: the Live API and Agent Engine."""

    needs_project = True

    def test_live_api_text_turn(self):
        result = self.project.live_generate("Say hello in one short sentence.")
        evidence("live api", f"transcription={result['transcription']!r} audio={len(result['audio'])} B "
                             f"mime={result['audio_mime_type']}")
        self.assertTrue(result["transcription"].strip())
        self.assertGreater(len(result["audio"]), 1000)
        save_bytes("live.wav", GoogleAIWrapper.audio_to_wav({"mime_type": result["audio_mime_type"] or "audio/pcm",
                                                             "data": result["audio"]}))

    def test_agent_engine_list(self):
        result = self.call(self.project.list_agent_engines, page_size=5)
        engines = result.get("reasoningEngines", [])
        evidence("agent engine list", f"keys={list(result)} engines={len(engines)}")
        self.assertIsInstance(result, dict)
        self.assertIsInstance(engines, list)
        for engine in engines:
            self.assertTrue(engine["name"].startswith("projects/"))


class TestVertexErrorsLive(VertexLiveTestCase):
    """Clear errors for features an API key or Vertex AI cannot use."""

    def test_list_models_with_api_key_explains_oauth(self):
        with self.assertRaises(GoogleAIError) as context:
            self.express.list_models(page_size=2)
        message = str(context.exception)
        evidence("list_models with API key", f"status={context.exception.status_code} message={message[-160:]!r}")
        self.assertIn("OAuth", message)
        self.assertKeyNotLeaked(message)

    def test_files_api_not_implemented_on_vertex(self):
        with self.assertRaises(NotImplementedError) as context:
            self.express.list_files()
        evidence("files api on vertex", context.exception)
        self.assertIn("Developer API", str(context.exception))


@unittest.skipUnless(os.getenv("INTELLI_RUN_VEO") == "1", "set INTELLI_RUN_VEO=1 to run the Veo video test")
class TestVertexVeoLive(VertexLiveTestCase):
    """Veo text-to-video (opt-in: slow and billed per second of video)."""

    needs_project = True

    def test_veo_text_to_video(self):
        operation = self.call(self.project.generate_video, "A paper boat floating on a calm pond, gentle ripples",
                              {"durationSeconds": 4, "resolution": "720p", "generateAudio": False,
                               "sampleCount": 1}, model=VEO_MODEL)
        self.assertTrue(operation.get("name", "").startswith("projects/"))
        started = time.time()
        finished = self.project.wait_for_video_completion(operation["name"], max_wait_time=600, poll_interval=10)
        self.assertNotIn("error", finished)
        videos = GoogleAIWrapper.extract_videos(finished)
        self.assertEqual(len(videos), 1)
        data = base64.b64decode(videos[0]["data"])
        self.assertEqual(data[4:8], b"ftyp")
        save_bytes("veo.mp4", data)
        evidence("veo video", f"{videos[0]['mime_type']} {len(data)} B after {time.time() - started:.0f} s")


class TestVertexIntelliIntegrationLive(VertexLiveTestCase):
    """Chatbot, Flow agents and controllers with Vertex options."""

    needs_project = True

    def vertex_options(self):
        return {"vertex": True, "project_id": self.project_id}

    def test_chatbot_chat(self):
        bot = Chatbot(self.api_key, "gemini", self.vertex_options())
        chat_input = ChatModelInput("Answer with one word.", model=TEXT_MODEL)
        chat_input.add_user_message("What is the capital of France?")
        result = self.call(bot.chat, chat_input)
        evidence("chatbot chat (vertex project)", result)
        self.assertIsInstance(result, list)
        self.assertIn("paris", result[0].lower())

    def test_chatbot_stream(self):
        bot = Chatbot(self.api_key, "gemini", self.vertex_options())
        chat_input = ChatModelInput("Output only numbers.", model=TEXT_MODEL)
        chat_input.add_user_message("Count from 1 to 40, separated by commas.")
        chunks = self.call(lambda: list(bot.stream(chat_input)))
        text = "".join(chunks)
        evidence("chatbot stream (vertex project)", f"chunks={len(chunks)} tail={text[-30:]!r}")
        self.assertTrue(chunks)
        self.assertIn("40", text)

    def test_flow_text_agent_feeds_image_agent(self):
        text_agent = Agent(agent_type="text", provider="gemini",
                           mission="Write a one-sentence visual description for an icon generator",
                           model_params={"key": self.api_key, "model": TEXT_MODEL}, options=self.vertex_options())
        image_agent = Agent(agent_type="image", provider="gemini", mission="Generate a flat minimal icon",
                            model_params={"key": self.api_key, "model": IMAGE_MODEL}, options=self.vertex_options())
        flow = Flow(tasks={"describe": Task(TextTaskInput("a red apple"), text_agent),
                           "draw": Task(TextTaskInput("Draw the icon from the description"), image_agent)},
                    map_paths={"describe": ["draw"]})
        output = asyncio.run(flow.start())
        self.assertEqual(flow.errors, {})
        description = output["describe"]["output"]
        image = base64.b64decode(output["draw"]["output"])
        evidence("flow text -> image (vertex project)", f"description={description!r} image={len(image)} B")
        self.assertIn("apple", description.lower())
        self.assertTrue(is_image(image))
        save_bytes("flow_icon.png", image)

    def test_remote_speech_model_gemini_vertex(self):
        speech = RemoteSpeechModel(self.api_key, "gemini", options={"vertex": True})
        audio_b64 = self.call(speech.generate_speech, Text2SpeechInput("Hello from Intelli on Vertex."))
        self.assertIsInstance(audio_b64, str)
        wav = GoogleAIWrapper.pcm_to_wav(audio_b64)
        seconds = wav_seconds(wav)
        evidence("RemoteSpeechModel gemini (vertex express)", f"pcm={len(base64.b64decode(audio_b64))} B "
                                                              f"{seconds:.1f} s")
        self.assertGreater(seconds, 0.5)
        save_bytes("remote_speech.wav", wav)

    def test_remote_embed_model_gemini_vertex(self):
        embedder = RemoteEmbedModel(self.api_key, "gemini", options=self.vertex_options())
        result = self.call(embedder.get_embeddings, EmbedInput(["hello world"]))
        values = result["embedding"]["values"]
        evidence("RemoteEmbedModel gemini (vertex project)", f"dims={len(values)}")
        self.assertGreater(len(values), 100)


if __name__ == "__main__":
    unittest.main()

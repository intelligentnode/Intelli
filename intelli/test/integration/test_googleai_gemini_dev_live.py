"""
Live tests for the Gemini methods of GoogleAIWrapper on the Gemini Developer API
(generativelanguage.googleapis.com, AI Studio key).

Environment:
    GEMINI_API_KEY             Gemini Developer API key (TestGoogleAIGeminiDeveloperLive).
    VERTEX_API_KEY             Vertex AI / Agent Platform key, only for TestGoogleAIDeveloperKeyHint.
    GOOGLEAI_TEST_OUTPUT_DIR   Where generated media is saved (default ./temp/googleai_live).
    GOOGLEAI_STRICT_TESTS      Set to 1 to fail (instead of skip) when the API stays overloaded
                               or out of quota (429/5xx) after the retries.

Veo is not covered here (it is slow and billed per second of video).

Run:
    python3 -m pytest intelli/test/integration/test_googleai_gemini_dev_live.py -q
"""
import base64
import io
import json
import os
import time
import unittest
import uuid
import wave

from dotenv import load_dotenv

from intelli.wrappers.googleai_wrapper import GoogleAIWrapper, GoogleAIError, GoogleAIChatSession

load_dotenv()

TEXT_MODEL = "gemini-3.8-flash"
EMBEDDING_MODEL = "gemini-embedding-001"
IMAGE_MODEL = "gemini-3.1-flash-image"
TTS_MODEL = "gemini-3.8-flash-tts"
MUSIC_MODEL = "lyria-3.5"
LIVE_MODEL = "gemini-3.8-live"

OUTPUT_DIR = os.getenv("GOOGLEAI_TEST_OUTPUT_DIR") or os.path.join("temp", "googleai_live")
STRICT = os.getenv("GOOGLEAI_STRICT_TESTS", "").strip().lower() in ("1", "true", "yes")

RETRIES = 3
RETRY_DELAY_SECONDS = 5
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
TRANSIENT_MESSAGES = ("UNAVAILABLE", "RESOURCE_EXHAUSTED", "overloaded", "1011", "1013")

AUDIO_EXTENSIONS = {"audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/wav": ".wav", "audio/x-wav": ".wav",
                    "audio/ogg": ".ogg", "audio/flac": ".flac"}


def _is_transient(error):
    if not isinstance(error, GoogleAIError):
        return False
    if error.status_code in TRANSIENT_STATUS_CODES:
        return True
    return error.status_code is None and any(text in str(error) for text in TRANSIENT_MESSAGES)


def _save(name, data):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    with open(path, "wb") as file:
        file.write(data)
    return path


class _LiveTestCase(unittest.TestCase):
    """Shared retry helper: retry 429/5xx a few times, then skip (or fail with GOOGLEAI_STRICT_TESTS=1)."""

    def call(self, function, *args, **kwargs):
        for attempt in range(1, RETRIES + 1):
            try:
                return function(*args, **kwargs)
            except GoogleAIError as error:
                if not _is_transient(error):
                    raise
                if attempt == RETRIES:
                    if STRICT:
                        raise
                    self.skipTest(f"API busy or out of quota after {RETRIES} attempts "
                                  f"(status {error.status_code}): {str(error)[:200]}")
                time.sleep(RETRY_DELAY_SECONDS * attempt)


class TestGoogleAIGeminiDeveloperLive(_LiveTestCase):

    @classmethod
    def setUpClass(cls):
        cls.api_key = os.getenv("GEMINI_API_KEY")
        if not cls.api_key:
            raise unittest.SkipTest("GEMINI_API_KEY not set")
        cls.wrapper = GoogleAIWrapper(cls.api_key)

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------
    def test_wrapper_uses_developer_api(self):
        self.assertFalse(self.wrapper.vertex)
        self.assertEqual(self.wrapper.api_version, "v1beta")
        self.assertEqual(self.wrapper._model_url(TEXT_MODEL, "generateContent"),
                         "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent")

    # ------------------------------------------------------------------
    # Text, streaming, chat
    # ------------------------------------------------------------------
    def test_generate_text_gemini_3_8_flash(self):
        text = self.call(self.wrapper.generate_text, "Reply with exactly one word: pong", model=TEXT_MODEL)
        print(f"\n[{TEXT_MODEL}] {text[:80]!r}")
        self.assertIn("pong", text.lower())

    def test_generate_text_default_model(self):
        self.assertEqual(self.wrapper.models["text"], "gemini-2.5-flash")
        text = self.call(self.wrapper.generate_text, "What is 2 + 3? Reply with the digit only.")
        print(f"\n[default text model] {text[:40]!r}")
        self.assertIn("5", text)

    def test_generate_content_plain_prompt_uses_default_model(self):
        default_model = self.wrapper.models["text"]
        response = self.call(self.wrapper.generate_content, "What is 2 + 3? Reply with the digit only.")
        print(f"\n[generate_content] modelVersion={response.get('modelVersion')}")
        self.assertIn("5", GoogleAIWrapper.extract_text(response))
        self.assertTrue(response["modelVersion"].startswith(default_model))

    def test_stream_text(self):
        chunks = self.call(lambda: list(self.wrapper.stream_text(
            "Write three short sentences about the sea. Every sentence must contain the word 'ocean'.",
            model=TEXT_MODEL)))
        joined = "".join(chunks)
        print(f"\n[stream] {len(chunks)} chunks: {joined[:80]!r}")
        self.assertGreaterEqual(len(chunks), 1)
        self.assertTrue(all(isinstance(chunk, str) and chunk for chunk in chunks))
        self.assertIn("ocean", joined.lower())
        self.assertGreater(len(joined), 40)

    def test_start_chat_two_turns(self):
        chat = self.wrapper.start_chat(model=TEXT_MODEL, system_instruction="Answer in as few words as possible.")
        self.assertIsInstance(chat, GoogleAIChatSession)

        first = self.call(chat.send_text, "My favorite color is teal. Reply with OK.")
        second = self.call(chat.send_text, "What is my favorite color? Reply with the color only.")
        print(f"\n[chat] turn1={first[:40]!r} turn2={second[:40]!r}")

        self.assertIn("teal", second.lower())
        self.assertEqual([content["role"] for content in chat.history], ["user", "model", "user", "model"])

    # ------------------------------------------------------------------
    # Structured output and token counting
    # ------------------------------------------------------------------
    def test_structured_output(self):
        schema = {
            "type": "OBJECT",
            "properties": {
                "city": {"type": "STRING"},
                "country": {"type": "STRING"},
                "population_millions": {"type": "NUMBER"},
            },
            "required": ["city", "country", "population_millions"],
        }
        response = self.call(
            self.wrapper.generate_structured_content,
            [{"text": "Give the capital city of France, its country and its approximate population in millions."}],
            schema,
            model_override=TEXT_MODEL,
        )
        data = json.loads(GoogleAIWrapper.extract_text(response))
        print(f"\n[structured] {data}")
        self.assertEqual(data["city"].lower(), "paris")
        self.assertIn("france", data["country"].lower())
        self.assertIsInstance(data["population_millions"], (int, float))

    def test_count_tokens_with_system_instruction(self):
        contents = [{"role": "user", "parts": [{"text": "Hello, how are you today?"}]}]
        plain = self.call(self.wrapper.count_tokens, {"contents": contents}, model=TEXT_MODEL)
        # The Developer API only accepts systemInstruction inside generateContentRequest;
        # count_tokens moves it there, so the request must succeed and count the extra tokens.
        with_system = self.call(
            self.wrapper.count_tokens,
            {"contents": contents,
             "system_instruction": "You are a pirate. Answer every question in rhyming pirate slang."},
            model=TEXT_MODEL,
        )
        print(f"\n[countTokens] plain={plain.get('totalTokens')} with_system={with_system.get('totalTokens')}")
        self.assertGreater(plain["totalTokens"], 0)
        self.assertGreater(with_system["totalTokens"], plain["totalTokens"])

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    def test_embed_texts(self):
        vectors = self.call(self.wrapper.embed_texts, ["Hello world", "Intelli is a Python library"],
                            model=EMBEDDING_MODEL)
        print(f"\n[embed_texts] count={len(vectors)} dims={[len(v) for v in vectors]}")
        self.assertEqual(len(vectors), 2)
        self.assertGreater(len(vectors[0]), 0)
        self.assertEqual(len(vectors[0]), len(vectors[1]))
        self.assertTrue(all(isinstance(value, float) for value in vectors[0][:10]))

    def test_embed_texts_output_dimensionality(self):
        vectors = self.call(self.wrapper.embed_texts, "Hello world", model=EMBEDDING_MODEL,
                            task_type="RETRIEVAL_QUERY", output_dimensionality=256)
        print(f"\n[embed_texts 256] dims={[len(v) for v in vectors]}")
        self.assertEqual(len(vectors), 1)
        self.assertEqual(len(vectors[0]), 256)

    def test_get_embeddings(self):
        result = self.call(self.wrapper.get_embeddings, {
            "model": EMBEDDING_MODEL,
            "content": {"parts": [{"text": "Hello world"}]},
        })
        values = result["embedding"]["values"]
        print(f"\n[get_embeddings] dims={len(values)}")
        self.assertGreater(len(values), 0)
        self.assertIsInstance(values[0], float)

    # ------------------------------------------------------------------
    # Image, speech and music generation
    # ------------------------------------------------------------------
    def test_generate_image(self):
        response = self.call(self.wrapper.generate_image,
                             "A simple flat illustration of a red apple on a white background",
                             model_override=IMAGE_MODEL)
        images = GoogleAIWrapper.extract_images(response)
        self.assertGreaterEqual(len(images), 1, "no image part in the response")

        data = base64.b64decode(images[0]["data"])
        is_png = data[:8] == b"\x89PNG\r\n\x1a\n"
        is_jpeg = data[:3] == b"\xff\xd8\xff"
        self.assertTrue(is_png or is_jpeg, f"unexpected image bytes for {images[0]['mime_type']}")
        path = _save("dev_gemini_image" + (".png" if is_png else ".jpg"), data)
        print(f"\n[image] {images[0]['mime_type']} {len(data)} bytes -> {path}")

    def _assert_speech(self, response, file_name):
        audio = GoogleAIWrapper.extract_audio(response)
        self.assertGreaterEqual(len(audio), 1, "no audio part in the response")
        wav = GoogleAIWrapper.audio_to_wav(audio[0])
        self.assertEqual(wav[:4], b"RIFF")
        with wave.open(io.BytesIO(wav)) as wav_file:
            frames, rate = wav_file.getnframes(), wav_file.getframerate()
        self.assertGreater(frames, 0)
        path = _save(file_name, wav)
        print(f"\n[tts] {response.get('modelVersion')} {audio[0]['mime_type']} "
              f"{frames / rate:.2f}s at {rate} Hz -> {path}")

    def test_generate_gemini_speech_default_model(self):
        self.assertEqual(self.wrapper.models["tts"], "gemini-2.5-flash-preview-tts")
        response = self.call(self.wrapper.generate_gemini_speech, "Hello from the Intelli test suite.")
        self._assert_speech(response, "dev_tts_default.wav")

    def test_generate_gemini_speech_3_8_flash_tts(self):
        response = self.call(self.wrapper.generate_gemini_speech, "Hello from the Intelli test suite.",
                             model_override=TTS_MODEL, voice="Puck")
        self._assert_speech(response, "dev_tts_3_8_flash.wav")

    def test_generate_music_lyria_3_5(self):
        response = self.call(self.wrapper.generate_music, "A short calm solo piano melody, slow tempo",
                             model=MUSIC_MODEL)
        audio = GoogleAIWrapper.extract_audio(response)
        self.assertGreaterEqual(len(audio), 1, "no audio part in the Lyria response")

        data = base64.b64decode(audio[0]["data"])
        self.assertGreater(len(data), 10_000)
        extension = AUDIO_EXTENSIONS.get(audio[0]["mime_type"].split(";")[0].lower(), ".bin")
        path = _save("dev_lyria_3_5" + extension, data)
        print(f"\n[music] {audio[0]['mime_type']} {len(data)} bytes -> {path} "
              f"text={GoogleAIWrapper.extract_text(response)[:60]!r}")

    # ------------------------------------------------------------------
    # Files API
    # ------------------------------------------------------------------
    def test_files_api_upload_get_use_delete(self):
        codename = f"PINEAPPLE{uuid.uuid4().hex[:6].upper()}"
        local_path = os.path.join(OUTPUT_DIR, "dev_files_api_note.txt")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(local_path, "w") as file:
            file.write(f"Project notes.\nThe project codename is {codename}.\nThe launch is planned for spring.\n")

        uploaded = self.call(self.wrapper.upload_file, local_path, display_name="intelli-live-test-note")
        file_name = uploaded["file"]["name"]
        deleted = []
        self.addCleanup(lambda: deleted or self.wrapper.delete_file(file_name))

        info = self.call(self.wrapper.get_file, file_name)
        for _ in range(10):
            if info.get("state") != "PROCESSING":
                break
            time.sleep(2)
            info = self.call(self.wrapper.get_file, file_name)
        print(f"\n[files] {file_name} state={info.get('state')} mime={info.get('mimeType')}")
        self.assertEqual(info["state"], "ACTIVE")
        self.assertTrue(info["uri"].startswith("https://"))

        answer = self.call(self.wrapper.generate_text,
                           "What is the project codename in the attached file? Reply with the codename only.",
                           model=TEXT_MODEL, media=[(info["uri"], info["mimeType"])])
        print(f"[files] answer={answer[:40]!r}")
        self.assertIn(codename, answer.upper())

        self.call(self.wrapper.delete_file, file_name)
        deleted.append(True)
        with self.assertRaises(GoogleAIError) as context:
            self.wrapper.get_file(file_name)
        self.assertIn(context.exception.status_code, (403, 404))

    # ------------------------------------------------------------------
    # Live API and models
    # ------------------------------------------------------------------
    def test_live_generate(self):
        result = self.call(self.wrapper.live_generate, "Say hello in one short sentence.", model=LIVE_MODEL)
        print(f"\n[live] messages={result['messages']} audio={len(result['audio'])} bytes "
              f"mime={result['audio_mime_type']} transcription={result['transcription'][:60]!r}")
        self.assertGreater(result["messages"], 0)
        self.assertGreater(len(result["audio"]), 0)
        self.assertTrue((result["audio_mime_type"] or "").startswith("audio/pcm"))
        self.assertTrue(result["transcription"].strip())
        _save("dev_live.wav", GoogleAIWrapper.audio_to_wav(
            {"mime_type": result["audio_mime_type"], "data": result["audio"]}))

    def test_list_models(self):
        names, page_token = [], None
        for _ in range(10):
            page = self.call(self.wrapper.list_models, page_size=100, page_token=page_token)
            names.extend(model["name"] for model in page.get("models", []))
            page_token = page.get("nextPageToken")
            if not page_token:
                break
        print(f"\n[list_models] {len(names)} models")
        for model in (TEXT_MODEL, EMBEDDING_MODEL, IMAGE_MODEL, TTS_MODEL, MUSIC_MODEL, LIVE_MODEL,
                      self.wrapper.models["text"], self.wrapper.models["tts"]):
            self.assertIn(f"models/{model}", names)


class TestGoogleAIDeveloperKeyHint(_LiveTestCase):
    """A Vertex AI key sent to the Developer API is rejected; the error must say how to fix it."""

    @classmethod
    def setUpClass(cls):
        cls.vertex_api_key = os.getenv("VERTEX_API_KEY")
        if not cls.vertex_api_key:
            raise unittest.SkipTest("VERTEX_API_KEY not set")

    def test_vertex_key_on_developer_api_raises_hint(self):
        wrapper = GoogleAIWrapper(self.vertex_api_key, vertex=False)
        with self.assertRaises(GoogleAIError) as context:
            self.call(wrapper.generate_text, "Hello", model=TEXT_MODEL)

        error = context.exception
        message = str(error)
        print(f"\n[hint] status={error.status_code} message={message[:120]!r}...")
        self.assertIn("API_KEY_SERVICE_BLOCKED", message)
        self.assertIn("hint: this key is not enabled for the Gemini Developer API", message)
        self.assertIn("vertex=True", message)
        # Do not use assertNotIn here: its failure message would print the key.
        self.assertFalse(self.vertex_api_key in message, "the API key leaked into the error message")
        self.assertFalse(self.vertex_api_key in json.dumps(error.details), "the API key leaked into details")


if __name__ == "__main__":
    unittest.main()

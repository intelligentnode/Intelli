"""
Offline unit tests for AWSWrapper and the 'aws' provider in Chatbot, the controllers and Flow.

Nothing here calls AWS: the HTTP session is a fake, credentials are made up and the AWS SDK
(botocore) is only used, when installed, to check that the wrapper builds and signs requests
exactly like the SDK does.
"""
import asyncio
import base64
import datetime
import json
import os
import struct
import unittest
import zlib
from unittest.mock import patch

import requests

from intelli.controller.remote_embed_model import RemoteEmbedModel
from intelli.controller.remote_image_model import RemoteImageModel
from intelli.controller.remote_speech_model import RemoteSpeechModel
from intelli.controller.remote_vision_model import RemoteVisionModel
from intelli.flow import Flow, Task, TextTaskInput, ToolDynamicConnector
from intelli.flow.agents.agent import Agent
from intelli.function.chatbot import Chatbot, ChatProvider
from intelli.model.input.chatbot_input import ChatModelInput
from intelli.model.input.embed_input import EmbedInput
from intelli.model.input.image_input import ImageModelInput
from intelli.model.input.text_speech_input import Text2SpeechInput
from intelli.model.input.vision_input import VisionModelInput
from intelli.wrappers.aws_wrapper import AWSError, AWSWrapper

API_KEY = "bedrock-api-key-unit-test-0123456789"
ACCESS_KEY = "AKIDEXAMPLEEXAMPLE00"
SECRET_KEY = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"
SESSION_TOKEN = "FQoGZXIvYXdzEXAMPLETOKEN//////////"
RUNTIME = "https://bedrock-runtime.us-east-1.amazonaws.com"
CLAUDE = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
CLAUDE_PATH = "us.anthropic.claude-haiku-4-5-20251001-v1%3A0"
NOVA = "us.amazon.nova-lite-v1:0"
FIXED_TIME = datetime.datetime(2026, 10, 4, 12, 0, 0, tzinfo=datetime.timezone.utc)
AWS_ENV_VARS = ("AWS_BEARER_TOKEN_BEDROCK", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
                "AWS_REGION", "AWS_DEFAULT_REGION", "AWS_PROFILE")

WEATHER_SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
WEATHER_SPEC = {"toolSpec": {"name": "get_weather", "description": "Get the weather",
                             "inputSchema": {"json": WEATHER_SCHEMA}}}


def converse_reply(text="Hello from Bedrock", stop_reason="end_turn"):
    return {"output": {"message": {"role": "assistant", "content": [{"text": text}]}},
            "stopReason": stop_reason, "usage": {"inputTokens": 3, "outputTokens": 4, "totalTokens": 7}}


def tool_reply():
    content = [{"text": "Let me check."},
               {"toolUse": {"toolUseId": "tool-1", "name": "get_weather", "input": {"city": "Paris"}}}]
    return {"output": {"message": {"role": "assistant", "content": content}}, "stopReason": "tool_use"}


def event_frame(event_type, payload, message_type="event", extra_headers=None):
    """Encode one AWS event stream message."""
    headers = {":event-type": event_type, ":content-type": "application/json", ":message-type": message_type}
    headers.update(extra_headers or {})
    header_bytes = b""
    for name, value in headers.items():
        header_bytes += bytes([len(name)]) + name.encode() + b"\x07" + struct.pack(">H", len(value)) + value.encode()
    body = json.dumps(payload).encode("utf-8")
    prelude = struct.pack(">II", 16 + len(header_bytes) + len(body), len(header_bytes))
    message = prelude + struct.pack(">I", zlib.crc32(prelude)) + header_bytes + body
    return message + struct.pack(">I", zlib.crc32(message))


def text_stream(*texts):
    frames = [event_frame("messageStart", {"role": "assistant"})]
    frames += [event_frame("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": t}}) for t in texts]
    frames.append(event_frame("messageStop", {"stopReason": "end_turn"}))
    return b"".join(frames)


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, json_data=None, status_code=200, content=None, headers=None, chunk=5):
        self.status_code = status_code
        self.headers = requests.structures.CaseInsensitiveDict(headers or {})
        self.content = content if content is not None else (
            json.dumps(json_data).encode("utf-8") if json_data is not None else b"")
        self.text = self.content.decode("utf-8", errors="replace")
        self.closed = False
        self._chunk = chunk

    def json(self):
        return json.loads(self.text)

    def iter_content(self, chunk_size=1024):
        for start in range(0, len(self.content), self._chunk):
            yield self.content[start:start + self._chunk]

    def close(self):
        self.closed = True


def error_response(status, error_type, message):
    return FakeResponse({"message": message}, status_code=status, headers={"x-amzn-ErrorType": error_type})


class FakeSession:
    """Returns the queued responses in order and records every request."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, data=None, headers=None, timeout=None, stream=False):
        self.calls.append({"method": method, "url": url, "headers": headers, "stream": stream,
                           "body": json.loads(data) if data else None})
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


class AWSTestCase(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for name in AWS_ENV_VARS:
            os.environ.pop(name, None)
        AWSWrapper._cooldowns.clear()

    def key_wrapper(self, *responses, **kwargs):
        session = FakeSession(*responses)
        return AWSWrapper(API_KEY, region="us-east-1", session=session, **kwargs), session

    def iam_wrapper(self, *responses, **kwargs):
        session = FakeSession(*responses)
        wrapper = AWSWrapper(access_key_id=ACCESS_KEY, secret_access_key=SECRET_KEY, region="us-east-1",
                             session=session, **kwargs)
        return wrapper, session


class TestAuth(AWSTestCase):
    def test_api_key_is_sent_as_bearer_token(self):
        wrapper, session = self.key_wrapper(FakeResponse(converse_reply()))
        wrapper.converse({"messages": [{"role": "user", "content": [{"text": "hi"}]}]}, CLAUDE)
        call = session.calls[0]
        self.assertEqual(call["url"], f"{RUNTIME}/model/{CLAUDE_PATH}/converse")
        self.assertEqual(call["headers"]["Authorization"], f"Bearer {API_KEY}")
        self.assertNotIn("X-Amz-Date", call["headers"])

    def test_api_key_from_environment(self):
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = API_KEY
        self.assertEqual(AWSWrapper(region="us-east-1").api_key, API_KEY)
        # explicit IAM keys mean the caller wants SigV4, so the environment key is not used
        wrapper = AWSWrapper(access_key_id=ACCESS_KEY, secret_access_key=SECRET_KEY, region="us-east-1")
        self.assertIsNone(wrapper.api_key)

    def test_sigv4_matches_the_aws_documentation_example(self):
        wrapper = AWSWrapper(access_key_id="AKIDEXAMPLE", secret_access_key=SECRET_KEY, region="us-east-1")
        when = datetime.datetime(2015, 8, 30, 12, 36, 0, tzinfo=datetime.timezone.utc)
        with patch.object(AWSWrapper, "_utcnow", staticmethod(lambda: when)):
            headers = wrapper._sigv4_headers(
                "GET", "https://iam.amazonaws.com/?Action=ListUsers&Version=2010-05-08", b"", "iam",
                extra_headers={"content-type": "application/x-www-form-urlencoded; charset=utf-8"})
        self.assertEqual(
            headers["Authorization"],
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/iam/aws4_request, "
            "SignedHeaders=content-type;host;x-amz-date, "
            "Signature=5d672d79c15b13162d9279b0855cfba6789a8edb4c82c400e06b5924a6f2b5d7")

    def test_iam_keys_sign_the_request(self):
        session = FakeSession(FakeResponse(converse_reply()))
        wrapper = AWSWrapper(access_key_id=ACCESS_KEY, secret_access_key=SECRET_KEY, session_token=SESSION_TOKEN,
                             region="eu-west-1", session=session)
        wrapper.converse({"messages": []}, NOVA)
        headers = session.calls[0]["headers"]
        self.assertTrue(session.calls[0]["url"].startswith("https://bedrock-runtime.eu-west-1.amazonaws.com/"))
        self.assertIn(f"Credential={ACCESS_KEY}/", headers["Authorization"])
        self.assertIn("/eu-west-1/bedrock/aws4_request", headers["Authorization"])
        self.assertIn("SignedHeaders=host;x-amz-date;x-amz-security-token", headers["Authorization"])
        self.assertEqual(headers["X-Amz-Security-Token"], SESSION_TOKEN)

    def test_iam_keys_from_environment(self):
        os.environ.update({"AWS_ACCESS_KEY_ID": ACCESS_KEY, "AWS_SECRET_ACCESS_KEY": SECRET_KEY,
                           "AWS_REGION": "us-west-2"})
        with patch.object(AWSWrapper, "_get_sdk_session", return_value=None):
            wrapper = AWSWrapper(session=FakeSession(FakeResponse(converse_reply())))
            wrapper.converse({"messages": []}, NOVA)
        self.assertEqual(wrapper.region, "us-west-2")
        self.assertIn(f"Credential={ACCESS_KEY}/", wrapper.session.calls[0]["headers"]["Authorization"])

    def test_sdk_credentials_object_is_used_and_refreshed_per_request(self):
        class Frozen:
            access_key, secret_key, token = ACCESS_KEY, SECRET_KEY, SESSION_TOKEN

        class SdkSession:  # the shape of a boto3 / botocore session
            region_name = "ap-southeast-2"
            frozen = 0

            def get_credentials(self):
                return self

            def get_frozen_credentials(self):
                SdkSession.frozen += 1
                return Frozen()

        session = FakeSession(FakeResponse(converse_reply()))
        wrapper = AWSWrapper(credentials=SdkSession(), session=session)
        wrapper.converse({"messages": []}, NOVA)
        wrapper.converse({"messages": []}, NOVA)
        self.assertEqual(wrapper.region, "ap-southeast-2")
        self.assertEqual(SdkSession.frozen, 2)
        self.assertEqual(session.calls[0]["headers"]["X-Amz-Security-Token"], SESSION_TOKEN)

    def test_missing_sdk_gives_an_install_hint(self):
        with patch.dict("sys.modules", {"botocore": None, "botocore.session": None}):
            wrapper = AWSWrapper(region="us-east-1", session=FakeSession(FakeResponse(converse_reply())))
            with self.assertRaises(AWSError) as raised:
                wrapper.converse({"messages": []}, NOVA)
        self.assertIn("pip install intelli[aws]", str(raised.exception))

    def test_api_key_cannot_call_polly_or_agents(self):
        wrapper, session = self.key_wrapper(FakeResponse({}))
        with patch.object(AWSWrapper, "_get_sdk_session", side_effect=AWSError("No AWS credentials were given.")):
            for call in (lambda: wrapper.synthesize_speech("hi"), lambda: wrapper.retrieve("KB1", "question")):
                with self.assertRaises(AWSError) as raised:
                    call()
                self.assertIn("a Bedrock API key cannot call", str(raised.exception))
        self.assertEqual(session.calls, [])

    def test_api_key_for_bedrock_and_iam_keys_for_other_services(self):
        session = FakeSession(FakeResponse(converse_reply()), FakeResponse(content=b"ID3audio"))
        wrapper = AWSWrapper(API_KEY, access_key_id=ACCESS_KEY, secret_access_key=SECRET_KEY, region="us-east-1",
                             session=session)
        wrapper.converse({"messages": []}, NOVA)
        wrapper.synthesize_speech("hi")
        self.assertEqual(session.calls[0]["headers"]["Authorization"], f"Bearer {API_KEY}")
        self.assertIn("/us-east-1/polly/aws4_request", session.calls[1]["headers"]["Authorization"])

    def test_from_options(self):
        wrapper = AWSWrapper.from_options(None, {"aws_region": "eu-central-1", "aws_access_key_id": ACCESS_KEY,
                                                 "secret_access_key": SECRET_KEY, "timeout": 30,
                                                 "fallback_cooldown": 60})
        self.assertEqual((wrapper.region, wrapper.timeout, wrapper.fallback_cooldown), ("eu-central-1", 30, 60))
        self.assertEqual(wrapper._static_credentials, (ACCESS_KEY, SECRET_KEY, None))
        self.assertEqual(AWSWrapper.from_options(API_KEY, {"region": "us-west-2"}).api_key, API_KEY)
        with self.assertRaises(AWSError):
            AWSWrapper(access_key_id=ACCESS_KEY)

    def test_errors_are_parsed_and_secrets_removed(self):
        wrapper, _ = self.key_wrapper(error_response(
            400, "ValidationException:http://internal.amazon.com/coral/",
            f"Invocation with on-demand throughput isn't supported (key {API_KEY})"))
        with self.assertRaises(AWSError) as raised:
            wrapper.converse({"messages": []}, "anthropic.claude-sonnet-4-6")
        error = raised.exception
        self.assertEqual((error.status_code, error.error_type), (400, "ValidationException"))
        self.assertIn("inference profile id", str(error))
        self.assertNotIn(API_KEY, str(error))

        wrapper, _ = self.iam_wrapper()
        wrapper.session.request = lambda *a, **k: (_ for _ in ()).throw(requests.exceptions.ConnectionError("down"))
        with self.assertRaises(AWSError):
            wrapper.converse({"messages": []}, NOVA)


class TestConverse(AWSTestCase):
    def test_convenience_keys_are_mapped_to_the_converse_request(self):
        wrapper, session = self.key_wrapper(FakeResponse(converse_reply()))
        openai_tool = {"type": "function", "function": {"name": "get_weather", "description": "Get the weather",
                                                        "parameters": WEATHER_SCHEMA}}
        wrapper.converse({"model": CLAUDE, "system": "Be brief.", "max_tokens": 50, "temperature": 0.2,
                          "top_p": 0.9, "stop_sequences": "END", "tools": [openai_tool], "tool_choice": "required",
                          "stream": True, "guardrailConfig": {"guardrailIdentifier": "g", "guardrailVersion": "1"},
                          "messages": [{"role": "user", "content": [{"text": "hi"}]}]})
        self.assertEqual(session.calls[0]["url"], f"{RUNTIME}/model/{CLAUDE_PATH}/converse")
        self.assertEqual(session.calls[0]["body"], {
            "messages": [{"role": "user", "content": [{"text": "hi"}]}],
            "system": [{"text": "Be brief."}],
            "inferenceConfig": {"maxTokens": 50, "temperature": 0.2, "topP": 0.9, "stopSequences": ["END"]},
            "toolConfig": {"tools": [WEATHER_SPEC], "toolChoice": {"any": {}}},
            "guardrailConfig": {"guardrailIdentifier": "g", "guardrailVersion": "1"},
        })

    def test_tool_formats(self):
        anthropic_tool = {"name": "get_weather", "description": "Get the weather", "input_schema": WEATHER_SCHEMA}
        self.assertEqual(AWSWrapper.to_tool_config([anthropic_tool]), {"tools": [WEATHER_SPEC]})
        self.assertEqual(AWSWrapper.to_tool_config([WEATHER_SPEC, {"cachePoint": {"type": "default"}}]),
                         {"tools": [WEATHER_SPEC, {"cachePoint": {"type": "default"}}]})
        no_schema = AWSWrapper.to_tool_config([{"name": "ping"}])["tools"][0]["toolSpec"]
        self.assertEqual(no_schema, {"name": "ping", "inputSchema": {"json": {"type": "object", "properties": {}}}})
        for choice, expected in (
                ("auto", {"auto": {}}), ("any", {"any": {}}), ("none", None),
                ({"type": "function", "function": {"name": "get_weather"}}, {"tool": {"name": "get_weather"}}),
                ({"type": "tool", "name": "get_weather"}, {"tool": {"name": "get_weather"}}),
                ({"type": "any"}, {"any": {}}), ({"tool": {"name": "x"}}, {"tool": {"name": "x"}})):
            self.assertEqual(AWSWrapper.to_tool_config([WEATHER_SPEC], choice).get("toolChoice"), expected)

    def test_model_ids_are_percent_encoded(self):
        arn = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.amazon.nova-lite-v1:0"
        wrapper, session = self.key_wrapper(FakeResponse(converse_reply()))
        wrapper.converse({"messages": []}, arn)
        self.assertEqual(session.calls[0]["url"], f"{RUNTIME}/model/arn%3Aaws%3Abedrock%3Aus-east-1%3A123456789012"
                                                  "%3Ainference-profile%2Fus.amazon.nova-lite-v1%3A0/converse")

    def test_default_model_follows_the_region(self):
        for region, model in (("us-east-1", "us.amazon.nova-lite-v1:0"), ("eu-west-1", "eu.amazon.nova-lite-v1:0"),
                              ("ap-southeast-2", "apac.amazon.nova-lite-v1:0")):
            session = FakeSession(FakeResponse(converse_reply()))
            wrapper = AWSWrapper(API_KEY, region=region, session=session)
            self.assertEqual(wrapper.generate_text("hi"), "Hello from Bedrock")
            self.assertEqual(wrapper.last_model, model)

    def test_extractors(self):
        response = tool_reply()
        response["output"]["message"]["content"].insert(0, {"reasoningContent": {"reasoningText": {"text": "hm"}}})
        self.assertEqual(AWSWrapper.extract_text(response), "Let me check.")
        self.assertEqual(AWSWrapper.extract_tool_calls(response), [{
            "id": "tool-1", "type": "function",
            "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris"})}}])
        self.assertEqual(AWSWrapper.extract_text({}), "")
        self.assertEqual(AWSWrapper.extract_tool_calls(converse_reply()), [])

    def test_fallback_to_the_next_model_when_throttled_or_not_granted(self):
        for status, error_type in ((429, "ThrottlingException"), (403, "AccessDeniedException"),
                                   (404, "ResourceNotFoundException"), (503, "ServiceUnavailableException")):
            wrapper, session = self.key_wrapper(error_response(status, error_type, "not now"),
                                                FakeResponse(converse_reply("from nova")))
            text = wrapper.generate_text("hi", CLAUDE, system="sys", fallback_models=[NOVA])
            self.assertEqual(text, "from nova")
            self.assertEqual(wrapper.last_model, NOVA)
            self.assertEqual([c["url"].split("/model/")[1] for c in session.calls],
                             [f"{CLAUDE_PATH}/converse", "us.amazon.nova-lite-v1%3A0/converse"])
            self.assertEqual(session.calls[0]["body"], session.calls[1]["body"])

    def test_bad_request_stops_the_fallback_chain(self):
        wrapper, session = self.key_wrapper(error_response(400, "ValidationException", "bad"),
                                            FakeResponse(converse_reply()))
        with self.assertRaises(AWSError):
            wrapper.converse({"messages": [], "fallback_models": [NOVA]}, CLAUDE)
        self.assertEqual(len(session.calls), 1)

    def test_last_fallback_error_is_raised(self):
        wrapper, session = self.key_wrapper(error_response(429, "ThrottlingException", "slow down"))
        with self.assertRaises(AWSError) as raised:
            wrapper.converse({"messages": []}, CLAUDE, fallback_models=[NOVA])
        self.assertEqual(raised.exception.status_code, 429)
        self.assertEqual(len(session.calls), 2)

    def test_fallback_cooldown_skips_a_failed_model(self):
        wrapper, session = self.key_wrapper(error_response(429, "ThrottlingException", "slow down"),
                                            FakeResponse(converse_reply()), fallback_cooldown=60)
        wrapper.converse({"messages": []}, CLAUDE, fallback_models=[NOVA])
        # a new wrapper (as Chatbot builds per call) goes straight to the fallback model
        other, other_session = self.key_wrapper(FakeResponse(converse_reply()), fallback_cooldown=60)
        other.converse({"messages": []}, CLAUDE, fallback_models=[NOVA])
        self.assertEqual(len(other_session.calls), 1)
        self.assertEqual(other.last_model, NOVA)
        # without the option nothing is remembered
        plain, plain_session = self.key_wrapper(FakeResponse(converse_reply()))
        plain.converse({"messages": []}, CLAUDE, fallback_models=[NOVA])
        self.assertEqual(plain.last_model, CLAUDE)

    def test_media_blocks_and_vision(self):
        png = b"\x89PNG fake"
        encoded = base64.b64encode(png).decode()
        self.assertEqual(AWSWrapper.media_block(png, "png"), {"image": {"format": "png", "source": {"bytes": encoded}}})
        self.assertEqual(AWSWrapper.media_block(encoded, "jpg")["image"]["format"], "jpeg")
        self.assertEqual(AWSWrapper.media_block("s3://bucket/clips/intro.mp4"),
                         {"video": {"format": "mp4", "source": {"s3Location": {"uri": "s3://bucket/clips/intro.mp4"}}}})
        document = AWSWrapper.media_block(b"%PDF", "pdf", name="Q3 report_v2.pdf")["document"]
        self.assertEqual((document["format"], document["name"]), ("pdf", "Q3 report v2 pdf"))
        with self.assertRaises(AWSError):
            AWSWrapper.media_block(b"data")

        wrapper, session = self.key_wrapper(FakeResponse(converse_reply("a cat")))
        response = wrapper.image_to_text("What is this?", encoded, "png", max_tokens=100)
        self.assertEqual(AWSWrapper.extract_text(response), "a cat")
        self.assertEqual(session.calls[0]["body"], {
            "messages": [{"role": "user", "content": [
                {"image": {"format": "png", "source": {"bytes": encoded}}}, {"text": "What is this?"}]}],
            "inferenceConfig": {"maxTokens": 100}})

    def test_count_tokens(self):
        wrapper, session = self.key_wrapper(FakeResponse({"inputTokens": 12}))
        result = wrapper.count_tokens({"system": "sys", "max_tokens": 5,
                                       "messages": [{"role": "user", "content": [{"text": "hi"}]}]}, CLAUDE)
        self.assertEqual(result, {"inputTokens": 12})
        self.assertEqual(session.calls[0]["url"], f"{RUNTIME}/model/{CLAUDE_PATH}/count-tokens")
        self.assertEqual(session.calls[0]["body"], {"input": {"converse": {
            "messages": [{"role": "user", "content": [{"text": "hi"}]}], "system": [{"text": "sys"}]}}})


class TestStreaming(AWSTestCase):
    def test_stream_text_and_events(self):
        for chunk in (1, 7, 4096):
            response = FakeResponse(content=text_stream("Hel", "lo ", "wörld"), chunk=chunk)
            wrapper, session = self.key_wrapper(response)
            self.assertEqual("".join(wrapper.stream_text("hi", CLAUDE)), "Hello wörld")
            self.assertTrue(response.closed)
            self.assertTrue(session.calls[0]["stream"])
            self.assertEqual(session.calls[0]["url"], f"{RUNTIME}/model/{CLAUDE_PATH}/converse-stream")

        wrapper, _ = self.key_wrapper(FakeResponse(content=text_stream("a")))
        events = list(wrapper.converse_stream({"messages": []}, CLAUDE))
        self.assertEqual([next(iter(e)) for e in events], ["messageStart", "contentBlockDelta", "messageStop"])
        self.assertEqual(events[-1], {"messageStop": {"stopReason": "end_turn"}})

    def test_stream_exception_event_raises(self):
        content = text_stream("a")[:0] + event_frame("contentBlockDelta", {"delta": {"text": "a"}}) + event_frame(
            "throttlingException", {"message": "Too many requests"}, message_type="exception",
            extra_headers={":exception-type": "throttlingException"})
        wrapper, _ = self.key_wrapper(FakeResponse(content=content))
        stream = wrapper.stream_text("hi", CLAUDE)
        self.assertEqual(next(stream), "a")
        with self.assertRaises(AWSError) as raised:
            next(stream)
        self.assertEqual(raised.exception.error_type, "throttlingException")

    def test_corrupt_or_cut_stream_raises(self):
        good = text_stream("abc")
        corrupt = bytearray(good)
        corrupt[-6] ^= 0xFF
        for content in (bytes(corrupt), good[:-3]):
            wrapper, _ = self.key_wrapper(FakeResponse(content=content))
            with self.assertRaises(AWSError):
                list(wrapper.converse_stream({"messages": []}, CLAUDE))

    def test_http_error_before_the_stream(self):
        wrapper, _ = self.key_wrapper(error_response(429, "ThrottlingException", "slow down"))
        with self.assertRaises(AWSError) as raised:
            list(wrapper.stream_text("hi", CLAUDE))
        self.assertEqual(raised.exception.status_code, 429)

    def test_invoke_model_stream_decodes_chunks(self):
        chunks = [{"type": "content_block_delta", "delta": {"text": "Hi"}}, {"type": "message_stop"}]
        content = b"".join(event_frame("chunk", {"bytes": base64.b64encode(json.dumps(c).encode()).decode()})
                           for c in chunks)
        wrapper, session = self.key_wrapper(FakeResponse(content=content))
        self.assertEqual(list(wrapper.invoke_model_stream(CLAUDE, {"messages": []})), chunks)
        self.assertTrue(session.calls[0]["url"].endswith("/invoke-with-response-stream"))


class TestModels(AWSTestCase):
    def test_titan_embeddings_one_request_per_text(self):
        wrapper, session = self.key_wrapper(FakeResponse({"embedding": [0.1, 0.2], "inputTextTokenCount": 3}))
        result = wrapper.get_embeddings({"texts": ["a", "b"], "dimensions": 256})
        self.assertEqual(result, {"embeddings": [[0.1, 0.2], [0.1, 0.2]], "model": "amazon.titan-embed-text-v2:0",
                                  "input_tokens": 6})
        self.assertEqual([c["body"] for c in session.calls],
                         [{"inputText": "a", "dimensions": 256}, {"inputText": "b", "dimensions": 256}])
        self.assertEqual(session.calls[0]["url"], f"{RUNTIME}/model/amazon.titan-embed-text-v2%3A0/invoke")

    def test_cohere_and_nova_embeddings(self):
        wrapper, session = self.key_wrapper(FakeResponse({"embeddings": {"float": [[1.0], [2.0]]}}))
        result = wrapper.get_embeddings({"texts": ["a", "b"], "model": "cohere.embed-v4:0"})
        self.assertEqual(result["embeddings"], [[1.0], [2.0]])
        self.assertEqual(session.calls[0]["body"], {"texts": ["a", "b"], "input_type": "search_document"})

        wrapper, session = self.key_wrapper(FakeResponse({"embeddings": [{"embeddingType": "TEXT",
                                                                          "embedding": [3.0]}]}))
        result = wrapper.get_embeddings({"texts": "a", "model": "amazon.nova-2-multimodal-embeddings-v1:0",
                                         "dimensions": 1024})
        self.assertEqual(result["embeddings"], [[3.0]])
        self.assertEqual(session.calls[0]["body"], {"taskType": "SINGLE_EMBEDDING", "singleEmbeddingParams": {
            "embeddingPurpose": "GENERIC_INDEX", "embeddingDimension": 1024,
            "text": {"truncationMode": "END", "value": "a"}}})
        with self.assertRaises(AWSError):
            wrapper.get_embeddings({"texts": []})

    def test_generate_image(self):
        wrapper, session = self.key_wrapper(FakeResponse({"images": ["aW1n"]}))
        response = wrapper.generate_image("a red fox", negative_prompt="blurry", number_of_images=1, width=1024,
                                          height=1024, quality="premium")
        self.assertEqual(AWSWrapper.extract_images(response), ["aW1n"])
        self.assertEqual(session.calls[0]["url"], f"{RUNTIME}/model/amazon.nova-canvas-v1%3A0/invoke")
        self.assertEqual(session.calls[0]["body"], {
            "taskType": "TEXT_IMAGE", "textToImageParams": {"text": "a red fox", "negativeText": "blurry"},
            "imageGenerationConfig": {"numberOfImages": 1, "width": 1024, "height": 1024, "quality": "premium"}})

        wrapper.generate_image("a red fox", "stability.sd3-5-large-v1:0", seed=7, params={"aspect_ratio": "16:9"})
        self.assertEqual(session.calls[1]["body"], {"prompt": "a red fox", "seed": 7, "aspect_ratio": "16:9"})
        self.assertEqual(AWSWrapper.extract_images({"artifacts": [{"base64": "b2xk"}]}), ["b2xk"])

    def test_video_job(self):
        arn = "arn:aws:bedrock:us-east-1:123456789012:async-invoke/abc123"
        wrapper, session = self.key_wrapper(FakeResponse({"invocationArn": arn}),
                                            FakeResponse({"status": "InProgress"}),
                                            FakeResponse({"status": "Completed", "invocationArn": arn}))
        job = wrapper.generate_video("a cat on a beach", "s3://bucket/videos", seed=3)
        self.assertEqual(session.calls[0]["body"], {
            "modelId": "amazon.nova-reel-v1:1",
            "modelInput": {"taskType": "TEXT_VIDEO", "textToVideoParams": {"text": "a cat on a beach"},
                           "videoGenerationConfig": {"durationSeconds": 6, "fps": 24, "dimension": "1280x720",
                                                     "seed": 3}},
            "outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://bucket/videos"}}})
        with patch("intelli.wrappers.aws_wrapper.time.sleep"):
            self.assertEqual(wrapper.wait_for_async_invoke(job["invocationArn"])["status"], "Completed")
        self.assertEqual(session.calls[1]["method"], "GET")
        self.assertTrue(session.calls[1]["url"].endswith("/async-invoke/arn%3Aaws%3Abedrock%3Aus-east-1"
                                                         "%3A123456789012%3Aasync-invoke%2Fabc123"))

        wrapper, _ = self.key_wrapper(FakeResponse({"status": "Failed", "failureMessage": "blocked"}))
        with self.assertRaises(AWSError):
            wrapper.wait_for_async_invoke(arn)

    def test_guardrail_and_model_listing(self):
        wrapper, session = self.key_wrapper(FakeResponse({"action": "NONE"}))
        self.assertEqual(wrapper.apply_guardrail("gr-1", "hello", version="2", source="OUTPUT"), {"action": "NONE"})
        self.assertEqual(session.calls[0]["url"], f"{RUNTIME}/guardrail/gr-1/version/2/apply")
        self.assertEqual(session.calls[0]["body"], {"source": "OUTPUT", "content": [{"text": {"text": "hello"}}]})

        wrapper.list_foundation_models(provider="Amazon", output_modality="TEXT")
        wrapper.list_inference_profiles("SYSTEM_DEFINED", 5)
        self.assertEqual(session.calls[1]["url"], "https://bedrock.us-east-1.amazonaws.com/foundation-models"
                                                  "?byProvider=Amazon&byOutputModality=TEXT")
        self.assertEqual(session.calls[2]["url"], "https://bedrock.us-east-1.amazonaws.com/inference-profiles"
                                                  "?type=SYSTEM_DEFINED&maxResults=5")
        self.assertEqual(session.calls[1]["headers"]["Authorization"], f"Bearer {API_KEY}")


class TestKnowledgeBasesAgentsAndSpeech(AWSTestCase):
    def test_retrieve_and_generate(self):
        results = {"retrievalResults": [{"content": {"text": " first "}, "score": 0.9}, {"content": {"text": "second"}}]}
        wrapper, session = self.iam_wrapper(FakeResponse(results), FakeResponse({"output": {"text": "answer"}}))
        response = wrapper.retrieve("KB12345678", "what is intelli?", 2, search_type="HYBRID")
        self.assertEqual(AWSWrapper.retrieval_to_text(response), "first\n\nsecond")
        self.assertEqual(session.calls[0]["url"],
                         "https://bedrock-agent-runtime.us-east-1.amazonaws.com/knowledgebases/KB12345678/retrieve")
        self.assertEqual(session.calls[0]["body"], {
            "retrievalQuery": {"text": "what is intelli?"},
            "retrievalConfiguration": {"vectorSearchConfiguration": {"numberOfResults": 2,
                                                                      "overrideSearchType": "HYBRID"}}})
        self.assertIn("/us-east-1/bedrock/aws4_request", session.calls[0]["headers"]["Authorization"])

        answer = wrapper.retrieve_and_generate("q", "KB12345678", "arn:model", session_id="s1")
        self.assertEqual(answer["output"]["text"], "answer")
        self.assertEqual(session.calls[1]["body"], {
            "input": {"text": "q"}, "sessionId": "s1",
            "retrieveAndGenerateConfiguration": {"type": "KNOWLEDGE_BASE", "knowledgeBaseConfiguration": {
                "knowledgeBaseId": "KB12345678", "modelArn": "arn:model"}}})

    def test_invoke_agent(self):
        def chunk(text):
            return event_frame("chunk", {"bytes": base64.b64encode(text.encode()).decode()})

        content = chunk("The answer ") + event_frame("trace", {"trace": {"orchestrationTrace": {}}}) + chunk("is 42.")
        wrapper, session = self.iam_wrapper(FakeResponse(content=content))
        result = wrapper.invoke_agent("AGENT12345", "ALIAS12345", "question", "session-1", enable_trace=True)
        self.assertEqual((result["text"], result["session_id"]), ("The answer is 42.", "session-1"))
        self.assertEqual(result["events"], [{"trace": {"trace": {"orchestrationTrace": {}}}}])
        self.assertEqual(session.calls[0]["url"], "https://bedrock-agent-runtime.us-east-1.amazonaws.com/agents/"
                                                  "AGENT12345/agentAliases/ALIAS12345/sessions/session-1/text")
        self.assertEqual(session.calls[0]["body"], {"inputText": "question", "enableTrace": True})

        wrapper, _ = self.iam_wrapper(FakeResponse(content=content))
        self.assertEqual(list(wrapper.stream_agent("AGENT12345", "ALIAS12345", "question")),
                         ["The answer ", "is 42."])

    def test_invoke_agentcore_runtime(self):
        arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/my_agent-abc"
        session_id = "session-" + "0" * 32
        wrapper, session = self.iam_wrapper(FakeResponse({"result": "done"}))
        self.assertEqual(wrapper.invoke_agent_runtime(arn, {"prompt": "hi"}, session_id=session_id,
                                                      qualifier="DEFAULT"), {"result": "done"})
        call = session.calls[0]
        self.assertEqual(call["url"], "https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/arn%3Aaws%3Abedrock"
                                      "-agentcore%3Aus-east-1%3A123456789012%3Aruntime%2Fmy_agent-abc/invocations"
                                      "?qualifier=DEFAULT")
        self.assertEqual(call["body"], {"prompt": "hi"})
        self.assertEqual(call["headers"]["X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"], session_id)
        self.assertIn("/us-east-1/bedrock-agentcore/aws4_request", call["headers"]["Authorization"])

        # OAuth (JWT authorizer) runtimes take a bearer token; SSE answers come back as a list
        sse = FakeResponse(content=b'data: {"delta": "a"}\n\ndata: plain\n\n',
                           headers={"Content-Type": "text/event-stream"})
        wrapper, session = self.key_wrapper(sse)
        self.assertEqual(wrapper.invoke_agent_runtime(arn, {"prompt": "hi"}, access_token="jwt-token"), [{"delta": "a"}, "plain"])
        self.assertEqual(session.calls[0]["headers"]["Authorization"], "Bearer jwt-token")

    def test_polly(self):
        wrapper, session = self.iam_wrapper(FakeResponse(content=b"ID3audio"), FakeResponse({"Voices": []}))
        audio = wrapper.synthesize_speech("Hello", "Matthew", engine="generative", language_code="en-US",
                                          sample_rate=24000)
        self.assertEqual(audio, b"ID3audio")
        self.assertEqual(session.calls[0]["url"], "https://polly.us-east-1.amazonaws.com/v1/speech")
        self.assertEqual(session.calls[0]["body"], {"Text": "Hello", "VoiceId": "Matthew", "Engine": "generative",
                                                    "OutputFormat": "mp3", "LanguageCode": "en-US",
                                                    "SampleRate": "24000"})
        wrapper.list_voices("neural", "en-US")
        self.assertEqual(session.calls[1]["url"],
                         "https://polly.us-east-1.amazonaws.com/v1/voices?Engine=neural&LanguageCode=en-US")


class TestMatchesTheAwsSdk(AWSTestCase):
    """The wrapper must put the same request on the wire as botocore, with the same signature."""

    def setUp(self):
        super().setUp()
        try:
            import botocore.session  # noqa: F401
        except ImportError:
            self.skipTest("botocore is not installed")

    def sdk_request(self, service, operation, params):
        import botocore.session
        from botocore.awsrequest import AWSResponse

        client = botocore.session.Session().create_client(
            service, region_name="us-east-1", aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY,
            aws_session_token=SESSION_TOKEN)
        captured = {}

        class Raw:
            def read(self, *args, **kwargs):
                return b"{}"

            def stream(self, *args, **kwargs):
                return iter([b"{}"])

        def before_send(request, **kwargs):
            captured.update(method=request.method, url=request.url, body=request.body)
            return AWSResponse(request.url, 200, {}, Raw())

        client.meta.events.register("before-send", before_send)
        try:
            getattr(client, operation)(**params)
        except Exception:
            pass  # the empty fake reply may not parse; the request is already captured
        return captured

    def sdk_signature(self, method, url, body, signing_name):
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest
        from botocore.credentials import Credentials

        request = AWSRequest(method=method, url=url, data=body or None)
        with patch("botocore.auth.datetime") as fake:
            fake.datetime.utcnow.return_value = FIXED_TIME.replace(tzinfo=None)
            fake.datetime.now.return_value = FIXED_TIME
            SigV4Auth(Credentials(ACCESS_KEY, SECRET_KEY, SESSION_TOKEN), signing_name, "us-east-1").add_auth(request)
        return request.headers["Authorization"]

    def assert_same_request(self, service, operation, sdk_params, call, signing_name):
        sent = []

        class Recorder:
            def request(self, method, url, data=None, headers=None, timeout=None, stream=False):
                sent.append({"method": method, "url": url, "data": data, "headers": headers})
                return FakeResponse(content=b"") if stream else FakeResponse({})

        wrapper = AWSWrapper(access_key_id=ACCESS_KEY, secret_access_key=SECRET_KEY, session_token=SESSION_TOKEN,
                             region="us-east-1", session=Recorder())
        with patch.object(AWSWrapper, "_utcnow", staticmethod(lambda: FIXED_TIME)):
            result = call(wrapper)
            if hasattr(result, "__next__"):
                list(result)
        mine, sdk = sent[0], self.sdk_request(service, operation, sdk_params)
        self.assertEqual((mine["method"], mine["url"]), (sdk["method"], sdk["url"]))
        if sdk["body"]:
            expected = json.loads(sdk["body"])
            expected.pop("clientRequestToken", None)  # optional idempotency token the SDK adds
            self.assertEqual(json.loads(mine["data"]), expected)
        else:
            self.assertFalse(mine["data"])
        self.assertEqual(mine["headers"]["Authorization"],
                         self.sdk_signature(mine["method"], mine["url"], mine["data"], signing_name))

    def test_bedrock_runtime_requests(self):
        arn = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.amazon.nova-lite-v1:0"
        messages = [{"role": "user", "content": [{"text": "hi é"}]}]
        tool = {"type": "function", "function": {"name": "get_weather", "description": "Get the weather",
                                                 "parameters": WEATHER_SCHEMA}}
        self.assert_same_request(
            "bedrock-runtime", "converse",
            dict(modelId=CLAUDE, messages=messages, system=[{"text": "sys"}],
                 inferenceConfig={"maxTokens": 50, "temperature": 0.5},
                 toolConfig={"tools": [WEATHER_SPEC], "toolChoice": {"any": {}}}),
            lambda w: w.converse({"model": CLAUDE, "messages": messages, "system": "sys", "max_tokens": 50,
                                  "temperature": 0.5, "tools": [tool], "tool_choice": "required"}), "bedrock")
        self.assert_same_request("bedrock-runtime", "converse", dict(modelId=arn, messages=messages),
                                 lambda w: w.converse({"messages": messages}, arn), "bedrock")
        self.assert_same_request("bedrock-runtime", "converse_stream", dict(modelId=CLAUDE, messages=messages),
                                 lambda w: w.converse_stream({"messages": messages}, CLAUDE), "bedrock")
        self.assert_same_request(
            "bedrock-runtime", "invoke_model",
            dict(modelId="amazon.titan-embed-text-v2:0", body=json.dumps({"inputText": "hello"}),
                 accept="application/json", contentType="application/json"),
            lambda w: w.invoke_model("amazon.titan-embed-text-v2:0", {"inputText": "hello"}), "bedrock")
        self.assert_same_request(
            "bedrock-runtime", "start_async_invoke",
            dict(modelId="amazon.nova-reel-v1:1", modelInput={"taskType": "TEXT_VIDEO"},
                 outputDataConfig={"s3OutputDataConfig": {"s3Uri": "s3://bucket/out"}}),
            lambda w: w.start_async_invoke("amazon.nova-reel-v1:1", {"taskType": "TEXT_VIDEO"}, "s3://bucket/out"),
            "bedrock")
        job = "arn:aws:bedrock:us-east-1:123456789012:async-invoke/abc123def456"
        self.assert_same_request("bedrock-runtime", "get_async_invoke", dict(invocationArn=job),
                                 lambda w: w.get_async_invoke(job), "bedrock")
        self.assert_same_request(
            "bedrock-runtime", "apply_guardrail",
            dict(guardrailIdentifier="gr-123", guardrailVersion="DRAFT", source="INPUT",
                 content=[{"text": {"text": "hello"}}]),
            lambda w: w.apply_guardrail("gr-123", "hello"), "bedrock")

    def test_bedrock_control_plane_and_agent_requests(self):
        self.assert_same_request("bedrock", "list_foundation_models", dict(byProvider="Amazon", byOutputModality="TEXT"),
                                 lambda w: w.list_foundation_models("Amazon", "TEXT"), "bedrock")
        self.assert_same_request("bedrock", "list_inference_profiles", dict(typeEquals="SYSTEM_DEFINED", maxResults=5),
                                 lambda w: w.list_inference_profiles("SYSTEM_DEFINED", 5), "bedrock")
        self.assert_same_request(
            "bedrock-agent-runtime", "retrieve",
            dict(knowledgeBaseId="KB12345678", retrievalQuery={"text": "q"},
                 retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 3}}),
            lambda w: w.retrieve("KB12345678", "q", 3), "bedrock")
        self.assert_same_request(
            "bedrock-agent-runtime", "retrieve_and_generate",
            dict(input={"text": "q"}, retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {"knowledgeBaseId": "KB12345678", "modelArn": "arn:model"}}),
            lambda w: w.retrieve_and_generate("q", "KB12345678", "arn:model"), "bedrock")
        self.assert_same_request(
            "bedrock-agent-runtime", "invoke_agent",
            dict(agentId="AGENT12345", agentAliasId="ALIAS12345", sessionId="session-1", inputText="hello"),
            lambda w: w.stream_agent("AGENT12345", "ALIAS12345", "hello", "session-1"), "bedrock")

    def test_polly_requests(self):
        self.assert_same_request("polly", "synthesize_speech",
                                 dict(Text="hello", VoiceId="Joanna", Engine="neural", OutputFormat="mp3"),
                                 lambda w: w.synthesize_speech("hello"), "polly")
        self.assert_same_request("polly", "describe_voices", dict(Engine="neural", LanguageCode="en-US"),
                                 lambda w: w.list_voices("neural", "en-US"), "polly")

    def test_event_stream_decoding_matches_the_sdk(self):
        from botocore.eventstream import EventStreamBuffer

        stream = text_stream("Hel", "lo ", "wörld")
        buffer = EventStreamBuffer()
        buffer.add_data(stream)
        expected = [(message.headers[":event-type"], json.loads(message.payload)) for message in buffer]
        for chunk in (1, 7, 4096):
            decoded = [(headers[":event-type"], json.loads(payload)) for headers, payload in
                       AWSWrapper._iter_event_stream(FakeResponse(content=stream, chunk=chunk))]
            self.assertEqual(decoded, expected)


class HttpPatch:
    """Patch requests.Session.request for code that builds its own AWSWrapper (Chatbot, controllers, Flow)."""

    def __init__(self, test, handler):
        self.calls = []

        def fake(session, method, url, data=None, headers=None, timeout=None, stream=False):
            call = {"method": method, "url": url, "headers": headers,
                    "body": json.loads(data) if data else None}
            self.calls.append(call)
            return handler(call)

        patcher = patch.object(requests.Session, "request", new=fake)
        patcher.start()
        test.addCleanup(patcher.stop)


class TestChatbotAndControllers(AWSTestCase):
    def test_chat_input(self):
        chat_input = ChatModelInput("Be brief.", model=CLAUDE, max_tokens=64, temperature=0.3,
                                    fallback_models=[NOVA])
        chat_input.add_user_message("hi")
        chat_input.add_assistant_message("hello")
        self.assertEqual(chat_input.get_aws_input(), {
            "model": CLAUDE, "system": [{"text": "Be brief."}], "temperature": 0.3, "max_tokens": 64,
            "fallback_models": [NOVA],
            "messages": [{"role": "user", "content": [{"text": "hi"}]},
                         {"role": "assistant", "content": [{"text": "hello"}]}]})
        # Claude models without sampling parameters must not get a temperature on Bedrock either
        self.assertNotIn("temperature", ChatModelInput("s", model="us.anthropic.claude-opus-4-7").get_aws_input())
        self.assertNotIn("temperature", ChatModelInput("s", model="global.anthropic.claude-sonnet-5").get_aws_input())

    def test_chatbot_chat(self):
        http = HttpPatch(self, lambda call: FakeResponse(converse_reply("4")))
        bot = Chatbot(API_KEY, ChatProvider.AWS, {"region": "eu-west-1"})
        chat_input = ChatModelInput("You are a calculator.", model="eu.amazon.nova-lite-v1:0", max_tokens=20)
        chat_input.add_user_message("2+2?")
        self.assertEqual(bot.chat(chat_input), ["4"])
        call = http.calls[0]
        self.assertEqual(call["url"], "https://bedrock-runtime.eu-west-1.amazonaws.com/model/"
                                      "eu.amazon.nova-lite-v1%3A0/converse")
        self.assertEqual(call["headers"]["Authorization"], f"Bearer {API_KEY}")
        self.assertEqual(call["body"], {
            "messages": [{"role": "user", "content": [{"text": "2+2?"}]}],
            "system": [{"text": "You are a calculator."}],
            "inferenceConfig": {"temperature": 1, "maxTokens": 20}})

    def test_chatbot_with_iam_keys_and_no_api_key(self):
        http = HttpPatch(self, lambda call: FakeResponse(converse_reply()))
        bot = Chatbot(None, "aws", {"aws_access_key_id": ACCESS_KEY, "aws_secret_access_key": SECRET_KEY,
                                    "aws_region": "us-east-1"})
        chat_input = ChatModelInput("s", model=NOVA)
        chat_input.add_user_message("hi")
        self.assertEqual(bot.chat(chat_input), ["Hello from Bedrock"])
        self.assertTrue(http.calls[0]["headers"]["Authorization"].startswith("AWS4-HMAC-SHA256 "))

    def test_chatbot_fallback_models(self):
        def handler(call):
            if CLAUDE_PATH in call["url"]:
                return error_response(429, "ThrottlingException", "Too many tokens per day")
            return FakeResponse(converse_reply("from nova"))

        for model_params, options in (({"fallback_models": [NOVA]}, {}), ({}, {"fallback_models": [NOVA]})):
            http = HttpPatch(self, handler)
            bot = Chatbot(API_KEY, "aws", {"region": "us-east-1", **options})
            chat_input = ChatModelInput("s", model=CLAUDE, **model_params)
            chat_input.add_user_message("hi")
            self.assertEqual(bot.chat(chat_input), ["from nova"])
            self.assertEqual(len(http.calls), 2)
            self.assertNotIn("fallback_models", http.calls[1]["body"])

    def test_chatbot_tool_call_and_stream(self):
        HttpPatch(self, lambda call: FakeResponse(tool_reply()))
        bot = Chatbot(API_KEY, "aws", {"region": "us-east-1"})
        chat_input = ChatModelInput("s", model=CLAUDE, tools=[WEATHER_SPEC])
        chat_input.add_user_message("weather in Paris?")
        result = bot.chat(chat_input)[0]
        self.assertEqual(result["type"], "tool_response")
        self.assertEqual(result["tool_calls"][0]["function"],
                         {"name": "get_weather", "arguments": json.dumps({"city": "Paris"})})

        HttpPatch(self, lambda call: FakeResponse(content=text_stream("a", "b", "c")))
        self.assertEqual(list(bot.stream(chat_input)), ["a", "b", "c"])

    def test_embed_image_vision_and_speech_controllers(self):
        http = HttpPatch(self, lambda call: FakeResponse({"embedding": [0.5], "inputTextTokenCount": 1}))
        embed_input = EmbedInput(["hello"])
        embed_input.set_default_values("aws")
        result = RemoteEmbedModel(API_KEY, "aws", {"region": "us-east-1"}).get_embeddings(embed_input)
        self.assertEqual(result["embeddings"], [[0.5]])
        self.assertIn("/model/amazon.titan-embed-text-v2%3A0/invoke", http.calls[0]["url"])

        http = HttpPatch(self, lambda call: FakeResponse({"images": ["aW1n"]}))
        image_input = ImageModelInput("a red fox", width=1024, height=1024, quality="hd")
        self.assertEqual(RemoteImageModel(API_KEY, "aws").generate_images(image_input), ["aW1n"])
        self.assertEqual(http.calls[0]["body"], {
            "taskType": "TEXT_IMAGE", "textToImageParams": {"text": "a red fox"},
            "imageGenerationConfig": {"numberOfImages": 1, "width": 1024, "height": 1024}})

        http = HttpPatch(self, lambda call: FakeResponse(converse_reply("a cat")))
        vision_input = VisionModelInput("What is this?", image_data="aW1n", extension="png", model=CLAUDE)
        self.assertEqual(RemoteVisionModel(API_KEY, "aws").image_to_text(vision_input), "a cat")
        self.assertIn(f"/model/{CLAUDE_PATH}/converse", http.calls[0]["url"])
        self.assertEqual(http.calls[0]["body"]["messages"][0]["content"][0],
                         {"image": {"format": "png", "source": {"bytes": "aW1n"}}})

        http = HttpPatch(self, lambda call: FakeResponse(content=b"ID3audio"))
        options = {"access_key_id": ACCESS_KEY, "secret_access_key": SECRET_KEY, "region": "us-east-1"}
        speech = RemoteSpeechModel(None, "aws", options)
        # the flow defaults (OpenAI voice 'alloy', model 'tts-1') map to the Polly defaults
        self.assertEqual(speech.generate_speech(Text2SpeechInput("Hello", gender="MALE")), b"ID3audio")
        self.assertEqual(http.calls[0]["body"], {"Text": "Hello", "VoiceId": "Matthew", "Engine": "neural",
                                                 "OutputFormat": "mp3"})
        speech.generate_speech(Text2SpeechInput("Hello", voice="Ruth", model="generative"))
        self.assertEqual((http.calls[1]["body"]["VoiceId"], http.calls[1]["body"]["Engine"]), ("Ruth", "generative"))


class TestFlow(AWSTestCase):
    def test_text_flow_with_fallback_and_no_api_key(self):
        def handler(call):
            if CLAUDE_PATH in call["url"]:
                return error_response(403, "AccessDeniedException", "Model use case details have not been submitted")
            text = call["body"]["messages"][0]["content"][0]["text"]
            return FakeResponse(converse_reply("summary" if "Summarize" in call["body"]["system"][0]["text"]
                                               else f"draft about: {text}"))

        http = HttpPatch(self, handler)
        options = {"access_key_id": ACCESS_KEY, "secret_access_key": SECRET_KEY, "region": "us-east-1"}
        writer = Agent("text", "aws", "Write a short match briefing.",
                       {"model": CLAUDE, "fallback_models": [NOVA], "max_tokens": 400}, options)
        editor = Agent("text", "aws", "Summarize in one line.", {"model": NOVA}, options)
        tasks = {"write": Task(TextTaskInput("Brazil vs France"), writer),
                 "edit": Task(TextTaskInput("Summarize the briefing"), editor)}
        flow = Flow(tasks=tasks, map_paths={"write": ["edit"]})
        output = asyncio.run(flow.start())

        self.assertEqual(flow.errors, {})
        self.assertEqual(output["edit"]["output"], "summary")
        self.assertEqual([c["url"].split("/model/")[1] for c in http.calls],
                         [f"{CLAUDE_PATH}/converse", "us.amazon.nova-lite-v1%3A0/converse",
                          "us.amazon.nova-lite-v1%3A0/converse"])
        self.assertEqual(http.calls[1]["body"]["inferenceConfig"], {"temperature": 1, "maxTokens": 400})
        self.assertIn("draft about: Brazil vs France", http.calls[2]["body"]["messages"][0]["content"][0]["text"])

    def test_tool_routing_flow(self):
        def handler(call):
            return FakeResponse(tool_reply() if "toolConfig" in call["body"] else converse_reply("done"))

        HttpPatch(self, handler)
        model_params = {"key": API_KEY, "model": CLAUDE}
        tool = {"name": "get_weather", "description": "Get the weather", "input_schema": WEATHER_SCHEMA}
        tasks = {
            "llm": Task(TextTaskInput("Weather in Paris?"), Agent("text", "aws", "Use tools when needed.",
                                                                 {**model_params, "tools": [tool]})),
            "tool": Task(TextTaskInput("Run the tool"), Agent("text", "aws", "Tool step", model_params)),
            "direct": Task(TextTaskInput("Answer directly"), Agent("text", "aws", "Direct step", model_params)),
        }
        connector = ToolDynamicConnector(destinations={"tool_called": "tool", "no_tool": "direct"})
        flow = Flow(tasks=tasks, map_paths={}, dynamic_connectors={"llm": connector})
        output = asyncio.run(flow.start())
        self.assertEqual(flow.errors, {})
        self.assertIn("tool", output)
        self.assertNotIn("direct", output)

    def test_knowledge_base_search_agent(self):
        http = HttpPatch(self, lambda call: FakeResponse({"retrievalResults": [{"content": {"text": "Intelli docs"}}]}))
        options = {"access_key_id": ACCESS_KEY, "secret_access_key": SECRET_KEY, "region": "us-east-1"}
        agent = Agent("search", "aws", "search", {"knowledge_base_id": "KB12345678", "k": 2}, options)
        flow = Flow(tasks={"search": Task(TextTaskInput("What is Intelli?"), agent)}, map_paths={})
        output = asyncio.run(flow.start())
        self.assertEqual(output["search"]["output"], "Intelli docs")
        self.assertTrue(http.calls[0]["url"].endswith("/knowledgebases/KB12345678/retrieve"))
        self.assertEqual(http.calls[0]["body"]["retrievalConfiguration"],
                         {"vectorSearchConfiguration": {"numberOfResults": 2}})


if __name__ == "__main__":
    unittest.main()

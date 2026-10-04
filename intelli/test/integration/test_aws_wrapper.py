"""
Live tests for AWSWrapper and the 'aws' provider (Amazon Bedrock and Amazon Polly).

The tests are skipped unless AWS_LIVE_TESTS=1 is set, so a machine that happens to have AWS
credentials configured is never billed by accident.

Environment:
    AWS_LIVE_TESTS             Set to 1 to run these tests.
    AWS_BEARER_TOKEN_BEDROCK   Bedrock API key, or
    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (/ AWS_SESSION_TOKEN)   IAM keys, or
    AWS_PROFILE                a profile from ~/.aws (needs pip install intelli[aws]).
    AWS_REGION                 Region (default us-east-1).
    AWS_CHAT_MODEL             Chat model id (default: the Amazon Nova Lite inference profile of the region).
    AWS_TOOL_MODEL             Model for the tool-call test (default AWS_CHAT_MODEL).
    AWS_IMAGE_TESTS            Set to 1 to also generate one image with Amazon Nova Canvas (billed per image).
    AWS_KNOWLEDGE_BASE_ID      Bedrock Knowledge Base id, to run the retrieval test (IAM credentials).

The Polly test runs only with IAM credentials (a Bedrock API key cannot call Polly).
Video generation, Bedrock Agents and AgentCore are not covered (they need resources in the account).

Run:
    AWS_LIVE_TESTS=1 python3 -m pytest intelli/test/integration/test_aws_wrapper.py -q
"""
import asyncio
import base64
import os
import struct
import unittest
import zlib

from dotenv import load_dotenv

from intelli.controller.remote_embed_model import RemoteEmbedModel
from intelli.controller.remote_vision_model import RemoteVisionModel
from intelli.flow import Flow, Task, TextTaskInput
from intelli.flow.agents.agent import Agent
from intelli.function.chatbot import Chatbot, ChatProvider
from intelli.model.input.chatbot_input import ChatModelInput
from intelli.model.input.embed_input import EmbedInput
from intelli.model.input.vision_input import VisionModelInput
from intelli.wrappers.aws_wrapper import AWSError, AWSWrapper

load_dotenv()

LIVE = os.getenv("AWS_LIVE_TESTS", "").strip().lower() in ("1", "true", "yes")
REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
API_KEY = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
OPTIONS = {"region": REGION, **({"profile": os.environ["AWS_PROFILE"]} if os.getenv("AWS_PROFILE") else {})}

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather of a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}


def _png(width, height, rgb):
    """A plain one-color PNG, so the vision test needs no image file."""
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    rows = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


@unittest.skipUnless(LIVE, "Set AWS_LIVE_TESTS=1 to run the live AWS tests")
class TestAWSWrapperLive(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.wrapper = AWSWrapper.from_options(API_KEY, OPTIONS)
        cls.chat_model = os.getenv("AWS_CHAT_MODEL") or cls.wrapper._default_model("chat")
        cls.tool_model = os.getenv("AWS_TOOL_MODEL") or cls.chat_model

    def iam_wrapper(self):
        """The wrapper, or skip when only a Bedrock API key is configured (no IAM credentials for SigV4)."""
        try:
            self.wrapper._resolve_credentials()
        except AWSError as error:
            self.skipTest(f"IAM credentials are not configured: {error}")
        return self.wrapper

    def test_generate_text(self):
        text = self.wrapper.generate_text("What is the capital of France?", self.chat_model,
                                          system="Answer with one word.", max_tokens=20)
        print("generate_text:", text)
        self.assertIn("paris", text.lower())
        self.assertEqual(self.wrapper.last_model, self.chat_model)

    def test_converse_usage_and_token_count(self):
        request = {"messages": [{"role": "user", "content": [{"text": "Say hi in three words."}]}],
                   "max_tokens": 30}
        response = self.wrapper.converse(request, self.chat_model)
        self.assertTrue(AWSWrapper.extract_text(response))
        self.assertGreater(response["usage"]["inputTokens"], 0)
        try:
            count = self.wrapper.count_tokens(request, self.chat_model)
        except AWSError as error:
            self.skipTest(f"CountTokens is not available for {self.chat_model}: {error}")
        print("count_tokens:", count)
        self.assertGreater(count["inputTokens"], 0)

    def test_stream_text(self):
        chunks = list(self.wrapper.stream_text("Count from 1 to 5, digits only.", self.chat_model, max_tokens=40))
        print("stream_text:", chunks)
        self.assertTrue(chunks)
        self.assertIn("3", "".join(chunks))

    def test_tool_call(self):
        response = self.wrapper.converse({
            "messages": [{"role": "user", "content": [{"text": "What is the weather in Paris right now?"}]}],
            "system": "Use the get_weather tool to answer weather questions.",
            "tools": [WEATHER_TOOL], "tool_choice": "required", "max_tokens": 200}, self.tool_model)
        calls = AWSWrapper.extract_tool_calls(response)
        print("tool calls:", calls)
        self.assertEqual(response["stopReason"], "tool_use")
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertIn("paris", calls[0]["function"]["arguments"].lower())

    def test_embeddings(self):
        embed_input = EmbedInput(["hello world", "bonjour le monde"])
        embed_input.set_default_values("aws")
        result = RemoteEmbedModel(API_KEY, "aws", OPTIONS).get_embeddings(embed_input)
        self.assertEqual(len(result["embeddings"]), 2)
        self.assertGreater(len(result["embeddings"][0]), 100)

    def test_vision(self):
        image = base64.b64encode(_png(64, 64, (255, 0, 0))).decode()
        vision_input = VisionModelInput("What is the main color of this image? Answer with one word.",
                                        image_data=image, extension="png", model=self.chat_model)
        text = RemoteVisionModel(API_KEY, "aws", OPTIONS).image_to_text(vision_input)
        print("vision:", text)
        self.assertIn("red", text.lower())

    def test_list_models(self):
        models = self.wrapper.list_foundation_models(output_modality="TEXT")
        self.assertTrue(models.get("modelSummaries"))
        profiles = self.wrapper.list_inference_profiles(max_results=5)
        self.assertIn("inferenceProfileSummaries", profiles)

    def test_chatbot(self):
        bot = Chatbot(API_KEY, ChatProvider.AWS, OPTIONS)
        chat_input = ChatModelInput("You are a calculator. Answer with the number only.",
                                    model=self.chat_model, max_tokens=20)
        chat_input.add_user_message("What is 2 + 2?")
        result = bot.chat(chat_input)
        print("chatbot:", result)
        self.assertIn("4", result[0])
        self.assertIn("4", "".join(bot.stream(chat_input)))

    def test_flow(self):
        model_params = {"key": API_KEY, "model": self.chat_model, "max_tokens": 200}
        writer = Agent("text", "aws", "Write one sentence about the given topic.", model_params, OPTIONS)
        translator = Agent("text", "aws", "Translate the text to French. Return the translation only.",
                           model_params, OPTIONS)
        tasks = {"write": Task(TextTaskInput("The ocean"), writer),
                 "translate": Task(TextTaskInput("Translate the sentence"), translator)}
        flow = Flow(tasks=tasks, map_paths={"write": ["translate"]})
        output = asyncio.run(flow.start())
        print("flow:", output)
        self.assertEqual(flow.errors, {})
        self.assertTrue(output["translate"]["output"].strip())

    @unittest.skipUnless(os.getenv("AWS_IMAGE_TESTS", "").strip() in ("1", "true"), "Set AWS_IMAGE_TESTS=1")
    def test_generate_image(self):
        response = self.wrapper.generate_image("A small red paper boat on calm water", width=512, height=512)
        self.assertTrue(AWSWrapper.extract_images(response))

    def test_polly_speech(self):
        audio = self.iam_wrapper().synthesize_speech("Hello from Intelli.")
        self.assertGreater(len(audio), 1000)

    @unittest.skipUnless(os.getenv("AWS_KNOWLEDGE_BASE_ID"), "Set AWS_KNOWLEDGE_BASE_ID")
    def test_knowledge_base_retrieve(self):
        response = self.iam_wrapper().retrieve(os.environ["AWS_KNOWLEDGE_BASE_ID"], "overview", 2)
        self.assertIn("retrievalResults", response)


if __name__ == "__main__":
    unittest.main()

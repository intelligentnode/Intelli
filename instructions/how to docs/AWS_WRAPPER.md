# AWSWrapper Guide (Amazon Bedrock + Polly)

Read this before writing code that calls Amazon Bedrock models, Knowledge Bases, Bedrock Agents, AgentCore or Amazon Polly through Intelli.

## 1. Purpose and when to use it

`AWSWrapper` (`intelli/wrappers/aws_wrapper.py`) is the one entry point for AWS AI in Intelli. It calls the AWS REST APIs directly with `requests`, so it has no required dependency. The AWS SDK (`boto3`) is optional and is only used to read credentials from profiles, SSO and IAM roles.

| You want | Use |
| --- | --- |
| Chat with any Bedrock model (Claude, Nova, Llama, Mistral, ...), tools, vision, documents, streaming | `AWSWrapper.converse` / `generate_text` / `stream_text` |
| Embeddings, image generation, video generation | `get_embeddings`, `generate_image`, `generate_video` |
| RAG on a Bedrock Knowledge Base, Bedrock Agents, AgentCore Runtime agents | `retrieve`, `retrieve_and_generate`, `invoke_agent`, `invoke_agent_runtime` |
| Text to speech | `synthesize_speech` (Amazon Polly) |
| Provider-agnostic chat / Flow agents | `Chatbot(key, "aws", options)` and `Agent(..., provider="aws", options=...)` build an `AWSWrapper` for you (section 5) |

```python
from intelli.wrappers.aws_wrapper import AWSWrapper, AWSError
```

## 2. Quick start

```python
import os
from intelli.wrappers.aws_wrapper import AWSWrapper

w = AWSWrapper(os.environ["AWS_BEARER_TOKEN_BEDROCK"], region="us-east-1")   # Bedrock API key
print(w.generate_text("Explain RAG in one sentence.", "us.amazon.nova-lite-v1:0",
                      system="Be brief.", max_tokens=200))

for chunk in w.stream_text("Count to five.", "us.anthropic.claude-haiku-4-5-20251001-v1:0"):
    print(chunk, end="", flush=True)
```

## 3. Auth modes

Constructor: `AWSWrapper(api_key=None, timeout=180, *, region=None, access_key_id=None, secret_access_key=None, session_token=None, profile=None, credentials=None, base_url=None, session=None, fallback_cooldown=0)`

| Mode | Constructor | Auth sent |
| --- | --- | --- |
| Bedrock API key | `AWSWrapper(BEDROCK_API_KEY, region="us-east-1")` | `Authorization: Bearer <key>` |
| IAM keys | `AWSWrapper(access_key_id=ID, secret_access_key=SECRET, region="us-east-1")` (add `session_token` for temporary keys) | SigV4 signature |
| Environment | `AWSWrapper()` with `AWS_BEARER_TOKEN_BEDROCK`, or `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` (+ `AWS_SESSION_TOKEN`) | Bearer or SigV4 |
| Profile / SSO / IAM role | `AWSWrapper(profile="dev")`, or `AWSWrapper()` on a machine with a role. Needs `pip install intelli[aws]` | SigV4 signature |
| Your own boto3 session | `AWSWrapper(credentials=boto3.Session(profile_name="dev"))` (also accepts botocore credentials) | SigV4 signature |

Rules the code follows:

- A Bedrock API key works for Bedrock model calls only: Converse, InvokeModel, embeddings, images, video jobs, token counting, guardrails and model listing. Knowledge Bases, Bedrock Agents, AgentCore and Polly need IAM credentials (AWS does not accept the key there).
- When both an API key and IAM credentials are available, the key is used for the Bedrock model calls and SigV4 for the other services.
- `AWS_BEARER_TOKEN_BEDROCK` is read only when you pass no `api_key`, IAM keys, `profile` or `credentials`.
- Region: `region`, else `AWS_REGION`, else `AWS_DEFAULT_REGION`, else the region of the profile (SDK installed), else `us-east-1`.
- SigV4 is implemented in the wrapper. With the SDK installed, credentials are read again for every request, so role credentials refresh.
- `base_url` replaces the Bedrock Runtime endpoint (a string), or several endpoints (a dict keyed by `bedrock-runtime`, `bedrock`, `bedrock-agent-runtime`, `bedrock-agentcore`, `polly`), for VPC endpoints and proxies.

`AWSWrapper.from_options(api_key=None, options=None, timeout=None)` builds a wrapper from a Chatbot / Agent / controller options dict. Keys read: `region`, `access_key_id`, `secret_access_key`, `session_token`, `profile` (each also with an `aws_` prefix), `credentials`, `fallback_cooldown`, `timeout`.

## 4. Capability matrix

| Feature | Methods | Bedrock API key | IAM credentials |
| --- | --- | --- | --- |
| Chat, system prompt, tools, structured output, guardrails | `converse`, `generate_text`, `build_converse` | yes | yes |
| Streaming | `converse_stream`, `stream_text` | yes | yes |
| Model fallback | `converse(..., fallback_models=[...])` | yes | yes |
| Images, documents, video, audio as input | `media_block`, `image_to_text`, `generate_text(media=...)` | yes | yes |
| Token counting | `count_tokens` | yes | yes |
| Native model requests | `invoke_model`, `invoke_model_stream` | yes | yes |
| Embeddings (Titan, Nova, Cohere) | `get_embeddings` | yes | yes |
| Image generation (Nova Canvas, Titan, Stability) | `generate_image`, `extract_images` | yes | yes |
| Video generation (Nova Reel, output to S3) | `generate_video`, `start_async_invoke`, `get_async_invoke`, `wait_for_async_invoke` | yes | yes |
| Guardrail check without a model call | `apply_guardrail` | yes | yes |
| Model and inference profile listing | `list_foundation_models`, `list_inference_profiles` | yes | yes |
| Knowledge Base search and RAG answers | `retrieve`, `retrieve_and_generate`, `retrieval_to_text` | no | yes |
| Bedrock Agents | `invoke_agent`, `stream_agent` | no | yes |
| AgentCore Runtime agents | `invoke_agent_runtime` | no | yes (or an OAuth `access_token`) |
| Text to speech (Polly) | `synthesize_speech`, `list_voices` | no | yes |

Not covered: Amazon Transcribe (speech to text needs an S3 upload or an HTTP/2 stream) and Nova Sonic (bidirectional streaming).

## 5. Flows, agents and Chatbot

The provider name is `aws`. `key` is the Bedrock API key and is optional: leave it out to use IAM credentials from `options` or from the environment / AWS credential chain.

```python
import asyncio
from intelli.flow import Agent, Task, TextTaskInput, Flow

options = {"region": "us-east-1"}   # or add access_key_id / secret_access_key / profile

writer = Agent(
    agent_type="text", provider="aws", mission="Write a short match briefing.",
    model_params={
        "key": BEDROCK_API_KEY,
        "model": "us.anthropic.claude-sonnet-4-6",
        "fallback_models": ["us.amazon.nova-lite-v1:0"],   # used when Claude is throttled or not granted
        "max_tokens": 400,
    },
    options=options,
)
editor = Agent("text", "aws", "Summarize in one line.",
               {"key": BEDROCK_API_KEY, "model": "us.amazon.nova-lite-v1:0"}, options)

flow = Flow(
    tasks={"write": Task(TextTaskInput("Brazil vs France"), writer),
           "edit": Task(TextTaskInput("Summarize the briefing"), editor)},
    map_paths={"write": ["edit"]},
)
print(asyncio.run(flow.start())["edit"]["output"])
```

| Agent type | `model_params` | What runs |
| --- | --- | --- |
| `text` | `model`, `max_tokens`, `temperature`, `fallback_models`, `tools`, `tool_choice`, plus any Converse request field (`guardrailConfig`, `additionalModelRequestFields`, `outputConfig`, ...) | Bedrock Converse |
| `image` | `model` (default `amazon.nova-canvas-v1:0`), `width`, `height` | Nova Canvas / Titan / Stability, returns base64 |
| `vision` | `model` (required), `extension` | Converse with an image block |
| `embed` | `model` (default `amazon.titan-embed-text-v2:0`) | `{"embeddings": [[...]], ...}` |
| `speech` | `voice` (Polly voice id, default `Joanna` / `Matthew` by `gender`), `model` (Polly engine: `standard`, `neural`, `long-form`, `generative`) | Polly, returns mp3 bytes. IAM credentials only |
| `search` | `knowledge_base_id`, `k`, `as_text` | Knowledge Base retrieval. IAM credentials only |

Tool routing works like the other providers: a `text` agent with `tools` returns `{"type": "tool_response", "tool_calls": [...]}`, which `ToolDynamicConnector` routes on. Tools can be in OpenAI, Anthropic or Bedrock (`toolSpec`) format.

Chatbot:

```python
from intelli.function.chatbot import Chatbot, ChatProvider
from intelli.model.input.chatbot_input import ChatModelInput

bot = Chatbot(BEDROCK_API_KEY, ChatProvider.AWS, {"region": "us-east-1"})
chat_input = ChatModelInput("You are a helpful assistant.", model="us.amazon.nova-lite-v1:0", max_tokens=200)
chat_input.add_user_message("What is the capital of France?")
print(bot.chat(chat_input)[0])
for chunk in bot.stream(chat_input):
    print(chunk, end="")
```

Controllers: `RemoteEmbedModel(key, "aws", options)`, `RemoteImageModel(key, "aws", options)`, `RemoteVisionModel(key, "aws", options)`, `RemoteSpeechModel(None, "aws", options)`.

## 6. Wrapper examples

`converse(params, model=None, *, fallback_models=None)` takes a Converse request body. These extra keys are mapped for you: `model`, `tools` / `tool_choice`, `max_tokens`, `temperature`, `top_p`, `stop_sequences`, and `system` as a plain string. Every other key is sent as it is.

```python
import json

response = w.converse({
    "model": "us.anthropic.claude-sonnet-4-6",
    "system": "You are a football analyst.",
    "messages": [{"role": "user", "content": [{"text": "Who is favorite, Brazil or France?"}]}],
    "max_tokens": 300,
}, fallback_models=["us.amazon.nova-lite-v1:0"])
print(AWSWrapper.extract_text(response), w.last_model, response["usage"])

# Tools (OpenAI, Anthropic or Bedrock format)
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
response = w.converse({"messages": [...], "tools": tools, "tool_choice": "auto"}, MODEL)
calls = AWSWrapper.extract_tool_calls(response)      # [{"id", "type": "function", "function": {"name", "arguments"}}]

# Images, PDFs and video: a file path, bytes, base64 or an s3:// URI
text = w.generate_text("Summarize this report.", MODEL, media=["report.pdf", "chart.png"])

# Structured output (models that support it)
schema = {"type": "object", "properties": {"winner": {"type": "string"}}, "required": ["winner"],
          "additionalProperties": False}
response = w.converse({"messages": [...], "outputConfig": {"textFormat": {
    "type": "json_schema", "structure": {"jsonSchema": {"name": "prediction", "schema": json.dumps(schema)}}}}}, MODEL)

# Guardrail on a model call, and on its own
w.converse({"messages": [...], "guardrailConfig": {"guardrailIdentifier": "gr-id", "guardrailVersion": "1"}}, MODEL)
w.apply_guardrail("gr-id", "text to check", version="1", source="INPUT")

# Stream events (the same shape boto3 returns)
for event in w.converse_stream({"messages": [...]}, MODEL):
    if "contentBlockDelta" in event:
        print(event["contentBlockDelta"]["delta"].get("text", ""), end="")

w.count_tokens({"messages": [...], "system": "..."}, MODEL)   # {"inputTokens": n}

# Embeddings, images, video
vectors = w.get_embeddings({"texts": ["a", "b"], "dimensions": 256})["embeddings"]
images = AWSWrapper.extract_images(w.generate_image("a red fox", width=1024, height=1024))   # base64
job = w.generate_video("a cat on a beach", "s3://my-bucket/videos")
done = w.wait_for_async_invoke(job["invocationArn"])   # the mp4 is in the S3 folder

# Any model's native request
w.invoke_model(MODEL, {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 100,
                       "messages": [{"role": "user", "content": "Hello"}]})

# Knowledge Bases, Agents, AgentCore, Polly (IAM credentials)
hits = w.retrieve("KB12345678", "What is our refund policy?", 5)
print(AWSWrapper.retrieval_to_text(hits))
answer = w.retrieve_and_generate("What is our refund policy?", "KB12345678", MODEL_ARN)["output"]["text"]
reply = w.invoke_agent("AGENTID", "ALIASID", "Book a table for two")       # {"text", "session_id", "events"}
result = w.invoke_agent_runtime(AGENT_RUNTIME_ARN, {"prompt": "hello"}, session_id=SESSION_ID)
audio = w.synthesize_speech("Hello from Intelli.", "Joanna")               # mp3 bytes
```

### Model fallback

`fallback_models` are tried in order when the model before them answers with access not granted or not found (403, 404), throttling or quota (429), a timeout (408), a model error (424) or a service error (500, 503). A bad request (400) or a network error stops the chain and is raised. The model that answered is in `wrapper.last_model`.

With `fallback_cooldown` (seconds; wrapper argument or options key), a model that failed this way is skipped until the time is over, so a throttled model is not called again on every request. The memory is shared by all wrappers in the process, per region and model. Streaming uses the first model only.

## 7. Errors and gotchas

`AWSError` has `status_code`, `error_type` (for example `ThrottlingException`) and `details`. Keys are removed from the message.

1. Most models need the inference profile id, not the bare model id: `us.anthropic.claude-sonnet-4-6`, not `anthropic.claude-sonnet-4-6`. The error "on-demand throughput isn't supported" means this. `list_inference_profiles()` shows the ids of your region.
2. Anthropic models need the use case form (Bedrock console, Model access). Until it is submitted the API answers 403 / 404 with "use case"; Amazon Nova models need no form, which makes them a good fallback.
3. The default chat model is the Amazon Nova Lite profile of the region (`us.`, `eu.` or `apac.` prefix). Pass `model` for anything else. Defaults are in `config['url']['aws']['models']`.
4. Claude Opus 4.7+ and the Claude 5 family reject `temperature`; the `aws` chat input leaves it out for them, as the `anthropic` provider does.
5. Polly, Knowledge Bases, Agents and AgentCore raise "a Bedrock API key cannot call ..." when no IAM credentials are available.
6. AgentCore session ids must be at least 33 characters.

## 8. Tests

- `intelli/test/unit/test_aws_wrapper.py`: offline. It also checks, when `botocore` is installed, that each request (URL, body and SigV4 signature) is the same as the one the AWS SDK builds.
- `intelli/test/integration/test_aws_wrapper.py`: live, runs only with `AWS_LIVE_TESTS=1` and credentials (see the file header).

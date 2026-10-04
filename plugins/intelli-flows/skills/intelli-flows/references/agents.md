# Agent types and providers

`Agent(agent_type, provider, mission, model_params, options=None)`. Every provider except the local ones
takes its API key as `model_params["key"]`.

| Agent type | Receives | Returns | Providers | `model_params` beyond `key` |
| --- | --- | --- | --- | --- |
| `text` | text | text, or a tool call when `tools` is set | `openai`, `anthropic`, `gemini`, `aws`, `mistral`, `nvidia`, `vllm`, `llamacpp`, `keras` | `model`, `max_tokens`, `temperature`, `tools`, `tool_choice`; `aws` also takes `fallback_models` |
| `image` | text | image (base64 string) | `openai`, `gemini`, `aws`, `stability` | `model`, `width`, `height` (`openai`: `gpt-image-2` with 1024 x 1024) |
| `vision` | image | text | `openai`, `gemini`, `aws`, `google` (Cloud Vision) | `model` (required), `extension` (default `png`) |
| `speech` | text | audio (bytes) | `openai`, `gemini`, `elevenlabs`, `aws` (Polly), `google` | `model`, `voice`; `openai` also needs `"stream": False` |
| `recognition` | audio | text | `openai`, `elevenlabs`, `speechmatics`, `keras` | `model` (`openai` default `whisper-1`), `language` |
| `embed` | text | embedding vectors (the format depends on the provider) | `openai`, `gemini`, `mistral`, `aws`, `nvidia`, `vllm` | `model` |
| `search` | text | text | Intellicloud (`one_key`), Google Custom Search (`google_api_key`, `google_cse_id`), Amazon Bedrock Knowledge Base (`knowledge_base_id`, provider `aws`) | `k` |
| `mcp` | text | text | any MCP server | `command` + `args`, or `url`; `tool`; `arg_<name>` values |

Notes

- Local providers need no key: `vllm` (Ollama or vLLM, `options={"baseUrl": "http://localhost:11434"}`),
  `llamacpp` (`options={"model_path": ...}`) and `keras` (`options={"model_name": ...}`).
- `aws` uses the Bedrock API key in `key`, or IAM credentials in `options`
  (`region`, `access_key_id`, `secret_access_key`, or `profile`). Its speech and search steps need IAM
  credentials. Model ids are inference profiles such as `us.anthropic.claude-sonnet-4-6` or
  `us.amazon.nova-lite-v1:0`.
- `gemini` on Vertex AI: add `options={"vertex": True, "project_id": ..., "location": ...}`.
- An `embed` step's output is not text. End the flow with it, or pass it to a `CustomAgent` step that
  stores or compares the vectors.
- `Flow(..., auto_save_outputs=True, output_dir="./outputs")` writes each image as PNG, each audio clip
  as MP3 and each text output as TXT, named after the step.
- The picture colors steps by type: text sky blue, image green, vision salmon, speech gold, recognition
  orchid, embed coral, search light blue, MCP purple, others gray.

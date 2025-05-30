config = {
    "url": {
        "intellicloud": {
            "base": "https://ue8sdr9bij.execute-api.us-east-2.amazonaws.com/v1",
            "semantic_search": "/semantic_search/"
        },
        "openai": {
            "base": "https://api.openai.com",
            "completions": "/v1/completions",
            "chatgpt": "/v1/chat/completions",
            "imagegenerate": "/v1/images/generations",
            "embeddings": "/v1/embeddings",
            "audiotranscriptions": "/v1/audio/transcriptions",
            "audiospeech": "/v1/audio/speech",
            "files": "/v1/files",
            "finetuning": "/v1/fine_tuning/jobs",
            "organization": None
        },
        "azure_openai": {
            "base": "https://{resource-name}.openai.azure.com/openai",
            "completions": "/deployments/{deployment-id}/completions?api-version={api-version}",
            "chatgpt": "/deployments/{deployment-id}/chat/completions?api-version={api-version}",
            "imagegenerate": "/images/generations:submit?api-version={api-version}",
            "embeddings": "/deployments/{deployment-id}/embeddings?api-version={api-version}",
            "audiotranscriptions": "/deployments/{deployment-id}/audio/transcriptions?api-version={api-version}",
            "audiospeech": "/deployments/{deployment-id}/audio/speech?api-version={api-version}",
            "files": "/files?api-version={api-version}",
            "finetuning": "/fine_tuning/jobs?api-version={api-version}"
        },
        "cohere": {
            "base": "https://api.cohere.ai",
            "completions": "/generate",
            "chat": "/chat",
            "embed": "/v1/embed",
            "version": "2022-12-06"
        },
        "google": {
            "base": "https://{0}.googleapis.com/v1",
            "speech": {
                "prefix": "texttospeech",
                "synthesize": {
                    "postfix": "/text:synthesize"
                }
            },
            "speechtotext": {
                "prefix": "speech",
                "recognize": {
                    "postfix": "/speech:recognize"
                }
            },
            "vision": {
                "prefix": "vision",
                "annotate": {
                    "postfix": "/images:annotate"
                }
            },
            "language": {
                "prefix": "language",
                "analyze": {
                    "postfix": "/documents:analyzeSentiment"
                }
            },
            "translation": {
                "prefix": "translate",
                "translate": {
                    "postfix": "/languages:translate"
                }
            }
        },
        "stability": {
            "base": "https://api.stability.ai",
            "text_to_image": "/v1/generation/{0}/text-to-image",
            "upscale": "/v1/generation/{0}/image-to-image/upscale",
            "image_to_image": "/v1/generation/{0}/image-to-image"
        },
        "huggingface": {
            "base": "https://api-inference.huggingface.co/models"
        },
        "replicate": {
            "base": "https://api.replicate.com",
            "predictions": "/v1/predictions"
        },
        "mistral": {
            "base": "https://api.mistral.ai",
            "completions": "/v1/chat/completions",
            "embed": "/v1/embeddings"
        },
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
            },
            "endpoints": {
                "generateContent": ":generateContent",
                "embedContent": ":embedContent",
                "batchEmbedContents": ":batchEmbedContents",
                "predictLongRunning": ":predictLongRunning",
                "upload": "",
                "files": ""
            }
        },
        "anthropic": {
            "base": "https://api.anthropic.com",
            "messages": "/v1/messages",
            "version": "2023-06-01"
        },
        "nvidia": {
            "base": "https://integrate.api.nvidia.com",
            "chat": "/v1/chat/completions",
            "embeddings": "/v1/embeddings"
        },
        "vllm": {
            "completions": "/v1/completions",
            "chat": "/v1/chat/completions",
            "embed": "/embed"
        },
        "elevenlabs": {
            "base": "https://api.elevenlabs.io/v1",
            "models": {
                "tts": "eleven_multilingual_v2",
                "stt": "scribe_v1",
                "sts": "eleven_multilingual_sts_v2"
            }
        }
    },
    "models": {
        "replicate": {
            "llama": {
                "70b": "70b-chat",
                "13b": "13b-chat",
                "70b-chat": "70b-chat",
                "13b-chat": "13b-chat",
                "34b-code": "34b-code",
                "34b-python": "34b-python",
                "13b-code-instruct": "13b-code-instruct",
                "llama-2-13b-embeddings": "llama-2-13b-embeddings",
                "70b-chat-version": "02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
                "13b-chat-version": "f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d",
                "34b-code-version": "efbd2ef6feefb242f359030fa6fe08ce32bfced18f3868b2915db41d41251b46",
                "34b-python-version": "482ba325daab209d121f45a0030f2f3ed942df98b185d41635ab3f19165a3547",
                "13b-code-instruct-version": "ca8c51bf3c1aaf181f9df6f10f31768f065c9dddce4407438adc5975a59ce530",
                "llama-2-13b-embeddings-version": "7115a4c65b86815e31412e53de1211c520164c190945a84c425b59dccbc47148"
            }
        }
    }
}
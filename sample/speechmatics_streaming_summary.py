import os
import asyncio
from dotenv import load_dotenv

from intelli.flow import Flow, Task, TextTaskInput, Agent
from intelli.flow.input.agent_input import AgentInput
from intelli.flow.agents.speechmatics_agent import StreamingSpeechAgent

load_dotenv()


async def main():
    """
    Stream transcriptions from Speechmatics (via `on_chunk`) while the full transcript is aggregated and then summarized by an LLM.
    The listener callback runs on every final transcript chunk, so you can forward updates to a UI/websocket/logger in real time.

    You can send listener using listener_callback function.
    """

    # Input audio (repo root): ./temp/temp.mp3
    audio_path = os.path.join(os.getcwd(), "temp", "temp.mp3")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    out_dir = os.path.join(
        os.path.dirname(__file__), "temp", "speechmatics_streaming_summary"
    )
    os.makedirs(out_dir, exist_ok=True)

    speechmatics_key = os.getenv("SPEECHMATICS_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    if not speechmatics_key:
        raise ValueError("Missing SPEECHMATICS_API_KEY")
    if not openai_key:
        raise ValueError("Missing OPENAI_API_KEY")

    def on_chunk(text: str):
        print(f"[stream] {text}")

    stt_agent = StreamingSpeechAgent(
        api_key=speechmatics_key,
        listener_callback=on_chunk,
    )

    llm_agent = Agent(
        agent_type="text",
        provider="openai",
        mission="Summarize the transcript in 5 bullet points.",
        model_params={"key": openai_key, "model": "gpt-4o-mini"},
    )

    flow = Flow(
        tasks={
            "transcribe": Task(
                TextTaskInput("Transcribe the audio."),
                agent=stt_agent,
                model_params={
                    "max_wait_seconds": 30,
                    "idle_timeout_seconds": 2,
                    "chunk_ms": 50,
                },
                log=True,
            ),
            "summarize": Task(
                TextTaskInput("Summarize the transcript."),
                agent=llm_agent,
                log=True,
            ),
        },
        map_paths={"transcribe": ["summarize"]},
        log=True,
        auto_save_outputs=True,
        output_dir=out_dir,
    )

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    results = await flow.start(initial_input=audio_bytes, initial_input_type="audio")

    print(f"Saved outputs to: {out_dir}")
    print("Transcript:", results["transcribe"]["output"][:400], "...")
    print("Summary:", results["summarize"]["output"])


if __name__ == "__main__":
    asyncio.run(main())

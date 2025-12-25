import os
import asyncio
from dotenv import load_dotenv

from intelli.flow import VibeAgent

load_dotenv()

async def main():
    out_dir = os.path.join(os.path.dirname(__file__), "temp", "vibeagent_research_factory")
    os.makedirs(out_dir, exist_ok=True)

    vagent = VibeAgent(
        planner_provider="openai",
        planner_api_key=os.getenv("OPENAI_API_KEY"),
        planner_model="gpt-5.2",
        planner_options={"temperature": 0.2},
        image_model="gemini gemini-3-pro-image-preview",
    )

    intent = (
        "Create a 3-step linear flow for a 'Research-to-Content Factory': "
        "1. Search: Perform a web research using ONLY 'google' as provider for solid-state battery breakthroughs in the last 30 days. "
        "2. Analyst: Summarize the findings into key technical metrics. "
        "3. Creator: Generate an image using 'gemini' showing a futuristic representation of these battery findings."
    )

    flow = await vagent.build(
        intent,
        save_dir=out_dir,
        graph_name="vibeagent_research_factory_graph",
        render_graph=True,
    )

    flow.output_dir = out_dir
    flow.auto_save_outputs = True

    results = await flow.start()

    print(f"Saved outputs to: {out_dir}")
    for name, data in results.items():
        preview = data.get("output")
        if isinstance(preview, str):
            preview = preview[:120].replace("\n", " ")
        print(f"- {name}: type={data.get('type')} preview={preview}")

if __name__ == "__main__":
    asyncio.run(main())


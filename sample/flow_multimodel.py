import os
import asyncio
from dotenv import load_dotenv
from intelli.flow import Agent, Task, Flow, TextTaskInput, AgentTypes

# Load environment variables
load_dotenv()

# Access API keys
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
MISTRAL_KEY = os.getenv("MISTRAL_API_KEY")
STABILITY_KEY = os.getenv("STABILITY_API_KEY")
ELEVENLABS_KEY = os.getenv("ELEVENLABS_API_KEY")

def get_elevenlabs_voice_id():
    """Helper method to get an ElevenLabs voice ID"""
    if not ELEVENLABS_KEY:
        return None

    try:
        from intelli.controller.remote_speech_model import RemoteSpeechModel
        speech_model = RemoteSpeechModel(
            key_value=ELEVENLABS_KEY, provider="elevenlabs"
        )

        # List available voices
        voices_result = speech_model.list_voices()
        if "voices" in voices_result and len(voices_result["voices"]) > 0:
            # Get the first voice ID
            voice_id = voices_result["voices"][0]["voice_id"]
            print(f"🔊 Using ElevenLabs voice: {voices_result['voices'][0]['name']} ({voice_id})")
            return voice_id
    except Exception as e:
        print(f"⚠️ Error getting ElevenLabs voices: {e}")

    return None

def create_agents():
    """Create and return all required agents"""
    # Get a valid ElevenLabs voice ID
    voice_id = get_elevenlabs_voice_id()
    
    # Text generation agent for itinerary creation
    itinerary_agent = Agent(
        agent_type=AgentTypes.TEXT.value,
        provider="anthropic",
        mission="Create a detailed travel itinerary",
        model_params={"key": ANTHROPIC_KEY, "model": "claude-3-7-sonnet-20250219"}
    )

    # Speech synthesis agent for audio guides
    speech_agent = Agent(
        agent_type=AgentTypes.SPEECH.value,
        provider="elevenlabs",
        mission="Convert travel information to speech",
        model_params={
            "key": ELEVENLABS_KEY,
            "voice": voice_id,  # Use the retrieved voice ID
            "model": "eleven_multilingual_v2"
        }
    )

    # Image generation agent
    image_agent = Agent(
        agent_type=AgentTypes.IMAGE.value,
        provider="stability",
        mission="Generate travel destination images",
        model_params={"key": STABILITY_KEY}
    )

    # Vision analysis agent
    vision_agent = Agent(
        agent_type=AgentTypes.VISION.value,
        provider="openai",
        mission="Analyze travel destination images",
        model_params={
            "key": OPENAI_KEY,
            "model": "gpt-4o",
            "extension": "png"
        }
    )

    # Text agent for final travel guide creation
    guide_agent = Agent(
        agent_type=AgentTypes.TEXT.value,
        provider="mistral",
        mission="Create comprehensive travel guides",
        model_params={"key": MISTRAL_KEY, "model": "mistral-medium"}
    )
    
    return itinerary_agent, speech_agent, image_agent, vision_agent, guide_agent

def create_tasks(itinerary_agent, speech_agent, image_agent, vision_agent, guide_agent):
    """Create and return all required tasks"""
    # Task 1: Generate a travel itinerary
    itinerary_task = Task(
        TextTaskInput(
            "Create a 3-day travel itinerary for Rome, Italy. Include major attractions, food recommendations, and transportation tips."
        ),
        itinerary_agent,
        log=True
    )

    # Task 2: Convert the text to speech
    speech_task = Task(
        TextTaskInput(
            "Convert the first day of this itinerary to speech for the traveler"
        ),
        speech_agent,
        log=True
    )

    # Task 3: Generate an image prompt
    image_prompt_task = Task(
        TextTaskInput(
            "Create a short, specific image generation prompt for Rome showing the iconic Colosseum"
        ),
        itinerary_agent,
        log=True
    )

    # Task 4: Generate a destination image
    image_task = Task(
        TextTaskInput(
            "Rome with the iconic Colosseum under clear blue sky"
        ),
        image_agent,
        log=True
    )

    # Task 5: Analyze the generated image
    vision_task = Task(
        TextTaskInput(
            "Identify the landmarks and notable features in this image that would be relevant for a traveler"
        ),
        vision_agent,
        log=True
    )

    # Task 6: Combine all information
    guide_task = Task(
        TextTaskInput(
            "Create a comprehensive travel guide for Rome by combining the itinerary and image analysis"
        ),
        guide_agent,
        log=True
    )
    
    return itinerary_task, speech_task, image_prompt_task, image_task, vision_task, guide_task

async def run_travel_assistant(itinerary_task, speech_task, image_prompt_task, image_task, vision_task, guide_task):
    """Create and execute the workflow"""
    # Create the flow with task map
    flow = Flow(
        tasks={
            "itinerary": itinerary_task,
            "speech": speech_task,
            "image_prompt": image_prompt_task,
            "image": image_task,
            "vision": vision_task,
            "guide": guide_task
        },
        map_paths={
            "itinerary": ["speech", "image_prompt", "guide"],
            "image_prompt": ["image"],
            "image": ["vision"],
            "vision": ["guide"],
            "guide": []
        },
        log=True
    )
    
    # Generate flow visualization
    flow.generate_graph_img(
        name="travel_assistant_flow",
        save_path="../temp"
    )
    
    # Execute the flow
    results = await flow.start(max_workers=3)
    
    # Save speech output if it exists
    if "speech" in results and results["speech"]:
        try:
            audio_path = "../temp/rome_day1_audio.mp3"
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)
            
            # Extract the audio data
            audio_data = results["speech"]["output"] if isinstance(results["speech"], dict) else results["speech"]
            
            with open(audio_path, "wb") as f:
                f.write(audio_data)
            print(f"Audio guide saved to {audio_path}")
        except Exception as e:
            print(f"Error saving audio file: {e}")
    
    return results

def execute_flow():
    """Main function to execute the travel assistant workflow"""
    # Create agents and tasks
    itinerary_agent, speech_agent, image_agent, vision_agent, guide_agent = create_agents()
    itinerary_task, speech_task, image_prompt_task, image_task, vision_task, guide_task = create_tasks(
        itinerary_agent, speech_agent, image_agent, vision_agent, guide_agent
    )
    
    # Run the flow
    results = asyncio.run(run_travel_assistant(
        itinerary_task, speech_task, image_prompt_task, image_task, vision_task, guide_task
    ))
    
    print("Travel assistant workflow completed successfully!")
    return results

if __name__ == "__main__":
    execute_flow()
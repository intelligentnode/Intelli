#!/usr/bin/env python3
"""
Test script for the new gpt-image-1 model and parameters
"""
import os
from dotenv import load_dotenv
from intelli.model.input.image_input import ImageModelInput
from intelli.controller.remote_image_model import RemoteImageModel

# Load environment variables
load_dotenv()

def test_gpt_image_1_default():
    """Test that gpt-image-1 is set as default"""
    print("=== Testing gpt-image-1 as default model ===")
    
    # Create image input without specifying model
    image_input = ImageModelInput(prompt="A cute baby sea otter")
    
    # Set default values for OpenAI
    image_input.set_default_values("openai")
    
    print(f"Default model: {image_input.model}")
    print(f"Default size: {image_input.imageSize}")
    
    assert image_input.model == "gpt-image-1", f"Expected gpt-image-1, got {image_input.model}"
    print("✅ Default model test passed!")

def test_new_parameters():
    """Test that new parameters are included in OpenAI inputs"""
    print("\n=== Testing new parameters ===")
    
    image_input = ImageModelInput(
        prompt="A cute baby sea otter",
        model="gpt-image-1",
        background="transparent",
        quality="high",
        output_format="png",
        output_compression=90,
        moderation="auto",
        user="test_user"
    )
    
    openai_inputs = image_input.get_openai_inputs()
    
    print("OpenAI inputs:")
    for key, value in openai_inputs.items():
        print(f"  {key}: {value}")
    
    # Check that new parameters are included
    expected_params = ["background", "quality", "output_format", "output_compression", "moderation", "user"]
    for param in expected_params:
        assert param in openai_inputs, f"Parameter {param} not found in OpenAI inputs"
    
    print("✅ New parameters test passed!")

def test_actual_generation():
    """Test actual image generation with gpt-image-1 (requires API key)"""
    print("\n=== Testing actual image generation ===")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️ OPENAI_API_KEY not found, skipping actual generation test")
        return
    
    try:
        # Create image input with new parameters
        image_input = ImageModelInput(
            prompt="A simple geometric logo with a python snake",
            model="gpt-image-1",
            background="transparent",
            quality="high",
            output_format="png",
            output_compression=85
        )
        
        # Create image model
        image_model = RemoteImageModel(api_key, "openai")
        
        print("Generating image with gpt-image-1...")
        results = image_model.generate_images(image_input)
        
        print(f"✅ Image generation successful! Generated {len(results)} image(s)")
        print(f"Result type: {type(results[0])}")
        
        # Save the image if it's base64 data
        if isinstance(results[0], str) and len(results[0]) > 100:
            import base64
            import os
            
            os.makedirs("temp", exist_ok=True)
            image_data = base64.b64decode(results[0])
            
            with open("temp/gpt_image_1_test.png", "wb") as f:
                f.write(image_data)
            
            print("💾 Image saved to temp/gpt_image_1_test.png")
        
    except Exception as e:
        print(f"❌ Image generation failed: {e}")

if __name__ == "__main__":
    print("Testing gpt-image-1 model and new parameters...\n")
    
    test_gpt_image_1_default()
    test_new_parameters()
    test_actual_generation()
    
    print("\n🎉 All tests completed!") 
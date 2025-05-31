# Flow Auto-Save Functionality

The Flow class now supports automatic saving of image, audio, and text outputs to files during execution. This eliminates the need to manually save outputs using `FlowHelper` after flow completion.

## Overview

When `auto_save_outputs=True` is enabled, the Flow will automatically save task outputs to files based on their type:

- **Image outputs** → `.png` files
- **Audio outputs** → `.mp3` files  
- **Text outputs** → `.txt` files

## Basic Usage

```python
from intelli.flow.flow import Flow

# Create flow with auto-save enabled
flow = Flow(
    tasks=tasks,
    map_paths=map_paths,
    auto_save_outputs=True,        # Enable automatic saving
    output_dir="./my_outputs",     # Directory for saved files
    log=True
)

# Execute flow - files are saved automatically
results = await flow.start()

# Get information about saved files
saved_files = flow.get_saved_files()
print(f"Saved {len(saved_files)} files:")
for task_name, file_info in saved_files.items():
    print(f"  {task_name}: {file_info['path']}")
```

## Configuration Options

### Constructor Parameters

- **`auto_save_outputs`** (bool, default: False): Enable/disable automatic saving
- **`output_dir`** (str, default: "./outputs"): Directory where files will be saved
- **`output_file_map`** (dict, optional): Custom file names for specific tasks

### Custom File Names

You can specify custom file names for specific tasks:

```python
flow = Flow(
    tasks=tasks,
    map_paths=map_paths,
    auto_save_outputs=True,
    output_dir="./results",
    output_file_map={
        "image_task": "my_image.png",
        "audio_task": "narration.mp3", 
        "text_task": "story.txt"
    }
)
```

### Default File Naming

When no custom names are provided, files are named using the pattern:
- `{task_name}_output.{extension}`

For example:
- `generate_image_output.png`
- `create_speech_output.mp3`
- `write_story_output.txt`

## File Path Resolution

- **Relative paths** in `output_file_map` are resolved relative to `output_dir`
- **Absolute paths** in `output_file_map` are used as-is
- The `output_dir` is created automatically if it doesn't exist

## Getting Saved File Information

### `get_saved_files()`

Returns a dictionary mapping task names to file information:

```python
saved_files = flow.get_saved_files()
# Returns:
# {
#     "task_name": {
#         "path": "/path/to/file.png",
#         "type": "image", 
#         "size": 1024000
#     }
# }
```

### `get_flow_summary()`

Returns comprehensive flow execution information:

```python
summary = flow.get_flow_summary()
# Returns:
# {
#     "outputs": {...},           # Task outputs
#     "saved_files": {...},       # Saved file info
#     "errors": {...},            # Any errors
#     "auto_save_enabled": True,  # Auto-save status
#     "output_directory": "..."   # Output directory path
# }
```

## Complete Example

```python
import asyncio
import os
from intelli.flow.agents.agent import Agent
from intelli.flow.input.task_input import TextTaskInput
from intelli.flow.tasks.task import Task
from intelli.flow.flow import Flow
from intelli.flow.types import AgentTypes

async def main():
    # Create tasks
    tasks = {
        "story": Task(
            TextTaskInput("Write a short story about AI"),
            Agent(
                agent_type=AgentTypes.TEXT.value,
                provider="openai",
                mission="Create an engaging story",
                model_params={"key": "your-api-key", "model": "gpt-4o"}
            )
        ),
        "image": Task(
            TextTaskInput("Generate an image of a futuristic AI"),
            Agent(
                agent_type=AgentTypes.IMAGE.value,
                provider="stability", 
                mission="Create a futuristic image",
                model_params={"key": "your-api-key"}
            )
        )
    }
    
    # Create flow with auto-save
    flow = Flow(
        tasks=tasks,
        map_paths={"story": ["image"], "image": []},
        auto_save_outputs=True,
        output_dir="./ai_story_outputs",
        output_file_map={
            "story": "ai_story.txt",
            "image": "ai_future.png"
        },
        log=True
    )
    
    # Execute flow
    print("🚀 Starting flow with auto-save...")
    results = await flow.start()
    
    # Check saved files
    saved_files = flow.get_saved_files()
    print(f"\n💾 Auto-saved {len(saved_files)} files:")
    
    for task_name, file_info in saved_files.items():
        path = file_info["path"]
        file_type = file_info["type"]
        size = file_info["size"]
        
        print(f"  📄 {task_name}: {path}")
        print(f"     Type: {file_type}, Size: {size:,} bytes")
        print(f"     Exists: {'✅' if os.path.exists(path) else '❌'}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Benefits

1. **Automatic**: No need to manually save outputs after flow completion
2. **Flexible**: Support for custom file names and paths
3. **Type-aware**: Automatically handles different output types (image, audio, text)
4. **Trackable**: Get detailed information about saved files
5. **Optional**: Can be enabled/disabled as needed

## Migration from Manual Saving

### Before (Manual Saving)
```python
# Execute flow
results = await flow.start()

# Manually save outputs
from intelli.flow.utils.flow_helper import FlowHelper

if "image_task" in results:
    FlowHelper.save_image_output(
        results["image_task"]["output"], 
        "./outputs/image.png"
    )

if "audio_task" in results:
    FlowHelper.save_audio_output(
        results["audio_task"]["output"],
        "./outputs/audio.mp3" 
    )
```

### After (Auto-Save)
```python
# Create flow with auto-save
flow = Flow(
    tasks=tasks,
    map_paths=map_paths,
    auto_save_outputs=True,
    output_dir="./outputs",
    output_file_map={
        "image_task": "image.png",
        "audio_task": "audio.mp3"
    }
)

# Execute flow - files saved automatically
results = await flow.start()
saved_files = flow.get_saved_files()
```

## Error Handling

- If auto-save fails for a task, an error is logged but flow execution continues
- The `get_flow_summary()` method includes any errors that occurred
- Original task outputs are still available in the results even if saving fails

## Performance Considerations

- Auto-save happens immediately after each task completes
- Files are saved in parallel with other task executions
- No significant performance impact on flow execution
- Large files (images/audio) are saved efficiently using binary write operations 
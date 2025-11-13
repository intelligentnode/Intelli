import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Lazy import check
def _check_openai_import():
    try:
        import openai
        from openai.types.beta.assistant import Assistant
        return openai, Assistant
    except ImportError:
        raise ImportError(
            "openai package is required for Azure Assistant support. "
            "Install it with: pip install openai"
        )


class AzureAssistantWrapper:
    """
    Wrapper for Azure OpenAI Assistants API.
    Provides methods to create, update, delete, and retrieve assistants.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        api_version: Optional[str] = None,
    ):
        """
        Initialize Azure OpenAI client for Assistants API.

        Args:
            api_key: Azure OpenAI API key (can be set via AZURE_OPENAI_API_KEY env var)
            base_url: Azure OpenAI endpoint URL (e.g., "https://your-resource.openai.azure.com/")
                     Typically set via AZURE_OPENAI_ENDPOINT environment variable
            api_version: API version to use (optional, defaults to AZURE_OPENAI_API_VERSION env var 
                        or "2024-05-01-preview" for reasoning_effort support)
        """
        if not api_key:
            raise ValueError("Azure OpenAI API key is required")

        if not base_url:
            raise ValueError("Azure OpenAI base_url is required")

        self.api_key = api_key
        self.base_url = base_url.rstrip("/") + "/"  # Ensure trailing slash
        # Read API version from env var if not provided, with default fallback
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")

        # Initialize Azure OpenAI client
        openai, _ = _check_openai_import()
        self.client = openai.AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.base_url,
        )

    def create_assistant(
        self,
        name: str,
        model: str,
        instructions: str,
        description: Optional[str] = None,
        tools: Optional[list] = None,
        tool_resources: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Any:
        """
        Create an assistant in Azure OpenAI.

        Args:
            name: Assistant name
            model: Model name (e.g., 'gpt-4', 'gpt-5-mini')
            instructions: System instructions for the assistant
            description: Optional description
            tools: Optional list of tools
            tool_resources: Optional tool resources
            metadata: Optional metadata
            temperature: Temperature parameter (for GPT-4 models)
            reasoning_effort: Reasoning effort parameter (for GPT-5 models, e.g., 'low', 'medium', 'high')

        Returns:
            Created Assistant object
        """
        model_name_clean = model.lower().strip()

        # Prepare assistant parameters
        assistant_params: Dict[str, Any] = {
            "name": name,
            "model": model,
            "instructions": instructions,
        }

        if description:
            assistant_params["description"] = description
        if tools:
            assistant_params["tools"] = tools
        if tool_resources:
            assistant_params["tool_resources"] = tool_resources
        if metadata:
            assistant_params["metadata"] = metadata

        # Set model-specific parameters
        if model_name_clean.startswith("gpt-4") and not model_name_clean.startswith("gpt-5"):
            # GPT-4 models use temperature
            assistant_params["temperature"] = temperature if temperature is not None else 0.4
            logger.info(f"Using temperature={assistant_params['temperature']} for model: {model}")
        else:
            # GPT-5 and other models use reasoning_effort
            assistant_params["reasoning_effort"] = reasoning_effort if reasoning_effort else "low"
            logger.info(f"Using reasoning_effort={assistant_params['reasoning_effort']} for model: {model}")

        try:
            assistant = self.client.beta.assistants.create(**assistant_params)
            logger.info(f"Successfully created assistant: {assistant.id}")
            return assistant
        except Exception as e:
            logger.error(f"Failed to create assistant in Azure: {e}")
            raise Exception(f"Failed to create assistant in Azure: {e}")

    def update_assistant(
        self,
        assistant_id: str,
        model: Optional[str] = None,
        name: Optional[str] = None,
        instructions: Optional[str] = None,
        description: Optional[str] = None,
        tools: Optional[list] = None,
        tool_resources: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Any:
        """
        Update an assistant in Azure OpenAI.

        Args:
            assistant_id: ID of the assistant to update
            model: Model name (optional)
            name: Assistant name (optional)
            instructions: System instructions (optional)
            description: Description (optional)
            tools: List of tools (optional)
            tool_resources: Tool resources (optional)
            metadata: Metadata (optional)
            temperature: Temperature parameter (for GPT-4 models)
            reasoning_effort: Reasoning effort parameter (for GPT-5 models)

        Returns:
            Updated Assistant object
        """
        update_params: Dict[str, Any] = {
            "assistant_id": assistant_id,
        }

        if model:
            update_params["model"] = model
        if name:
            update_params["name"] = name
        if instructions:
            update_params["instructions"] = instructions
        if description:
            update_params["description"] = description
        if tools:
            update_params["tools"] = tools
        if tool_resources:
            update_params["tool_resources"] = tool_resources
        if metadata:
            update_params["metadata"] = metadata

        # Set model-specific parameters
        if model:
            model_name_clean = model.lower().strip()
            if model_name_clean.startswith("gpt-4") and not model_name_clean.startswith("gpt-5"):
                if temperature is not None:
                    update_params["temperature"] = temperature
                elif "temperature" not in update_params:
                    update_params["temperature"] = 0.4
                logger.info(f"Using temperature={update_params['temperature']} for model: {model}")
            else:
                if reasoning_effort:
                    update_params["reasoning_effort"] = reasoning_effort
                elif "reasoning_effort" not in update_params:
                    update_params["reasoning_effort"] = "low"
                logger.info(f"Using reasoning_effort={update_params['reasoning_effort']} for model: {model}")

        try:
            assistant = self.client.beta.assistants.update(**update_params)
            logger.info(f"Successfully updated assistant: {assistant.id}")
            return assistant
        except Exception as e:
            logger.error(f"Failed to update assistant in Azure: {e}")
            raise Exception(f"Failed to update assistant in Azure: {e}")

    def retrieve_assistant(self, assistant_id: str) -> Any:
        """
        Retrieve an assistant by ID.

        Args:
            assistant_id: ID of the assistant to retrieve

        Returns:
            Assistant object
        """
        try:
            assistant = self.client.beta.assistants.retrieve(assistant_id)
            return assistant
        except Exception as e:
            logger.error(f"Failed to retrieve assistant from Azure: {e}")
            raise Exception(f"Failed to retrieve assistant from Azure: {e}")

    def delete_assistant(self, assistant_id: str) -> Dict[str, Any]:
        """
        Delete an assistant by ID.

        Args:
            assistant_id: ID of the assistant to delete

        Returns:
            Deletion status object
        """
        try:
            deletion_status = self.client.beta.assistants.delete(assistant_id)
            logger.info(f"Successfully deleted assistant: {assistant_id}")
            return deletion_status
        except Exception as e:
            logger.error(f"Failed to delete assistant from Azure: {e}")
            raise Exception(f"Failed to delete assistant from Azure: {e}")

    def list_assistants(self, limit: Optional[int] = None, order: Optional[str] = None, after: Optional[str] = None, before: Optional[str] = None) -> Any:
        """
        List assistants.

        Args:
            limit: Maximum number of assistants to return
            order: Sort order ('asc' or 'desc')
            after: Cursor for pagination
            before: Cursor for pagination

        Returns:
            List of assistants
        """
        list_params: Dict[str, Any] = {}
        if limit:
            list_params["limit"] = limit
        if order:
            list_params["order"] = order
        if after:
            list_params["after"] = after
        if before:
            list_params["before"] = before

        try:
            assistants = self.client.beta.assistants.list(**list_params)
            return assistants
        except Exception as e:
            logger.error(f"Failed to list assistants from Azure: {e}")
            raise Exception(f"Failed to list assistants from Azure: {e}")


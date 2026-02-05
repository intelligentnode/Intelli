import inspect
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _check_azure_imports():
    try:
        from azure.identity import DefaultAzureCredential
        from azure.ai.projects import AIProjectClient
        return DefaultAzureCredential, AIProjectClient
    except ImportError:
        raise ImportError(
            "azure-ai-projects and azure-identity are required for Azure Agent support. "
            "Install them with: pip install azure-ai-projects azure-identity"
        )


class AzureAgentWrapper:
    """
    Wrapper for Azure AI Foundry Agents API.
    Provides methods to create, update, delete, and run agents.
    """

    @staticmethod
    def _extract_endpoint(conn_str: str) -> Optional[str]:
        if not conn_str:
            return None
        if conn_str.startswith("https://"):
            return conn_str
        parts = [part.strip() for part in conn_str.split(";") if part.strip()]
        for part in parts:
            if part.lower().startswith("endpoint="):
                return part.split("=", 1)[1].strip()
        return None

    def __init__(
        self,
        connection_string: Optional[str] = None,
        retry_attempts: int = 3,
        retry_delay_seconds: float = 0.5,
        timeout: Optional[float] = None,
    ):
        """
        Initialize Azure AI Foundry project client.

        Args:
            connection_string: Azure AI Foundry Project connection string.
                               Defaults to AZURE_PROJECT_CONNECTION_STRING env var.
        """
        conn_str = connection_string or os.getenv("AZURE_PROJECT_CONNECTION_STRING")
        if not conn_str:
            raise ValueError("AZURE_PROJECT_CONNECTION_STRING is required")

        DefaultAzureCredential, AIProjectClient = _check_azure_imports()
        credential = DefaultAzureCredential()
        if hasattr(AIProjectClient, "from_connection_string"):
            self.client = AIProjectClient.from_connection_string(
                credential=credential,
                conn_str=conn_str,
            )
        else:
            endpoint = self._extract_endpoint(conn_str)
            if not endpoint:
                raise ValueError(
                    "AZURE_PROJECT_CONNECTION_STRING must include an endpoint or be an endpoint URL "
                    "when the SDK lacks from_connection_string support."
                )
            self.client = AIProjectClient(
                endpoint=endpoint,
                credential=credential,
            )
        self._openai_client = None
        self._retry_attempts = max(1, retry_attempts)
        self._retry_delay_seconds = max(0.0, retry_delay_seconds)
        self._timeout = timeout
        if not hasattr(self.client.agents, "create_version") or not hasattr(
            self.client, "get_openai_client"
        ):
            raise RuntimeError(
                "Azure Agent Wrapper requires the new Agents API. "
                "Please upgrade azure-ai-projects to a version that supports agents.create_version "
                "and project_client.get_openai_client()."
            )

    def _get_openai_client(self):
        if self._openai_client is None:
            self._openai_client = self.client.get_openai_client()
        return self._openai_client

    def _coerce_agent_version(self, version: Any) -> Optional[int]:
        if version is None:
            return None
        if isinstance(version, int):
            return version
        if isinstance(version, str):
            stripped = version.strip()
            if stripped.isdigit():
                return int(stripped)
            raise ValueError("agent version must be an integer.")
        raise ValueError("agent version must be an integer.")

    def _parse_agent_id(self, agent_id: str):
        if ":" in agent_id:
            name, version = agent_id.rsplit(":", 1)
            return name, self._coerce_agent_version(version)
        return agent_id, None

    def _resolve_agent_reference(self, agent: Any) -> Dict[str, str]:
        if isinstance(agent, str):
            name, version = self._parse_agent_id(agent)
        elif isinstance(agent, dict):
            name = agent.get("name")
            version = agent.get("version")
        else:
            name = getattr(agent, "name", None)
            version = getattr(agent, "version", None)

        if name is None or version is None:
            raise ValueError(
                "agent reference must include name and version. "
                "Provide a '<name>:<version>' string or an object with name/version."
            )
        coerced_version = self._coerce_agent_version(version)
        return {"name": name, "version": str(coerced_version)}

    def _call_agents_method(self, method, *args, **kwargs):
        if not kwargs:
            return method(*args)
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            try:
                return method(*args, **kwargs)
            except TypeError:
                return method(*args)

        params = signature.parameters
        supports_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
        )
        if supports_kwargs:
            return method(*args, **kwargs)

        filtered_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in params and params[key].kind != inspect.Parameter.POSITIONAL_ONLY
        }
        if filtered_kwargs:
            return method(*args, **filtered_kwargs)
        return method(*args)

    def _with_retry(self, label: str, func):
        last_error = None
        for attempt in range(1, self._retry_attempts + 1):
            try:
                return func()
            except ValueError:
                raise
            except TypeError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt < self._retry_attempts:
                    backoff = self._retry_delay_seconds * (2 ** (attempt - 1))
                    jitter = random.uniform(0, backoff * 0.2)
                    time.sleep(backoff + jitter)
        raise last_error

    def _with_timeout(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if self._timeout is None or "timeout" in kwargs:
            return kwargs
        return {**kwargs, "timeout": self._timeout}

    def _normalize_response_format(self, response_format: Any) -> Optional[Dict[str, str]]:
        if response_format is None:
            return None
        if isinstance(response_format, dict):
            value = response_format.get("type")
            if not isinstance(value, str):
                raise ValueError("response_format dict must include a string 'type'.")
            if value not in ("text", "json_object"):
                raise ValueError("response_format must be 'text' or 'json_object'.")
            return response_format
        if isinstance(response_format, str):
            if response_format not in ("text", "json_object"):
                raise ValueError("response_format must be 'text' or 'json_object'.")
            return {"type": response_format}
        raise ValueError("response_format must be a string or dict with a 'type' key.")

    def _extract_response_id(self, response: Any) -> Optional[str]:
        if response is None:
            return None
        response_id = getattr(response, "id", None)
        if response_id is None and hasattr(response, "get"):
            response_id = response.get("id")
        return response_id

    def _extract_response_status(self, response: Any) -> Optional[str]:
        if response is None:
            return None
        status = getattr(response, "status", None)
        if status is None and hasattr(response, "get"):
            status = response.get("status")
        return status

    def _require_conversations_client(self):
        client = self._get_openai_client()
        conversations = getattr(client, "conversations", None)
        if conversations is None:
            raise RuntimeError("OpenAI client does not expose conversations API.")
        return conversations

    def create_agent(
        self,
        model: str,
        name: str,
        instructions: str,
        description: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_resources: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        response_format: Optional[Any] = None,
    ) -> Any:
        """
        Create an agent.
        """
        try:
            definition: Dict[str, Any] = {
                "kind": "prompt",
                "model": model,
                "instructions": instructions,
            }
            if description is not None:
                definition["description"] = description
            if temperature is not None:
                if not isinstance(temperature, (int, float)):
                    raise ValueError("temperature must be a number.")
                definition["temperature"] = temperature
            if top_p is not None:
                if not isinstance(top_p, (int, float)):
                    raise ValueError("top_p must be a number.")
                definition["top_p"] = top_p
            normalized_response_format = self._normalize_response_format(response_format)
            if normalized_response_format is not None:
                definition["response_format"] = normalized_response_format
            if tools is not None:
                if not isinstance(tools, list):
                    raise ValueError("tools must be a list of dicts.")
                if any(not isinstance(tool, dict) for tool in tools):
                    raise ValueError("tools must be a list of dicts.")
                definition["tools"] = tools
            if tool_resources is not None:
                if not isinstance(tool_resources, dict):
                    raise ValueError("tool_resources must be a dict.")
                definition["tool_resources"] = tool_resources
            if metadata is not None and not isinstance(metadata, dict):
                raise ValueError("metadata must be a dict of string keys/values.")
            agent_kwargs = {
                "agent_name": name,
                "definition": definition,
                **self._with_timeout({}),
            }
            if metadata is not None:
                agent_kwargs["metadata"] = metadata
            agent = self._with_retry(
                "create_agent",
                lambda: self._call_agents_method(
                    self.client.agents.create_version,
                    **agent_kwargs,
                ),
            )
            logger.info(f"Successfully created agent version: {agent.name}:{agent.version}")
            return agent
        except Exception:
            logger.exception("Failed to create agent in Azure.")
            raise

    def get_agent(self, agent_id: str) -> Any:
        """
        Retrieve an agent by ID.
        """
        try:
            if not hasattr(self.client.agents, "get_version"):
                raise RuntimeError("SDK does not support agents.get_version.")
            name, version = self._parse_agent_id(agent_id)
            if version is None:
                raise ValueError("agent_id must include version as '<name>:<version>'.")
            return self._with_retry(
                "get_agent",
                lambda: self._call_agents_method(
                    self.client.agents.get_version,
                    agent_name=name,
                    agent_version=version,
                    **self._with_timeout({}),
                ),
            )
        except Exception:
            logger.exception("Failed to retrieve agent from Azure.")
            raise

    def update_agent(
        self,
        agent_id: str,
        model: Optional[str] = None,
        instructions: Optional[str] = None,
        description: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_resources: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        response_format: Optional[Any] = None,
    ) -> Any:
        """
        Update an agent.
        """
        try:
            if not hasattr(self.client.agents, "get_version"):
                raise RuntimeError("SDK does not support agents.get_version.")
            agent_name, version = self._parse_agent_id(agent_id)
            if version is None:
                raise ValueError("agent_id must include version as '<name>:<version>'.")
            has_updates = any(
                value is not None
                for value in (
                    model,
                    instructions,
                    description,
                    tools,
                    tool_resources,
                    metadata,
                    temperature,
                    top_p,
                    response_format,
                )
            )
            if not has_updates:
                raise ValueError("No definition fields provided for new agent version update.")
            definition: Dict[str, Any] = {}

            current = self._with_retry(
                "get_agent",
                lambda: self._call_agents_method(
                    self.client.agents.get_version,
                    agent_name=agent_name,
                    agent_version=version,
                    **self._with_timeout({}),
                ),
            )
            current_def = getattr(current, "definition", None)
            if current_def is None and hasattr(current, "get"):
                current_def = current.get("definition")
            if current_def:
                if isinstance(current_def, dict):
                    definition.update(current_def)
                else:
                    converted_def = None
                    for attr_name in ("as_dict", "to_dict", "model_dump", "dict"):
                        converter = getattr(current_def, attr_name, None)
                        if callable(converter):
                            converted_def = converter()
                            break
                    if converted_def is None and hasattr(current_def, "__dict__"):
                        converted_def = current_def.__dict__
                    if not isinstance(converted_def, dict):
                        raise ValueError("agent definition must be a dict or convertible to dict.")
                    definition.update(converted_def)

            if model is not None:
                definition["model"] = model
            if instructions is not None:
                definition["instructions"] = instructions
            if description is not None:
                definition["description"] = description
            if tools is not None:
                if not isinstance(tools, list):
                    raise ValueError("tools must be a list of dicts.")
                if any(not isinstance(tool, dict) for tool in tools):
                    raise ValueError("tools must be a list of dicts.")
                definition["tools"] = tools
            if tool_resources is not None:
                if not isinstance(tool_resources, dict):
                    raise ValueError("tool_resources must be a dict.")
                definition["tool_resources"] = tool_resources
            if temperature is not None:
                if not isinstance(temperature, (int, float)):
                    raise ValueError("temperature must be a number.")
                definition["temperature"] = temperature
            if top_p is not None:
                if not isinstance(top_p, (int, float)):
                    raise ValueError("top_p must be a number.")
                definition["top_p"] = top_p
            normalized_response_format = self._normalize_response_format(response_format)
            if normalized_response_format is not None:
                definition["response_format"] = normalized_response_format
            if "kind" not in definition:
                definition["kind"] = "prompt"
            if metadata is not None and not isinstance(metadata, dict):
                raise ValueError("metadata must be a dict of string keys/values.")

            agent_kwargs = {
                "agent_name": agent_name,
                "definition": definition,
                **self._with_timeout({}),
            }
            if metadata is not None:
                agent_kwargs["metadata"] = metadata
            agent = self._with_retry(
                "update_agent",
                lambda: self._call_agents_method(
                    self.client.agents.create_version,
                    **agent_kwargs,
                ),
            )
            logger.info(f"Successfully created new agent version: {agent.name}:{agent.version}")
            return agent
        except Exception:
            logger.exception("Failed to update agent in Azure.")
            raise

    def delete_agent(self, agent_id: str) -> Any:
        """
        Delete an agent by ID.
        """
        try:
            if not hasattr(self.client.agents, "delete_version"):
                raise RuntimeError("SDK does not support agents.delete_version.")
            name, version = self._parse_agent_id(agent_id)
            if version is None:
                raise ValueError("agent_id must include version as '<name>:<version>'.")
            deletion_status = self._with_retry(
                "delete_agent",
                lambda: self._call_agents_method(
                    self.client.agents.delete_version,
                    agent_name=name,
                    agent_version=version,
                    **self._with_timeout({}),
                ),
            )
            logger.info(f"Successfully deleted agent version: {name}:{version}")
            return deletion_status
        except Exception:
            logger.exception("Failed to delete agent from Azure.")
            raise

    def list_agents(self, agent_name: str, as_list: bool = False) -> Any:
        """
        List versions for an agent name.
        """
        try:
            if not hasattr(self.client.agents, "list_versions"):
                raise RuntimeError("SDK does not support agents.list_versions.")
            result = self._with_retry(
                "list_agents",
                lambda: self._call_agents_method(
                    self.client.agents.list_versions,
                    agent_name=agent_name,
                    **self._with_timeout({}),
                ),
            )
            return list(result) if as_list else result
        except Exception:
            logger.exception("Failed to list agents from Azure.")
            raise

    def create_conversation(
        self,
        items: List[Dict[str, Any]],
        metadata: Optional[Dict[str, str]] = None,
    ) -> Any:
        """
        Create a conversation (new Agents API).
        """
        try:
            conversations = self._require_conversations_client()
            params: Dict[str, Any] = {"items": items}
            if metadata:
                params["metadata"] = metadata
            return self._with_retry(
                "create_conversation",
                lambda: conversations.create(**self._with_timeout(params)),
            )
        except Exception:
            logger.exception("Failed to create conversation in Azure.")
            raise

    def update_conversation_metadata(
        self,
        conversation_id: str,
        metadata: Dict[str, str],
    ) -> Any:
        """
        Update conversation metadata (new Agents API).
        """
        try:
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be a dict of string keys/values.")
            conversations = self._require_conversations_client()
            if not hasattr(conversations, "update"):
                raise RuntimeError("SDK does not support conversations.update.")
            return self._with_retry(
                "update_conversation_metadata",
                lambda: self._call_agents_method(
                    conversations.update,
                    conversation_id=conversation_id,
                    metadata=metadata,
                    **self._with_timeout({}),
                ),
            )
        except Exception:
            logger.exception("Failed to update conversation metadata in Azure.")
            raise

    def delete_conversation(self, conversation_id: str) -> Any:
        """
        Delete a conversation (new Agents API).
        """
        try:
            conversations = self._require_conversations_client()
            if not hasattr(conversations, "delete"):
                raise RuntimeError("SDK does not support conversations.delete.")
            return self._with_retry(
                "delete_conversation",
                lambda: self._call_agents_method(
                    conversations.delete,
                    conversation_id,
                    **self._with_timeout({}),
                ),
            )
        except Exception:
            logger.exception("Failed to delete conversation in Azure.")
            raise

    def create_response(
        self,
        conversation_id: str,
        agent: Any,
        input_text: Optional[str] = None,
        input_items: Optional[List[Dict[str, Any]]] = None,
        wait_for_completion: bool = True,
        poll_interval_seconds: float = 0.5,
        timeout_seconds: Optional[float] = None,
        return_on_requires_action: bool = True,
    ) -> Any:
        """
        Create a response for a conversation (new Agents API).
        """
        try:
            if input_items is None:
                if input_text is None:
                    raise ValueError(
                        "Either input_text or input_items must be provided. "
                        "If both are supplied, input_items takes precedence."
                    )
                input_items = [{"type": "message", "role": "user", "content": input_text}]
            else:
                if not isinstance(input_items, list):
                    raise ValueError("input_items must be a list of message dicts.")
                normalized_items: List[Dict[str, Any]] = []
                for item in input_items:
                    if not isinstance(item, dict):
                        raise ValueError("input_items must be a list of message dicts.")
                    if isinstance(item, dict) and "type" not in item and "role" in item and "content" in item:
                        normalized_items.append({"type": "message", **item})
                    elif isinstance(item, dict) and "type" not in item:
                        raise ValueError("input_items dicts must include 'type' or message fields.")
                    else:
                        normalized_items.append(item)
                input_items = normalized_items
            agent_ref = self._resolve_agent_reference(agent)
            extra_body = {"agent": {"type": "agent_reference", **agent_ref}}
            payload = {
                "conversation": conversation_id,
                "input": input_items,
                "extra_body": extra_body,
            }
            client = self._get_openai_client()
            response = self._with_retry(
                "create_response",
                lambda: client.responses.create(**self._with_timeout(payload)),
            )
            if not wait_for_completion:
                return response
            response_id = self._extract_response_id(response)
            if not response_id:
                return response
            return self.wait_for_response(
                response_id=response_id,
                poll_interval_seconds=poll_interval_seconds,
                timeout_seconds=timeout_seconds,
                return_on_requires_action=return_on_requires_action,
            )
        except Exception:
            logger.exception("Failed to create response in Azure.")
            raise

    def submit_tool_outputs(
        self,
        response_id: str,
        tool_outputs: List[Dict[str, Any]],
        wait_for_completion: bool = True,
        poll_interval_seconds: float = 0.5,
        timeout_seconds: Optional[float] = None,
        return_on_requires_action: bool = True,
    ) -> Any:
        """
        Submit tool outputs for a response in requires_action state.
        """
        try:
            if not response_id:
                raise ValueError("response_id is required.")
            if not isinstance(tool_outputs, list):
                raise ValueError("tool_outputs must be a list of dicts.")
            if any(not isinstance(item, dict) for item in tool_outputs):
                raise ValueError("tool_outputs must be a list of dicts.")
            client = self._get_openai_client()
            responses = getattr(client, "responses", None)
            if responses is None or not hasattr(responses, "submit_tool_outputs"):
                raise RuntimeError("SDK does not support responses.submit_tool_outputs.")
            response = self._with_retry(
                "submit_tool_outputs",
                lambda: self._call_agents_method(
                    responses.submit_tool_outputs,
                    response_id=response_id,
                    tool_outputs=tool_outputs,
                    **self._with_timeout({}),
                ),
            )
            if not wait_for_completion:
                return response
            response_id = self._extract_response_id(response) or response_id
            return self.wait_for_response(
                response_id=response_id,
                poll_interval_seconds=poll_interval_seconds,
                timeout_seconds=timeout_seconds,
                return_on_requires_action=return_on_requires_action,
            )
        except Exception:
            logger.exception("Failed to submit tool outputs in Azure.")
            raise

    def get_response(self, response_id: str) -> Any:
        """
        Retrieve a response by ID (new Agents API).
        """
        try:
            client = self._get_openai_client()
            return self._with_retry(
                "get_response",
                lambda: self._call_agents_method(
                    client.responses.retrieve,
                    response_id,
                    **self._with_timeout({}),
                ),
            )
        except Exception:
            logger.exception("Failed to retrieve response from Azure.")
            raise

    def wait_for_response(
        self,
        response_id: str,
        poll_interval_seconds: float = 0.5,
        timeout_seconds: Optional[float] = None,
        return_on_requires_action: bool = True,
    ) -> Any:
        """
        Poll a response until completion or terminal status.
        """
        start = time.time()
        while True:
            response = self.get_response(response_id)
            status = self._extract_response_status(response)
            if status in ("completed", "failed", "cancelled"):
                return response
            if status == "requires_action" and return_on_requires_action:
                return response
            if timeout_seconds is not None and (time.time() - start) >= timeout_seconds:
                raise TimeoutError(
                    f"Timed out waiting for response {response_id} after {timeout_seconds} seconds."
                )
            time.sleep(max(0.05, poll_interval_seconds))

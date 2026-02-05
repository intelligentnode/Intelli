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
            self.client = AIProjectClient(
                endpoint=conn_str,
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

    def _parse_agent_id(self, agent_id: str):
        if ":" in agent_id:
            name, version = agent_id.rsplit(":", 1)
            return name, version
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

        if not name or not version:
            raise ValueError(
                "agent reference must include name and version. "
                "Provide a '<name>:<version>' string or an object with name/version."
            )
        return {"name": name, "version": version}

    def _call_agents_method(self, method, *args, **kwargs):
        try:
            return method(**kwargs)
        except TypeError:
            return method(*args)

    def _with_retry(self, label: str, func):
        last_error = None
        for attempt in range(1, self._retry_attempts + 1):
            try:
                return func()
            except ValueError:
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

    def create_agent(
        self,
        model: str,
        name: str,
        instructions: str,
        description: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
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
            if tools:
                definition["tools"] = tools
            agent = self._call_agents_method(
                self.client.agents.create_version,
                agent_name=name,
                definition=definition,
                **self._with_timeout({}),
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
            if not version:
                raise ValueError("agent_id must include version as '<name>:<version>'.")
            return self._with_retry(
                "get_agent",
                lambda: self._call_agents_method(
                    self.client.agents.get_version,
                    name,
                    version,
                    agent_name=name,
                    version=version,
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
    ) -> Any:
        """
        Update an agent.
        """
        try:
            has_updates = any(
                value is not None for value in (model, instructions, description, tools)
            )
            if not has_updates:
                raise ValueError("No definition fields provided for new agent version update.")
            if not hasattr(self.client.agents, "get_version"):
                raise RuntimeError("SDK does not support agents.get_version.")
            agent_name, version = self._parse_agent_id(agent_id)
            if not version:
                raise ValueError("agent_id must include version as '<name>:<version>'.")
            definition: Dict[str, Any] = {}

            current = self._call_agents_method(
                self.client.agents.get_version,
                agent_name,
                version,
                agent_name=agent_name,
                version=version,
            )
            current_def = getattr(current, "definition", None)
            if current_def is None and hasattr(current, "get"):
                current_def = current.get("definition")
            if current_def:
                definition.update(current_def)

            if model is not None:
                definition["model"] = model
            if instructions is not None:
                definition["instructions"] = instructions
            if description is not None:
                definition["description"] = description
            if tools is not None:
                definition["tools"] = tools
            if "kind" not in definition:
                definition["kind"] = "prompt"

            agent = self._call_agents_method(
                self.client.agents.create_version,
                agent_name=agent_name,
                definition=definition,
                **self._with_timeout({}),
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
            if not version:
                raise ValueError("agent_id must include version as '<name>:<version>'.")
            deletion_status = self._with_retry(
                "delete_agent",
                lambda: self._call_agents_method(
                    self.client.agents.delete_version,
                    name,
                    version,
                    agent_name=name,
                    version=version,
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
                    agent_name,
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
            client = self._get_openai_client()
            params: Dict[str, Any] = {"items": items}
            if metadata:
                params["metadata"] = metadata
            return self._with_retry(
                "create_conversation",
                lambda: client.conversations.create(**self._with_timeout(params)),
            )
        except Exception:
            logger.exception("Failed to create conversation in Azure.")
            raise

    def create_response(
        self,
        conversation_id: str,
        agent: Any,
        input_text: Optional[str] = None,
        input_items: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        """
        Create a response for a conversation (new Agents API).
        """
        try:
            client = self._get_openai_client()
            if input_items is None:
                if input_text is None:
                    raise ValueError("Either input_text or input_items must be provided.")
                input_items = [{"type": "message", "role": "user", "content": input_text}]
            else:
                normalized_items: List[Dict[str, Any]] = []
                for item in input_items:
                    if isinstance(item, dict) and "type" not in item and "role" in item and "content" in item:
                        normalized_items.append({"type": "message", **item})
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
            return self._with_retry(
                "create_response",
                lambda: client.responses.create(**self._with_timeout(payload)),
            )
        except Exception:
            logger.exception("Failed to create response in Azure.")
            raise

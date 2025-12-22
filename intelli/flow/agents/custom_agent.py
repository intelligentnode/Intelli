from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from intelli.flow.agents.agent import BasicAgent
from intelli.flow.types import AgentTypes


class CustomAgent(BasicAgent, ABC):
    """
    Base class for end-users to implement custom agents that plug into Intelli Flow.

    Requirements to work with `Task`/`Flow`:
    - Attributes: `type`, `provider`, `mission`, `model_params`, `options`
    - Method: `execute(agent_input, new_params={})`

    Notes:
    - `type` must be one of AgentTypes values (e.g. "text", "search", "mcp", ...).
    - `execute(...)` should return either a string (for text output) or the relevant
      output for the agent type; `Task` will handle joining/selection logic.
    """

    def __init__(
        self,
        agent_type: str,
        provider: str = "custom",
        mission: str = "",
        model_params: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        if agent_type not in AgentTypes._value2member_map_:
            raise ValueError(
                f"Incorrect agent type '{agent_type}'. Accepted types: "
                f"{list(AgentTypes._value2member_map_.keys())}"
            )

        self.type = agent_type
        self.provider = provider
        self.mission = mission
        self.model_params = model_params or {}
        self.options = options or {}

    def update_model_params(self, updates: Dict[str, Any]) -> None:
        """Convenience helper to update model params in-place."""
        if updates:
            self.model_params.update(updates)

    @abstractmethod
    def execute(self, agent_input, new_params: Optional[Dict[str, Any]] = None):
        """
        Execute your custom agent.

        Args:
            agent_input: Usually `TextAgentInput`/`ImageAgentInput` etc (see `flow/input/agent_input.py`)
            new_params: Params provided by `Task` (task.model_params) to override/extend
                       this agent's `model_params`.
        """
        raise NotImplementedError



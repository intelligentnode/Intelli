"""
Public API for the intelli.flow package.
This file aggregates key classes and functions from submodules for easier access.
"""

# Agents
from intelli.flow.agents.agent import Agent
from intelli.flow.agents.kagent import KerasAgent
from intelli.flow.agents.handlers import get_agent_handler

# Input types for tasks and agents
from intelli.flow.input.task_input import TaskInput, TextTaskInput, ImageTaskInput
from intelli.flow.input.agent_input import AgentInput, TextAgentInput, ImageAgentInput

# Processors and templates
from intelli.flow.processors.basic_processor import TextProcessor
from intelli.flow.template.basic_template import TextInputTemplate

# Core flow
from intelli.flow.sequence_flow import SequenceFlow
from intelli.flow.tasks.task import Task
from intelli.flow.flow import Flow

# Dynamic routing
from intelli.flow.dynamic_connector import DynamicConnector, ConnectorMode
from intelli.flow.tool_connector import ToolDynamicConnector

# Types
from intelli.flow.types import AgentTypes, InputTypes

# Additional utilities
from intelli.flow.utils.flow_helper import FlowHelper
from intelli.flow.store.memory import Memory
from intelli.flow.store.dbmemory import DBMemory

try:
    from intelli.mcp import (
        MCPServerBuilder,
        MCPJSONExtractor,
        create_mcp_preprocessor
    )
except ImportError:
    # MCP utilities may not be available if MCP isn't installed
    pass
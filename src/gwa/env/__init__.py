"""Deterministic CRM environment, tools, tasks, and verifiable rewards."""

from .crm_env import CRMEnv, Observation
from .reward import RewardConfig, dense_reward, episode_return, terminal_reward
from .state import CATEGORY_REMEDY, CRMState, KBArticle, Ticket, make_state
from .tasks import FAMILIES, Task, generate_split, load_tasks, make_task
from .tools import TOOL_NAMES, ToolCall, ToolResult, execute_tool

__all__ = [
    "CRMEnv",
    "Observation",
    "RewardConfig",
    "dense_reward",
    "episode_return",
    "terminal_reward",
    "CATEGORY_REMEDY",
    "CRMState",
    "KBArticle",
    "Ticket",
    "make_state",
    "FAMILIES",
    "Task",
    "generate_split",
    "load_tasks",
    "make_task",
    "TOOL_NAMES",
    "ToolCall",
    "ToolResult",
    "execute_tool",
]

"""
Simple Agent Base Class
MCP-style communication - each agent receives state, returns updated state.
"""
from typing import Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import traceback
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')

@dataclass
class AgentResult:
    success: bool
    data: dict = field(default_factory=dict)
    message: str = ""
    metrics: dict = field(default_factory=dict)

class BaseAgent:
    def __init__(self, name: str):
        self.name = name
        self.logger = []

    def run(self, state: dict) -> AgentResult:
        raise NotImplementedError

    def log(self, message: str):
        timestamp = datetime.now().isoformat()
        entry = f"[{timestamp}] {self.name}: {message}"
        self.logger.append(entry)
        logging.info(f"[{self.name}] {message}")

    def get_logs(self) -> list:
        return self.logger

    def safe_execute(self, fn: Callable, state: dict, error_message: str = "Agent error") -> AgentResult:
        try:
            return fn(state)
        except Exception as e:
            self.log(f"{error_message}: {e}\n{traceback.format_exc()}")
            return AgentResult(success=False, data={}, message=f"{error_message}: {e}")

    @staticmethod
    def validate_json_schema(data: dict, required_fields: list) -> (bool, str):
        missing = [f for f in required_fields if f not in data]
        if missing:
            return False, f"Missing required fields: {missing}"
        return True, ""

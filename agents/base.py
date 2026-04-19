"""
Simple Agent Base Class
MCP-style communication - each agent receives state, returns updated state.
"""
from typing import Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json

@dataclass
class AgentResult:
    """Standard return type for all agents"""
    success: bool
    data: dict = field(default_factory=dict)
    message: str = ""
    metrics: dict = field(default_factory=dict)

class BaseAgent:
    """Base class for all agents in the MAPE-K pipeline"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = []
    
    def run(self, state: dict) -> AgentResult:
        """Main entry point - override in subclass"""
        raise NotImplementedError
    
    def log(self, message: str):
        """Log agent activity"""
        timestamp = datetime.now().isoformat()
        entry = f"[{timestamp}] {self.name}: {message}"
        self.logger.append(entry)
        print(entry)
    
    def get_logs(self) -> list:
        return self.logger


class SimplePipeline:
    """Simple MCP-style pipeline - no LangGraph needed"""
    
    def __init__(self):
        self.agents = {}
        self.state = {}
    
    def register(self, name: str, agent: BaseAgent):
        """Register an agent"""
        self.agents[name] = agent
    
    def run(self, initial_state: dict = None) -> dict:
        """Run all registered agents in sequence"""
        self.state = initial_state or {}
        
        for name, agent in self.agents.items():
            self.log(f"Running {name}...")
            result = agent.run(self.state)
            
            if not result.success:
                self.log(f"Agent {name} failed: {result.message}")
                break
            
            # Merge result into state
            self.state.update(result.data)
        
        return self.state
    
    def log(self, message: str):
        print(f"[Pipeline] {message}")
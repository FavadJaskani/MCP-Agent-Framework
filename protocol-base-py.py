from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from mcp.utils.logging import get_logger

logger = get_logger(__name__)

class Protocol(ABC):
    """Base class for all agent communication protocols."""
    
    def __init__(self, name: str = None, description: str = None):
        self.name = name or self.__class__.__name__
        self.description = description or f"Protocol: {self.name}"
        self.agent = None  # The agent this protocol is attached to
    
    def register_agent(self, agent) -> None:
        """Register the agent that will use this protocol."""
        self.agent = agent
        logger.debug(f"Protocol '{self.name}' registered with agent '{agent.name}'")
    
    @abstractmethod
    def can_handle_message(self, message) -> bool:
        """Check if this protocol can handle the given message."""
        pass
    
    @abstractmethod
    def process_message(self, message) -> Optional[Any]:
        """Process a message according to this protocol."""
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this protocol."""
        return {
            "name": self.name,
            "description": self.description
        }
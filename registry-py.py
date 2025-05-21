from typing import Dict, Optional, List

from mcp.utils.logging import get_logger

logger = get_logger(__name__)

class AgentRegistry:
    """Registry for tracking available agents."""
    
    def __init__(self):
        self.agents = {}  # Dict[agent_id, agent]
    
    def register_agent(self, agent) -> None:
        """Register a new agent."""
        self.agents[agent.id] = agent
        logger.info(f"Agent '{agent.name}' registered with ID {agent.id}")
    
    def unregister_agent(self, agent_id: str) -> None:
        """Remove an agent from the registry."""
        if agent_id in self.agents:
            agent = self.agents.pop(agent_id)
            logger.info(f"Agent '{agent.name}' unregistered")
    
    def get_agent(self, agent_id: str) -> Optional[any]:
        """Get an agent by ID."""
        return self.agents.get(agent_id)
    
    def get_agents_by_name(self, name: str) -> List[any]:
        """Get all agents with a specific name."""
        return [agent for agent in self.agents.values() if agent.name == name]
    
    def get_all_agents(self) -> List[any]:
        """Get all registered agents."""
        return list(self.agents.values())
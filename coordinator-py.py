from typing import List, Dict, Any, Optional
from uuid import uuid4

from mcp.core.agent import Agent
from mcp.core.message import Message
from mcp.core.registry import AgentRegistry
from mcp.utils.logging import get_logger

logger = get_logger(__name__)

class Coordinator:
    """Manages communication and coordination between multiple agents."""
    
    def __init__(self, agents: List[Agent] = None):
        self.id = str(uuid4())
        self.registry = AgentRegistry()
        self.conversations = {}  # Store ongoing conversations
        
        # Register agents
        if agents:
            for agent in agents:
                self.registry.register_agent(agent)
    
    def execute_task(self, task_description: str) -> Dict[str, Any]:
        """Execute a complex task using the registered agents."""
        logger.info(f"Starting task execution: {task_description[:50]}...")
        
        # Create a new conversation
        conversation_id = str(uuid4())
        self.conversations[conversation_id] = {
            "description": task_description,
            "messages": [],
            "status": "in_progress",
            "result": None
        }
        
        # Simple approach: Find an agent that can handle planning
        planner_agent = self._find_planner_agent()
        if not planner_agent:
            logger.warning("No planning agent found, using first available agent")
            planner_agent = next(iter(self.registry.agents.values()))
        
        # Send the initial task to the planner
        initial_message = Message(
            sender_id=self.id,
            receiver_id=planner_agent.id,
            content=task_description,
            message_type="task",
            conversation_id=conversation_id
        )
        
        # Start the task execution
        self._process_message_chain(initial_message)
        
        # Return the conversation results
        return {
            "conversation_id": conversation_id,
            "status": self.conversations[conversation_id]["status"],
            "result": self.conversations[conversation_id]["result"]
        }
    
    def _process_message_chain(self, message: Message, max_steps: int = 10) -> None:
        """Process a chain of messages until completion or max steps reached."""
        conversation = self.conversations[message.conversation_id]
        conversation["messages"].append(message.to_dict())
        
        steps = 0
        current_message = message
        
        while steps < max_steps:
            # Get the receiving agent
            receiver = self.registry.get_agent(current_message.receiver_id)
            if not receiver:
                logger.error(f"Agent {current_message.receiver_id} not found")
                break
            
            # Process the message
            response = receiver.receive_message(current_message)
            if not response:
                logger.debug(f"No response from agent {receiver.name}")
                break
            
            # Store the response
            conversation["messages"].append(response.to_dict())
            
            # Check if this is the final response to the coordinator
            if response.receiver_id == self.id:
                conversation["status"] = "completed"
                conversation["result"] = response.content
                logger.info(f"Task completed: {response.content[:50]}...")
                break
            
            # Continue the chain
            current_message = response
            steps += 1
        
        # Check if we hit the maximum steps
        if steps >= max_steps:
            logger.warning(f"Max steps reached for conversation {message.conversation_id}")
            conversation["status"] = "max_steps_reached"
    
    def _find_planner_agent(self) -> Optional[Agent]:
        """Find an agent that can handle planning."""
        for agent in self.registry.agents.values():
            for protocol in agent.protocols:
                if "planning" in protocol.__class__.__name__.lower():
                    return agent
        return None
    
    def add_agent(self, agent: Agent) -> None:
        """Add a new agent to the coordinator."""
        self.registry.register_agent(agent)
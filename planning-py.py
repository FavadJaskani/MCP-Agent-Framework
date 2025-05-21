from typing import Optional, Dict, List, Any

from mcp.core.message import Message
from mcp.protocols.base import Protocol
from mcp.utils.logging import get_logger

logger = get_logger(__name__)

class PlanningProtocol(Protocol):
    """Protocol for task planning and decomposition."""
    
    def __init__(self):
        super().__init__(
            name="PlanningProtocol",
            description="Handles task planning, decomposition, and delegation"
        )
    
    def can_handle_message(self, message: Message) -> bool:
        """Check if this protocol can handle the message."""
        # Handle task messages or messages explicitly for planning
        return (message.message_type == "task" or 
                (message.metadata and message.metadata.get("protocol") == "planning"))
    
    def process_message(self, message: Message) -> Optional[Message]:
        """Process a message to create a plan."""
        logger.info(f"Planning protocol processing message: {message.content[:50]}...")
        
        # 1. Analyze the task
        task_analysis = self._analyze_task(message.content)
        
        # 2. Create a plan
        plan = self._create_plan(task_analysis, message.conversation_id)
        
        # 3. If we have a coordinator, find agents for subtasks
        if hasattr(self.agent, 'coordinator'):
            executor_agent = self._find_execution_agent(self.agent.coordinator)
            if executor_agent:
                # Send the plan to the executor
                return Message(
                    sender_id=self.agent.id,
                    receiver_id=executor_agent.id,
                    content=str(plan),
                    message_type="plan",
                    conversation_id=message.conversation_id,
                    reference_id=message.id,
                    metadata={"protocol": "execution", "plan": plan}
                )
        
        # If no execution agent found, respond directly
        plan_text = "Plan:\n" + "\n".join([f"{i+1}. {step}" for i, step in enumerate(plan["steps"])])
        return Message(
            sender_id=self.agent.id,
            receiver_id=message.sender_id,
            content=f"I've created a plan for this task:\n\n{plan_text}",
            message_type="response",
            conversation_id=message.conversation_id,
            reference_id=message.id
        )
    
    def _analyze_task(self, task_description: str) -> Dict[str, Any]:
        """Analyze a task to determine its components."""
        # In a real implementation, this would use the LLM to analyze the task
        # For this example, just return a simple analysis
        return {
            "main_objective": task_description,
            "complexity": "medium",
            "required_capabilities": ["research", "writing"],
            "estimated_steps": 3
        }
    
    def _create_plan(self, task_analysis: Dict[str, Any], conversation_id: str) -> Dict[str, Any]:
        """Create a structured plan for the task."""
        # This would normally use the LLM to generate a detailed plan
        steps = [
            "Research the topic",
            "Organize findings into key points",
            "Draft the final report"
        ]
        
        return {
            "conversation_id": conversation_id,
            "objective": task_analysis["main_objective"],
            "steps": steps,
            "estimated_time": "30 minutes"
        }
    
    def _find_execution_agent(self, coordinator) -> Optional[Any]:
        """Find an agent that can execute the plan."""
        for agent in coordinator.registry.get_all_agents():
            for protocol in agent.protocols:
                if "execution" in protocol.__class__.__name__.lower():
                    return agent
        return None
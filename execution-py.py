from typing import Optional, Dict, List, Any

from mcp.core.message import Message
from mcp.protocols.base import Protocol
from mcp.utils.logging import get_logger

logger = get_logger(__name__)

class ExecutionProtocol(Protocol):
    """Protocol for executing plans and coordinating subtasks."""
    
    def __init__(self):
        super().__init__(
            name="ExecutionProtocol",
            description="Handles execution of plans and coordination of multiple agents"
        )
        self.active_plans = {}  # Track plans being executed
    
    def can_handle_message(self, message: Message) -> bool:
        """Check if this protocol can handle the message."""
        return (message.message_type == "plan" or 
                (message.metadata and message.metadata.get("protocol") == "execution"))
    
    def process_message(self, message: Message) -> Optional[Message]:
        """Process a plan execution message."""
        logger.info(f"Execution protocol processing message: {message.content[:50]}...")
        
        # Extract the plan from the message metadata
        plan = message.metadata.get("plan") if message.metadata else None
        
        if not plan and message.message_type == "plan":
            # Try to parse the plan from the message content
            plan = self._parse_plan(message.content)
        
        if not plan:
            return Message(
                sender_id=self.agent.id,
                receiver_id=message.sender_id,
                content="Could not extract a valid plan from the message.",
                message_type="error",
                conversation_id=message.conversation_id,
                reference_id=message.id
            )
        
        # Store the plan
        self.active_plans[message.conversation_id] = {
            "plan": plan,
            "status": "in_progress",
            "current_step": 0,
            "results": []
        }
        
        # Execute the plan (in a real implementation, this would be asynchronous)
        result = self._execute_plan(message.conversation_id)
        
        # Return the result
        return Message(
            sender_id=self.agent.id,
            receiver_id=message.sender_id,
            content=result,
            message_type="result",
            conversation_id=message.conversation_id,
            reference_id=message.id
        )
    
    def _parse_plan(self, plan_text: str) -> Optional[Dict[str, Any]]:
        """Parse a plan from text format."""
        # This would normally use more sophisticated parsing
        lines = plan_text.strip().split("\n")
        steps = []
        
        for line in lines:
            if line.strip().startswith(("- ", "• ", "* ")):
                steps.append(line.strip()[2:])
            elif line.strip() and any(c.isdigit() for c in line[:2]):
                steps.append(line.strip()[2:] if line[1] == "." else line.strip()[3:])
        
        if not steps:
            return None
            
        return {
            "objective": lines[0] if lines else "Unknown objective",
            "steps": steps
        }
    
    def _execute_plan(self, conversation_id: str) -> str:
        """Execute a plan and return the result."""
        plan_info = self.active_plans[conversation_id]
        plan = plan_info["plan"]
        
        # In a real implementation, this would delegate to specialized agents
        # and track progress through the plan steps
        
        # For this example, just simulate executing each step
        results = []
        for i, step in enumerate(plan["steps"]):
            logger.debug(f"Executing step {i+1}: {step}")
            plan_info["current_step"] = i
            
            # Simulate executing the step
            step_result = f"Completed: {step}"
            plan_info["results"].append(step_result)
            results.append(step_result)
        
        plan_info["status"] = "completed"
        
        # Format the final result
        final_result = f"Plan execution completed for: {plan['objective']}\n\n"
        final_result += "Results:\n"
        for i, result in enumerate(results):
            final_result += f"{i+1}. {result}\n"
            
        return final_result
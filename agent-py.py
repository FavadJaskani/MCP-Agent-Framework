from typing import List, Dict, Any, Optional
from uuid import uuid4

from mcp.core.message import Message
from mcp.protocols.base import Protocol
from mcp.utils.logging import get_logger

logger = get_logger(__name__)

class Agent:
    """Base agent class that can process messages, use tools, and follow protocols."""
    
    def __init__(
        self,
        name: str,
        description: str = "",
        tools: List[Any] = None,
        protocols: List[Protocol] = None,
        model: str = "gpt-4",
        config: Dict[str, Any] = None
    ):
        self.id = str(uuid4())
        self.name = name
        self.description = description
        self.tools = tools or []
        self.protocols = protocols or []
        self.model = model
        self.config = config or {}
        self.memory = []  # Simple memory for storing messages
        
        # Register protocols
        for protocol in self.protocols:
            protocol.register_agent(self)
            
        logger.info(f"Agent '{name}' initialized with {len(self.tools)} tools and {len(self.protocols)} protocols")
    
    def receive_message(self, message: Message) -> None:
        """Process an incoming message."""
        logger.debug(f"Agent '{self.name}' received message: {message.content[:50]}...")
        
        # Store in memory
        self.memory.append(message)
        
        # Check if any protocol can handle this message
        for protocol in self.protocols:
            if protocol.can_handle_message(message):
                response = protocol.process_message(message)
                if response:
                    return response
        
        # Default handling if no protocol matched
        return self.process_message(message)
    
    def process_message(self, message: Message) -> Optional[Message]:
        """Process a message using the agent's capabilities."""
        # This would typically involve calling an LLM with the appropriate context
        response_content = self._run_inference(message.content)
        
        if response_content:
            return Message(
                sender_id=self.id,
                receiver_id=message.sender_id,
                content=response_content,
                message_type="response",
                conversation_id=message.conversation_id,
                reference_id=message.id
            )
        return None
    
    def send_message(self, receiver_id: str, content: str, message_type: str = "message",
                    conversation_id: Optional[str] = None) -> Message:
        """Create and send a new message."""
        message = Message(
            sender_id=self.id,
            receiver_id=receiver_id,
            content=content,
            message_type=message_type,
            conversation_id=conversation_id or str(uuid4())
        )
        logger.debug(f"Agent '{self.name}' sending message: {content[:50]}...")
        return message
    
    def use_tool(self, tool_name: str, **kwargs) -> Any:
        """Use one of the agent's tools."""
        for tool in self.tools:
            if tool.name == tool_name:
                logger.debug(f"Agent '{self.name}' using tool: {tool_name}")
                return tool.execute(**kwargs)
        logger.warning(f"Tool '{tool_name}' not found for agent '{self.name}'")
        return None
    
    def _run_inference(self, input_text: str) -> str:
        """Run the agent's LLM to generate a response."""
        # In a real implementation, this would call an LLM API
        # For now, just return a placeholder
        return f"Response from {self.name} to: {input_text[:30]}..."
from typing import Dict, Any, Optional
from uuid import uuid4
from datetime import datetime

class Message:
    """Standard message format for agent communication."""
    
    def __init__(
        self,
        sender_id: str,
        receiver_id: str,
        content: str,
        message_type: str = "message",
        conversation_id: Optional[str] = None,
        reference_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.id = str(uuid4())
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.content = content
        self.message_type = message_type
        self.conversation_id = conversation_id or str(uuid4())
        self.reference_id = reference_id
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "content": self.content,
            "message_type": self.message_type,
            "conversation_id": self.conversation_id,
            "reference_id": self.reference_id,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create a message from dictionary data."""
        message = cls(
            sender_id=data["sender_id"],
            receiver_id=data["receiver_id"],
            content=data["content"],
            message_type=data.get("message_type", "message"),
            conversation_id=data.get("conversation_id"),
            reference_id=data.get("reference_id"),
            metadata=data.get("metadata", {})
        )
        message.id = data["id"]
        message.timestamp = data["timestamp"]
        return message
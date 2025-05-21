from mcp.core.agent import Agent
from mcp.core.message import Message
from mcp.protocols.base import Protocol
from mcp.core.coordinator import Coordinator

class DebateProtocol(Protocol):
    """Protocol for structured debates between agents."""
    
    def __init__(self):
        super().__init__(
            name="DebateProtocol",
            description="Enables structured debates with arguments and counterarguments"
        )
        self.debates = {}  # Track ongoing debates
    
    def can_handle_message(self, message: Message) -> bool:
        """Check if this protocol can handle the message."""
        return (message.message_type == "debate" or 
                (message.metadata and message.metadata.get("protocol") == "debate"))
    
    def process_message(self, message: Message) -> Message:
        """Process a debate message."""
        conversation_id = message.conversation_id
        
        # Initialize debate if new
        if conversation_id not in self.debates:
            self.debates[conversation_id] = {
                "topic": message.content,
                "arguments": [],
                "turn": message.sender_id,
                "rounds": 0
            }
        
        debate = self.debates[conversation_id]
        
        # Store the argument
        debate["arguments"].append({
            "agent_id": message.sender_id,
            "content": message.content
        })
        
        # Update debate state
        debate["rounds"] += 0.5  # Two messages make a full round
        debate["turn"] = message.receiver_id
        
        # Generate a response - would normally use the LLM
        response_content = f"Counterargument to: {message.content}"
        
        # Check if debate should end (e.g., after 3 rounds)
        if debate["rounds"] >= 3:
            response_content = f"Final conclusion on the topic: {debate['topic']}\n\n"
            response_content += "Based on the arguments presented, both perspectives have merit..."
            
            # Reset the debate
            del self.debates[conversation_id]
        
        return Message(
            sender_id=self.agent.id,
            receiver_id=message.sender_id,
            content=response_content,
            message_type="debate",
            conversation_id=conversation_id,
            reference_id=message.id,
            metadata={"protocol": "debate"}
        )

def main():
    # Create agents with the debate protocol
    agent1 = Agent(
        name="proponent",
        description="Argues in favor of propositions",
        protocols=[DebateProtocol()],
        model="gpt-4"
    )
    
    agent2 = Agent(
        name="opponent",
        description="Argues against propositions",
        protocols=[DebateProtocol()],
        model="gpt-4"
    )
    
    # Create a coordinator
    coordinator = Coordinator([agent1, agent2])
    
    # Start a debate
    result = coordinator.execute_task(
        "Is artificial general intelligence achievable in the next decade?"
    )
    
    # Print the result
    print("Debate Conclusion:")
    print(result["result"])

if __name__ == "__main__":
    main()
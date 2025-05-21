from mcp.core.agent import Agent
from mcp.core.message import Message
from mcp.tools.web_search import WebSearchTool

def main():
    # Create a simple agent with a web search tool
    agent = Agent(
        name="research_assistant",
        description="Helps with research tasks",
        tools=[WebSearchTool()],
        model="gpt-4"
    )
    
    # Create a message
    message = Message(
        sender_id="user_123",
        receiver_id=agent.id,
        content="Find information about climate change impacts in coastal cities"
    )
    
    # Process the message
    response = agent.receive_message(message)
    
    # Print the response
    if response:
        print(f"Response: {response.content}")
    else:
        print("No response received")

if __name__ == "__main__":
    main()
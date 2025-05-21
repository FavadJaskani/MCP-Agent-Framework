# MCP Protocol Specification

The Multi-agent Communication Protocol (MCP) defines standards for message passing and communication between autonomous AI agents.

## Core Concepts

### Messages

Messages are the fundamental unit of communication in MCP. Each message includes:

- **ID**: Unique identifier for the message
- **Sender ID**: Identifier of the sending agent
- **Receiver ID**: Identifier of the receiving agent
- **Content**: The actual message payload
- **Message Type**: Type of message (e.g., task, response, error)
- **Conversation ID**: Identifier for the conversation thread
- **Reference ID**: (Optional) ID of the message being replied to
- **Metadata**: (Optional) Additional context for the message
- **Timestamp**: When the message was created

### Protocols

Protocols define how agents handle specific types of interactions. Each protocol:

- Determines if it can handle a given message
- Processes messages according to specific rules
- Manages state for ongoing interactions
- Generates appropriate responses

### Agents

Agents are the main actors in the system. Each agent:

- Has a unique identifier
- Can send and receive messages
- Implements one or more protocols
- May have access to tools for executing actions
- Maintains memory of past interactions

## Message Flow

1. **Initialization**: A message is created with sender, receiver, and content
2. **Routing**: The message is delivered to the receiving agent
3. **Protocol Selection**: The agent identifies which protocol can handle the message
4. **Processing**: The protocol processes the message according to its rules
5. **Response**: A response message is generated and sent back
6. **Completion**: When a conversation is complete, final results are returned

## Standard Protocols

MCP includes several standard protocols:

- **PlanningProtocol**: Task decomposition and planning
- **ExecutionProtocol**: Executing plans and coordinating subtasks
- **DebateProtocol**: Structured debates between agents
- **QueryResponseProtocol**: Simple question-answer interactions

## Extending the Protocol

Developers can extend MCP by:

1. Creating new protocols that inherit from the base Protocol class
2. Implementing custom tools for specific agent capabilities
3. Defining specialized agent types for domain-specific tasks

## Example Message Exchange

```json
// Initial message
{
  "id": "msg_123",
  "sender_id": "user_456",
  "receiver_id": "agent_789",
  "content": "Analyze the trends in renewable energy adoption",
  "message_type": "task",
  "conversation_id": "conv_001",
  "timestamp": "2025-05-21T10:30:00Z"
}

// Response message
{
  "id": "msg_124",
  "sender_id": "agent_789",
  "receiver_id": "user_456",
  "content": "Based on my analysis, renewable energy adoption has increased by 15% globally in the past year...",
  "message_type": "response",
  "conversation_id": "conv_001",
  "reference_id": "msg_123",
  "timestamp": "2025-05-21T10:30:05Z"
}
```
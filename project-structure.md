# MCP Agent Framework - Project Structure

```
mcp-agent/
├── README.md
├── LICENSE
├── setup.py
├── requirements.txt
├── docs/
│   ├── protocol_spec.md
│   ├── agent_design.md
│   └── examples.md
├── mcp/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── message.py
│   │   ├── coordinator.py
│   │   └── registry.py
│   ├── protocols/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── planning.py
│   │   └── execution.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging.py
│   │   └── validation.py
│   └── tools/
│       ├── __init__.py
│       ├── web_search.py
│       ├── code_execution.py
│       └── memory.py
└── examples/
    ├── simple_agent.py
    ├── multi_agent_system.py
    └── custom_protocol.py
```

## Core Components

- **agent.py**: Defines the base Agent class for creating autonomous agents
- **message.py**: Implements the Message class for standardized communication
- **coordinator.py**: Provides the Coordinator class for managing multi-agent systems
- **registry.py**: Implements the AgentRegistry for tracking agent instances

## Protocols

- **base.py**: Abstract Protocol class for defining communication protocols
- **planning.py**: PlanningProtocol for task decomposition and planning
- **execution.py**: ExecutionProtocol for executing plans and coordinating subtasks

## Utils

- **logging.py**: Logging utilities for the framework
- **validation.py**: Functions for validating configurations and messages

## Tools

- **web_search.py**: WebSearchTool for retrieving information from the web
- **code_execution.py**: CodeExecutionTool for executing code in various languages
- **memory.py**: MemoryTool for persistent storage and retrieval

## Examples

- **simple_agent.py**: Basic example of creating and using a single agent
- **multi_agent_system.py**: Example of a coordinated multi-agent system
- **custom_protocol.py**: Example of creating and using a custom protocol

## Documentation

- **protocol_spec.md**: Detailed specification of the MCP protocol
- **agent_design.md**: Guidelines for designing effective agents
- **examples.md**: Example applications built with the MCP framework
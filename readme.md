# MCP Agent Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

MCP (Multi-agent Communication Protocol) is a flexible framework for building autonomous AI agents that can communicate, coordinate, and collaborate to solve complex tasks.

## Features

- **Modular Agent Architecture**: Easily create specialized agents with different capabilities
- **Flexible Communication Protocol**: Standardized message format enabling seamless inter-agent communication
- **Tool Integration**: Easily extend agent capabilities with custom tools
- **Coordination Mechanisms**: Built-in planning and execution protocols
- **Memory Systems**: Short and long-term memory for agents
- **Scalable**: From single agents to complex multi-agent systems

## Installation

```bash
pip install mcp-agent
```

## Quick Start

```python
from mcp.core.agent import Agent
from mcp.core.coordinator import Coordinator
from mcp.protocols.planning import PlanningProtocol
from mcp.tools.web_search import WebSearchTool

# Create agents with different capabilities
research_agent = Agent(
    name="researcher",
    tools=[WebSearchTool()],
    description="Performs web research on topics"
)

planner_agent = Agent(
    name="planner",
    protocols=[PlanningProtocol()],
    description="Creates and coordinates execution plans"
)

# Create a coordinator to manage the agents
coordinator = Coordinator([research_agent, planner_agent])

# Execute a task
result = coordinator.execute_task("Research recent developments in AI and create a summary report")
print(result)
```

## Documentation

For complete documentation, see [the docs folder](docs/).

- [Protocol Specification](docs/protocol_spec.md)
- [Agent Design Guide](docs/agent_design.md) 
- [Example Applications](docs/examples.md)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
# MCP Agent Framework

A modern implementation of the Multi-agent Communication Protocol (MCP) for creating autonomous AI agents that can communicate, coordinate, and collaborate on complex tasks.

## Repository Structure

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

## README.md

```markdown
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
```

## Core Implementation Files

### mcp/core/agent.py

```python
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
```

### mcp/core/message.py

```python
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
```

### mcp/core/coordinator.py

```python
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
```

### mcp/core/registry.py

```python
from typing import Dict, Optional, List

from mcp.utils.logging import get_logger

logger = get_logger(__name__)

class AgentRegistry:
    """Registry for tracking available agents."""
    
    def __init__(self):
        self.agents = {}  # Dict[agent_id, agent]
    
    def register_agent(self, agent) -> None:
        """Register a new agent."""
        self.agents[agent.id] = agent
        logger.info(f"Agent '{agent.name}' registered with ID {agent.id}")
    
    def unregister_agent(self, agent_id: str) -> None:
        """Remove an agent from the registry."""
        if agent_id in self.agents:
            agent = self.agents.pop(agent_id)
            logger.info(f"Agent '{agent.name}' unregistered")
    
    def get_agent(self, agent_id: str) -> Optional[any]:
        """Get an agent by ID."""
        return self.agents.get(agent_id)
    
    def get_agents_by_name(self, name: str) -> List[any]:
        """Get all agents with a specific name."""
        return [agent for agent in self.agents.values() if agent.name == name]
    
    def get_all_agents(self) -> List[any]:
        """Get all registered agents."""
        return list(self.agents.values())
```

### mcp/protocols/base.py

```python
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from mcp.utils.logging import get_logger

logger = get_logger(__name__)

class Protocol(ABC):
    """Base class for all agent communication protocols."""
    
    def __init__(self, name: str = None, description: str = None):
        self.name = name or self.__class__.__name__
        self.description = description or f"Protocol: {self.name}"
        self.agent = None  # The agent this protocol is attached to
    
    def register_agent(self, agent) -> None:
        """Register the agent that will use this protocol."""
        self.agent = agent
        logger.debug(f"Protocol '{self.name}' registered with agent '{agent.name}'")
    
    @abstractmethod
    def can_handle_message(self, message) -> bool:
        """Check if this protocol can handle the given message."""
        pass
    
    @abstractmethod
    def process_message(self, message) -> Optional[Any]:
        """Process a message according to this protocol."""
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this protocol."""
        return {
            "name": self.name,
            "description": self.description
        }
```

### mcp/protocols/planning.py

```python
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
```

### mcp/protocols/execution.py

```python
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
```

### mcp/utils/logging.py

```python
import logging
import sys
from typing import Optional

def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Create a logger with the specified name and level."""
    logger = logging.getLogger(name)
    
    if level is not None:
        logger.setLevel(level)
    elif not logger.hasHandlers():
        # Default configuration for new loggers
        logger.setLevel(logging.INFO)
        
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
    
    return logger
```

### mcp/utils/validation.py

```python
from typing import Dict, Any, List, Optional, Callable

def validate_config(config: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """Validate a configuration dictionary against a schema."""
    errors = []
    
    for key, spec in schema.items():
        # Check required fields
        if spec.get("required", False) and key not in config:
            errors.append(f"Missing required field: {key}")
            continue
            
        # Skip validation if the field isn't present
        if key not in config:
            continue
            
        value = config[key]
        
        # Check type
        if "type" in spec:
            expected_type = spec["type"]
            if not isinstance(value, expected_type):
                errors.append(f"Field {key} should be of type {expected_type.__name__}, got {type(value).__name__}")
        
        # Check enum values
        if "enum" in spec and value not in spec["enum"]:
            errors.append(f"Field {key} must be one of {spec['enum']}, got {value}")
        
        # Check custom validation
        if "validate" in spec and callable(spec["validate"]):
            validation_result = spec["validate"](value)
            if validation_result is not True:
                errors.append(f"Field {key} validation failed: {validation_result}")
    
    return errors

def validate_message_format(message: Dict[str, Any]) -> List[str]:
    """Validate a message follows the correct format."""
    schema = {
        "id": {"required": True, "type": str},
        "sender_id": {"required": True, "type": str},
        "receiver_id": {"required": True, "type": str},
        "content": {"required": True, "type": str},
        "message_type": {"required": True, "type": str},
        "conversation_id": {"required": True, "type": str},
        "timestamp": {"required": True, "type": str}
    }
    
    return validate_config(message, schema)
```

### mcp/tools/web_search.py

```python
from typing import Dict, Any, List, Optional

class WebSearchTool:
    """Tool for performing web searches."""
    
    def __init__(self, api_key: Optional[str] = None, search_engine: str = "google"):
        self.name = "web_search"
        self.description = "Search the web for information"
        self.api_key = api_key
        self.search_engine = search_engine
    
    def execute(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """Perform a web search."""
        # In a real implementation, this would call a search API
        # For this example, return mock results
        results = [
            {
                "title": f"Result {i+1} for: {query}",
                "url": f"https://example.com/result{i+1}",
                "snippet": f"This is a snippet of text from search result {i+1} for the query: {query}..."
            }
            for i in range(num_results)
        ]
        
        return {
            "query": query,
            "results": results,
            "num_results": len(results)
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "query": "The search query string",
                "num_results": "Number of results to return (default: 5)"
            }
        }
```

### mcp/tools/code_execution.py

```python
import subprocess
from typing import Dict, Any, Optional

class CodeExecutionTool:
    """Tool for executing code in various languages."""
    
    def __init__(self, allowed_languages: Optional[list] = None):
        self.name = "code_execution"
        self.description = "Execute code in various languages"
        self.allowed_languages = allowed_languages or ["python", "javascript", "bash"]
    
    def execute(self, code: str, language: str = "python", timeout: int = 10) -> Dict[str, Any]:
        """Execute code and return the result."""
        if language not in self.allowed_languages:
            return {
                "status": "error",
                "error": f"Language {language} is not supported. Allowed languages: {', '.join(self.allowed_languages)}"
            }
        
        # IMPORTANT: In a real implementation, this would need security measures
        # like sandboxing to prevent malicious code execution
        
        # For this example, only simulate the execution
        return self._simulate_execution(code, language)
    
    def _simulate_execution(self, code: str, language: str) -> Dict[str, Any]:
        """Simulate code execution (for safety in this example)."""
        return {
            "status": "success",
            "language": language,
            "output": f"[Simulated {language} execution output for safety]\nCode would run in a sandbox in a real implementation.",
            "execution_time": 0.5
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "code": "The code to execute",
                "language": f"The programming language (allowed: {', '.join(self.allowed_languages)})",
                "timeout": "Maximum execution time in seconds (default: 10)"
            }
        }
```

### mcp/tools/memory.py

```python
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

class MemoryTool:
    """Tool for storing and retrieving information from long-term memory."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.name = "memory"
        self.description = "Store and retrieve information from long-term memory"
        self.storage_path = storage_path
        self.memory = {}  # In-memory storage (would use a database in production)
    
    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """Execute a memory operation."""
        if action == "store":
            return self._store_memory(**kwargs)
        elif action == "retrieve":
            return self._retrieve_memory(**kwargs)
        elif action == "search":
            return self._search_memory(**kwargs)
        else:
            return {
                "status": "error",
                "error": f"Unknown action: {action}. Supported actions: store, retrieve, search"
            }
    
    def _store_memory(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Store information in memory."""
        entry = {
            "value": value,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.memory[key] = entry
        
        return {
            "status": "success",
            "key": key,
            "message": f"Successfully stored information under key: {key}"
        }
    
    def _retrieve_memory(self, key: str) -> Dict[str, Any]:
        """Retrieve information from memory."""
        if key not in self.memory:
            return {
                "status": "error",
                "error": f"No information found for key: {key}"
            }
        
        entry = self.memory[key]
        
        return {
            "status": "success",
            "key": key,
            "value": entry["value"],
            "metadata": entry["metadata"],
            "timestamp": entry["timestamp"]
        }
    
    def _search_memory(self, query: str) -> Dict[str, Any]:
        """Search for information in memory."""
        # This would use vector search in a real implementation
        # For this example, just do a simple text search
        results = []
        
        for key, entry in self.memory.items():
            value_str = str(entry["value"])
            if query.lower() in value_str.lower():
                results.append({
                    "key": key,
                    "value": entry["value"],
                    "metadata": entry["metadata"],
                    "timestamp": entry["timestamp"]
                })
        
        return {
            "status": "success",
            "query": query,
            "results": results,
            "count": len(results)
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "action": "The memory operation to perform (store, retrieve, search)",
                "key": "The key to store or retrieve information",
                "value": "The value to store (for store action)",
                "metadata": "Additional metadata for the stored information",
                "query": "Search query (for search action)"
            }
        }
```

## Example Usage

### examples/simple_agent.py

```python
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
```

### examples/multi_agent_system.py

```python
from mcp.core.agent import Agent
from mcp.core.coordinator import Coordinator
from mcp.protocols.planning import PlanningProtocol
from mcp.protocols.execution import ExecutionProtocol
from mcp.tools.web_search import WebSearchTool
from mcp.tools.memory import MemoryTool

def main():
    # Create specialized agents
    planner_agent = Agent(
        name="planner",
        description="Creates plans for complex tasks",
        protocols=[PlanningProtocol()],
        model="gpt-4"
    )
    
    executor_agent = Agent(
        name="executor",
        description="Executes plans by coordinating subtasks",
        protocols=[ExecutionProtocol()],
        model="gpt-4"
    )
    
    researcher_agent = Agent(
        name="researcher",
        description="Performs research on various topics",
        tools=[WebSearchTool()],
        model="gpt-4"
    )
    
    memory_agent = Agent(
        name="memory_manager",
        description="Manages long-term memory and information retrieval",
        tools=[MemoryTool()],
        model="gpt-4"
    )
    
    # Create a coordinator to manage the agents
    coordinator = Coordinator([planner_agent, executor_agent, researcher_agent, memory_agent])
    
    # Execute a complex task
    result = coordinator.execute_task(
        "Research the impact of renewable energy on economic growth in developing countries and prepare a summary report"
    )
    
    # Print the result
    print("Task Result:")
    print(result["result"])

if __name__ == "__main__":
    main()
```

### examples/custom_protocol.py

```python
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
```

## Protocol Specification Documentation

### docs/protocol_spec.md

```markdown
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
```

### docs/agent_design.md

```markdown
# Agent Design Guide

This document provides guidelines for designing effective agents using the MCP framework.

## Agent Architecture

Each agent in MCP consists of the following components:

### Core Components

1. **Identity**: Unique name and ID for the agent
2. **Description**: The agent's capabilities and purpose
3. **Protocols**: Communication protocols the agent implements
4. **Tools**: Action capabilities available to the agent
5. **Model**: The underlying LLM powering the agent
6. **Memory**: Storage for past interactions and knowledge

### Design Principles

When designing agents, follow these principles:

1. **Single Responsibility**: Each agent should excel at a specific task or domain
2. **Clear Communication**: Agents should generate clear, structured messages
3. **Composability**: Agents should work well with other agents
4. **Explainability**: Agent decisions should be traceable and understandable
5. **Failure Handling**: Agents should gracefully handle errors and edge cases

## Specialized Agent Types

### Planner Agents

Planner agents specialize in:
- Task decomposition
- Step sequencing
- Resource allocation
- Goal refinement

Key design considerations:
- Include the PlanningProtocol
- Prioritize reasoning capabilities
- Consider including a MemoryTool for context awareness

### Executor Agents

Executor agents focus on:
- Following plans
- Coordinating between agents
- Tracking progress
- Handling failures

Key design considerations:
- Include the ExecutionProtocol
- Access to relevant tools for tasks
- Strong error handling capabilities

### Research Agents

Research agents excel at:
- Information gathering
- Knowledge synthesis
- Fact verification
- Data analysis

Key design considerations:
- Include tools like WebSearchTool
- Bias minimization strategies
- Citation and source tracking

### Memory Agents

Memory agents specialize in:
- Knowledge storage
- Information retrieval
- Pattern recognition
- Context management

Key design considerations:
- Include tools like MemoryTool
- Implement efficient retrieval mechanisms
- Balance between recall and relevance

## Implementation Best Practices

1. **Prompt Engineering**
   - Craft clear, concise agent descriptions
   - Provide examples of expected behavior
   - Include constraints and guidelines

2. **Tool Integration**
   - Only provide tools relevant to the agent's purpose
   - Ensure tools have clear error handling
   - Document tool usage patterns

3. **Protocol Selection**
   - Choose protocols that match interaction patterns
   - Combine complementary protocols
   - Create custom protocols for specialized needs

4. **Performance Optimization**
   - Minimize unnecessary message passing
   - Cache frequent interactions
   - Use efficient memory storage

5. **Testing**
   - Test individual agent capabilities
   - Test agent interactions
   - Simulate edge cases and failures
```

### docs/examples.md

```markdown
# Example Applications

This document showcases several example applications built with the MCP framework.

## 1. Research Assistant

A system that helps users research complex topics by gathering information, synthesizing findings, and generating reports.

### Components:

- **Planner Agent**: Decomposes research questions into sub-topics
- **Research Agent**: Gathers information using web search
- **Writer Agent**: Synthesizes findings into coherent reports
- **Fact-Checker Agent**: Verifies information accuracy

### Sample Code:

```python
from mcp.core.coordinator import Coordinator
from mcp.core.agent import Agent
from mcp.protocols.planning import PlanningProtocol
from mcp.tools.web_search import WebSearchTool

# Create specialized agents
planner = Agent(
    name="research_planner",
    protocols=[PlanningProtocol()],
    description="Plans research approaches for complex topics"
)

researcher = Agent(
    name="researcher",
    tools=[WebSearchTool()],
    description="Gathers information from the web"
)

writer = Agent(
    name="writer",
    description="Synthesizes research into coherent reports"
)

# Create coordinator
coordinator = Coordinator([planner, researcher, writer])

# Execute research task
result = coordinator.execute_task(
    "Research the impact of quantum computing on cryptography"
)
```

## 2. Collaborative Code Generator

A system that helps developers generate, explain, and optimize code.

### Components:

- **Requirements Agent**: Clarifies user requirements
- **Architect Agent**: Designs code structure
- **Coder Agent**: Generates implementation code
- **Tester Agent**: Creates test cases
- **Optimizer Agent**: Suggests performance improvements

### Sample Code:

```python
from mcp.core.coordinator import Coordinator
from mcp.core.agent import Agent
from mcp.protocols.planning import PlanningProtocol
from mcp.tools.code_execution import CodeExecutionTool

# Create specialized agents
architect = Agent(
    name="architect",
    protocols=[PlanningProtocol()],
    description="Designs software architecture"
)

coder = Agent(
    name="coder",
    tools=[CodeExecutionTool()],
    description="Generates implementation code"
)

tester = Agent(
    name="tester",
    tools=[CodeExecutionTool()],
    description="Creates test cases"
)

# Create coordinator
coordinator = Coordinator([architect, coder, tester])

# Generate code for a specific task
result = coordinator.execute_task(
    "Create a Python function to find the longest palindromic substring in a string"
)
```

## 3. Debate System

A system that facilitates structured debates between different perspectives on complex topics.

### Components:

- **Moderator Agent**: Enforces debate rules and structure
- **Pro Agent**: Argues in favor of a position
- **Con Agent**: Argues against a position
- **Fact-Checker Agent**: Verifies claims made during the debate
- **Summarizer Agent**: Provides debate summaries and conclusions

### Sample Code:

```python
from mcp.core.coordinator import Coordinator
from mcp.core.agent import Agent
from mcp.protocols.base import Protocol

# Define custom DebateProtocol
class DebateProtocol(Protocol):
    def __init__(self):
        super().__init__(
            name="DebateProtocol",
            description="Facilitates structured debates"
        )
        # Protocol implementation details...

# Create specialized agents
moderator = Agent(
    name="moderator",
    protocols=[DebateProtocol()],
    description="Enforces debate rules and structure"
)

pro_agent = Agent(
    name="proponent",
    protocols=[DebateProtocol()],
    description="Argues in favor of propositions"
)

con_agent = Agent(
    name="opponent",
    protocols=[DebateProtocol()],
    description="Argues against propositions"
)

# Create coordinator
coordinator = Coordinator([moderator, pro_agent, con_agent])

# Start a debate on a topic
result = coordinator.execute_task(
    "Debate topic: Should artificial intelligence be regulated by governments?"
)
```

## 4. Educational Tutor

A system that provides personalized learning assistance across various subjects.

### Components:

- **Assessment Agent**: Evaluates student knowledge
- **Curriculum Agent**: Plans learning pathways
- **Tutor Agent**: Delivers educational content
- **Exercise Agent**: Creates practice problems
- **Feedback Agent**: Provides constructive feedback

### Sample Code:

```python
from mcp.core.coordinator import Coordinator
from mcp.core.agent import Agent
from mcp.protocols.planning import PlanningProtocol
from mcp.tools.memory import MemoryTool

# Create specialized agents
assessment = Agent(
    name="assessment",
    tools=[MemoryTool()],
    description="Evaluates student knowledge"
)

curriculum = Agent(
    name="curriculum",
    protocols=[PlanningProtocol()],
    description="Plans personalized learning pathways"
)

tutor = Agent(
    name="tutor",
    tools=[MemoryTool()],
    description="Delivers educational content"
)

# Create coordinator
coordinator = Coordinator([assessment, curriculum, tutor])

# Start a tutoring session
result = coordinator.execute_task(
    "Help me understand calculus derivatives with examples and practice problems"
)
```
```

## setup.py

```python
from setuptools import setup, find_packages

setup(
    name="mcp-agent",
    version="0.1.0",
    description="Multi-agent Communication Protocol (MCP) Framework",
    author="AI Agent Developer",
    author_email="developer@example.com",
    packages=find_packages(),
    install_requires=[
        "requests>=2.25.0",
        "pydantic>=1.8.0",
        "python-dotenv>=0.15.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.9",
)
```

## requirements.txt

```
requests>=2.25.0
pydantic>=1.8.0
python-dotenv>=0.15.0
typing-extensions>=4.0.0
```

## LICENSE

```
MIT License

Copyright (c) 2025 MCP Agent Framework

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

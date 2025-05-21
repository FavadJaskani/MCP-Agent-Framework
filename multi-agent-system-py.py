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
"""
Autonomous agent capabilities for microgpt.
Multi-agent systems, planning, and tool use.
"""

from typing import List, Dict, Callable, Optional, Any
from dataclasses import dataclass
from .model import GPT


@dataclass
class Task:
    """Task for agent execution."""

    id: int
    description: str
    dependencies: List[int]
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Any = None


class Agent:
    """
    Autonomous agent with planning and execution.
    """

    def __init__(self, model: GPT, name: str = "Agent"):
        self.model = model
        self.name = name
        self.tools: Dict[str, Callable] = {}
        self.memory: List[Dict] = []
        self.task_history: List[Task] = []

    def register_tool(self, name: str, func: Callable):
        """Register a tool."""
        self.tools[name] = func

    def plan(self, goal: str) -> List[Task]:
        """
        Create plan to achieve goal.
        """
        # Generate plan
        plan_steps = self._generate_plan(goal)

        # Create tasks
        tasks = []
        for i, step in enumerate(plan_steps):
            task = Task(id=i, description=step, dependencies=[] if i == 0 else [i - 1])
            tasks.append(task)

        return tasks

    def _generate_plan(self, goal: str) -> List[str]:
        """Generate plan steps."""
        return [
            f"Understand: {goal}",
            "Gather information",
            "Analyze options",
            "Execute best option",
            "Verify result",
        ]

    def execute(self, task: Task) -> Any:
        """
        Execute a single task.
        """
        task.status = "in_progress"

        # Decide action
        action = self._decide_action(task.description)

        # Execute
        if action["type"] == "tool":
            result = self.tools.get(action["tool"], lambda x: None)(action["input"])
        elif action["type"] == "think":
            result = self._think(action["content"])
        else:
            result = None

        task.result = result
        task.status = "completed"
        self.task_history.append(task)

        return result

    def _decide_action(self, description: str) -> Dict:
        """Decide how to execute task."""
        if "search" in description.lower():
            return {"type": "tool", "tool": "search", "input": description}
        elif "calculate" in description.lower():
            return {"type": "tool", "tool": "calculator", "input": description}
        else:
            return {"type": "think", "content": description}

    def _think(self, content: str) -> str:
        """Internal reasoning."""
        return f"Thought about: {content}"

    def run(self, goal: str) -> str:
        """
        Execute full plan for goal.
        """
        plan = self.plan(goal)

        for task in plan:
            # Check dependencies
            if all(self.task_history[d].status == "completed" for d in task.dependencies):
                self.execute(task)

        # Synthesize results
        return self._synthesize_results(plan)

    def _synthesize_results(self, tasks: List[Task]) -> str:
        """Combine task results."""
        results = [t.result for t in tasks if t.result]
        return f"Results: {results}"


class MultiAgentSystem:
    """
    System of collaborating agents.
    """

    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.communication_log: List[Dict] = []

    def add_agent(self, agent: Agent):
        """Add agent to system."""
        self.agents[agent.name] = agent

    def delegate(self, task: str, agent_name: Optional[str] = None) -> str:
        """
        Delegate task to specific or best agent.
        """
        if agent_name and agent_name in self.agents:
            return self.agents[agent_name].run(task)

        # Find best agent
        best_agent = self._select_agent(task)
        return best_agent.run(task)

    def _select_agent(self, task: str) -> Agent:
        """Select best agent for task."""
        # Simplified - would use capability matching
        return list(self.agents.values())[0]

    def collaborate(self, task: str, agent_names: List[str]) -> str:
        """
        Multiple agents collaborate on task.
        """
        results = []
        for name in agent_names:
            if name in self.agents:
                result = self.agents[name].run(task)
                results.append(f"{name}: {result}")

        # Synthesize
        return self._merge_results(results)

    def _merge_results(self, results: List[str]) -> str:
        """Merge results from multiple agents."""
        return "\n".join(results)

    def debate(self, topic: str, num_rounds: int = 3) -> str:
        """
        Agents debate a topic.
        """
        positions = {}

        for round_num in range(num_rounds):
            for name, agent in self.agents.items():
                position = agent.run(f"Debate round {round_num + 1}: {topic}")
                positions[name] = position

        # Find consensus or best argument
        return self._resolve_debate(positions)

    def _resolve_debate(self, positions: Dict[str, str]) -> str:
        """Resolve debate."""
        return f"Consensus: {list(positions.values())[0]}"


class Planner:
    """
    Hierarchical task planner.
    """

    def __init__(self, model: GPT):
        self.model = model

    def hierarchical_plan(self, goal: str, depth: int = 3) -> Dict:
        """
        Create hierarchical plan.
        """
        if depth == 0:
            return {"action": goal}

        # Decompose
        subgoals = self._decompose(goal)

        # Recursively plan
        plan = {"goal": goal, "subplans": [self.hierarchical_plan(g, depth - 1) for g in subgoals]}

        return plan

    def _decompose(self, goal: str) -> List[str]:
        """Decompose goal into subgoals."""
        return [f"Subtask 1 of {goal}", f"Subtask 2 of {goal}"]

    def execute_hierarchical(self, plan: Dict) -> Any:
        """Execute hierarchical plan."""
        if "action" in plan:
            return self._execute_action(plan["action"])

        results = []
        for subplan in plan.get("subplans", []):
            result = self.execute_hierarchical(subplan)
            results.append(result)

        return results


class ToolLibrary:
    """
    Library of tools for agents.
    """

    def __init__(self):
        self.tools: Dict[str, Callable] = {
            "calculator": self._calculator,
            "search": self._search,
            "weather": self._weather,
            "datetime": self._datetime,
            "file_read": self._file_read,
            "file_write": self._file_write,
        }

    def _calculator(self, expression: str) -> str:
        """Safe calculator."""
        try:
            # Safe eval
            allowed = {"abs": abs, "max": max, "min": min, "sum": sum, "len": len}
            result = eval(expression, {"__builtins__": {}}, allowed)
            return str(result)
        except Exception:
            return "Error"

    def _search(self, query: str) -> str:
        """Mock search."""
        return f"Results for: {query}"

    def _weather(self, location: str) -> str:
        """Mock weather."""
        return f"Weather in {location}: 72°F"

    def _datetime(self, _: str) -> str:
        """Get datetime."""
        from datetime import datetime

        return datetime.now().isoformat()

    def _file_read(self, path: str) -> str:
        """Read file."""
        try:
            with open(path, "r") as f:
                return f.read()[:1000]
        except Exception:
            return "Error reading file"

    def _file_write(self, content: str) -> str:
        """Write file."""
        # Would write safely
        return "File written"

    def get_tool(self, name: str) -> Optional[Callable]:
        """Get tool by name."""
        return self.tools.get(name)


class Memory:
    """
    Long-term memory for agents.
    """

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.episodic: List[Dict] = []  # Experiences
        self.semantic: Dict[str, Any] = {}  # Facts
        self.procedural: List[str] = []  # Skills

    def add_episode(self, episode: Dict):
        """Add episodic memory."""
        self.episodic.append(episode)
        if len(self.episodic) > self.capacity:
            self.episodic.pop(0)

    def add_fact(self, key: str, value: Any):
        """Add semantic memory."""
        self.semantic[key] = value

    def add_skill(self, skill: str):
        """Add procedural memory."""
        self.procedural.append(skill)

    def recall(self, query: str, k: int = 5) -> List[Dict]:
        """Recall relevant memories."""
        # Simplified - would use embedding similarity
        return self.episodic[-k:]

    def search_facts(self, query: str) -> Dict[str, Any]:
        """Search semantic memory."""
        # Simplified
        return {k: v for k, v in self.semantic.items() if query in k}


class Environment:
    """
    Environment for agent interaction.
    """

    def __init__(self):
        self.state: Dict[str, Any] = {}
        self.agents: List[Agent] = []
        self.history: List[Dict] = []

    def add_agent(self, agent: Agent):
        """Add agent to environment."""
        self.agents.append(agent)

    def step(self) -> Dict:
        """Execute one environment step."""
        for agent in self.agents:
            # Agent perceives and acts
            perception = self._perceive(agent)
            action = agent._decide_action(str(perception))
            self._execute_action(action)

        return self.state

    def _perceive(self, agent: Agent) -> Dict:
        """Agent perceives environment."""
        return self.state

    def _execute_action(self, action: Dict):
        """Execute agent action."""
        self.history.append(action)

    def reset(self):
        """Reset environment."""
        self.state = {}
        self.history = []


def create_agent_system(model: GPT, num_agents: int = 3) -> MultiAgentSystem:
    """Create multi-agent system."""
    system = MultiAgentSystem()
    tools = ToolLibrary()

    for i in range(num_agents):
        agent = Agent(model, name=f"Agent-{i+1}")

        # Register tools
        for name, tool in tools.tools.items():
            agent.register_tool(name, tool)

        system.add_agent(agent)

    return system

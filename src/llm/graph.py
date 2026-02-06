# src/llm/graph.py

from langgraph.graph import StateGraph

from src.agents.analyst_agent import AnalystAgent
from src.agents.strategy_agent import StrategyAgent
from src.agents.auditor_agent import AuditorAgent

class ChurnDecisionGraph:
    def __init__(self):
        self.graph = StateGraph(dict)

        self.graph.add_node("analyst", AnalystAgent().run)
        self.graph.add_node("strategy", StrategyAgent().run)
        self.graph.add_node("auditor", AuditorAgent().run)

        self.graph.set_entry_point("analyst")
        self.graph.add_edge("analyst", "strategy")
        self.graph.add_edge("strategy", "auditor")
        self.graph.set_finish_point("auditor")

    def compile(self):
        return self.graph.compile()

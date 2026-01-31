# LangGraph state

from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    query: str
    plan: Dict[str, Any]
    research_results: List[Any]
    validated_results: List[Any]
    final_answer: str
    logs: List[str]
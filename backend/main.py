from fastapi import FastAPI
from langgraph.graph import StateGraph, END, START
from backend.agents import ManagerAgent, ResearchAgent, ValidationAgent, SummaryAgent
from backend.state import AgentState
from backend.tools import search_web, call_llm, call_llm_stream
from backend.memory import Memory
from fastapi.responses import StreamingResponse
import asyncio
import json

app = FastAPI()

manager = ManagerAgent()
research = ResearchAgent()
validation = ValidationAgent()
summary = SummaryAgent()

def manager_node(state: AgentState):
    return manager.run(state)

def research_node(state: AgentState):
    return research.run(state)

def validation_node(state: AgentState):
    return validation.run(state)

def summary_node(state: AgentState):
    return summary.run(state)

def decide_next_node(state: AgentState):
    strategy = state["plan"]["strategy"]
    if strategy == "direct_answer":
        return "summary"
    elif strategy == "web_research":
        return "research"
    elif strategy == "memory_retrieval":
        return "research"  # Use research node for memory retrieval
    elif strategy == "hybrid":
        return "research"  # Use research node for hybrid
    else:
        return "research"

graph = StateGraph(AgentState)

graph.add_node("manager", manager_node)
graph.add_node("research", research_node)
graph.add_node("validation", validation_node)
graph.add_node("summary", summary_node)

graph.add_edge(START, "manager")
graph.add_conditional_edges("manager", decide_next_node, {
    "direct_answer": "summary",
    "web_research": "research",
    "memory_retrieval": "research",
    "hybrid": "research"
})
graph.add_edge("research", "validation")
graph.add_edge("validation", "summary")
graph.add_edge("summary", END)

workflow = graph.compile()

@app.get("/chat")
async def chat(query: str):
    state: AgentState = {
        "query": query,
        "plan": {},
        "research_results": [],
        "validated_results": [],
        "final_answer": "",
        "logs": []
        }
    result = await workflow.ainvoke(state)
    return result

@app.get("/chat/stream")
async def chat_stream(query: str):
    async def event_generator():
        # Initialize state
        state: AgentState = {
            "query": query,
            "plan": {},
            "research_results": [],
            "validated_results": [],
            "final_answer": "",
            "logs": []
        }
        
        # Send initial event
        yield f"data: {json.dumps({'type': 'start', 'message': 'Starting research...'})}\n\n"
        await asyncio.sleep(0.1)
        
        # Manager Agent
        yield f"data: {json.dumps({'type': 'agent', 'agent': 'Manager', 'status': 'running'})}\n\n"
        state = manager_node(state)
        strategy = state["plan"].get("strategy", "unknown")
        yield f"data: {json.dumps({'type': 'agent', 'agent': 'Manager', 'status': 'complete', 'message': f'Strategy: {strategy}'})}\n\n"
        await asyncio.sleep(0.1)
        
        # Research Agent
        yield f"data: {json.dumps({'type': 'agent', 'agent': 'Research', 'status': 'running'})}\n\n"
        state = research_node(state)
        research_count = len(state["research_results"])
        yield f"data: {json.dumps({'type': 'agent', 'agent': 'Research', 'status': 'complete', 'message': f'Found {research_count} results'})}\n\n"
        await asyncio.sleep(0.1)
        
        # Validation Agent
        yield f"data: {json.dumps({'type': 'agent', 'agent': 'Validation', 'status': 'running'})}\n\n"
        state = validation_node(state)
        validation_count = len(state["validated_results"])
        yield f"data: {json.dumps({'type': 'agent', 'agent': 'Validation', 'status': 'complete', 'message': f'Validated {validation_count} facts'})}\n\n"
        await asyncio.sleep(0.1)
        
        # Summary Agent - with streaming
        yield f"data: {json.dumps({'type': 'agent', 'agent': 'Summary', 'status': 'running'})}\n\n"
        
        # Prepare the prompt for summary
        validated_result = state["validated_results"]
        prompt = f"""
            Create a clear and short answer using the validated facts below.
            Keep it under 5 bullet points.

            Validated facts:
            {validated_result}
            """
        
        # Stream the summary generation
        yield f"data: {json.dumps({'type': 'summary_start'})}\n\n"
        
        summary_text = ""
        for chunk in call_llm_stream(prompt):
            summary_text += chunk
            yield f"data: {json.dumps({'type': 'summary_chunk', 'content': chunk})}\n\n"
            await asyncio.sleep(0.01)  # Small delay for smooth streaming
        
        state["final_answer"] = summary_text
        state["logs"].append("Summary agent generated a final answer.")
        
        yield f"data: {json.dumps({'type': 'agent', 'agent': 'Summary', 'status': 'complete', 'message': 'Summary generated'})}\n\n"
        
        # Send final complete event with full state
        yield f"data: {json.dumps({'type': 'complete', 'final_answer': state['final_answer'], 'logs': state['logs']})}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

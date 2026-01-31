# ALL agents in one file

from backend.state import AgentState
from backend.tools import search_web, call_llm, call_llm_stream
from backend.memory import Memory

class ManagerAgent:
    def __init__(self):
        try:
            self.memory = Memory()
            self.memory_available = True
        except Exception as e:
            print(f"Memory initialization failed: {e}")
            self.memory = None
            self.memory_available = False
    
    def classify_query(self, query: str) -> str:
        prompt = f"""
        Classify the user query into ONE of the following categories:

        1. direct_answer - if the question is simple and can be answered without external search.
        2. web_research - if the question requires latest or external information.
        3. memory_retrieval - if the question is likely answerable from stored knowledge.
        4. hybrid - if both memory and web search may be useful.
        
        Return only the category name. No explanations.

        Query: "{query}"
        """

        strategy = call_llm(prompt).strip().lower()
        return strategy

    def run(self, state: AgentState) -> AgentState:
        query = state["query"]

        # Check if we have relevant memory (only if memory is available)
        memory_hits = []
        if self.memory_available:
            try:
                memory_hits = self.memory.search(query, k=3)
            except Exception as e:
                print(f"Memory search failed: {e}")
                memory_hits = []

        if memory_hits and len(memory_hits) > 0:
            strategy = "hybrid"  # Use hybrid if we have memory
        else:
            strategy = self.classify_query(query)

        # Validate strategy
        if strategy not in ["direct_answer", "web_research", "memory_retrieval", "hybrid"]:
            strategy = "web_research"
        
        # If memory not available, fallback to web_research
        if not self.memory_available and strategy in ["memory_retrieval", "hybrid"]:
            strategy = "web_research"
        
        state["plan"] = {"strategy": strategy}
        state["logs"].append(f"ManagerAgent decided strategy: {strategy}")
        return state

        
class ValidationAgent:
    def run(self, state: AgentState) -> AgentState:
        if not state["research_results"]:
            state["validated_results"] = []
            state["logs"].append("ValidationAgent: No results to validate.")
            return state
            
        validated_result = "\n".join(state["research_results"])
        prompt = f"""
                    Validate the following information and return only 3-5 concise verified facts.
                    Rules:
                    - No explanations
                    - No repetition
                    - Each point must be one sentence
                    - Output as bullet points

                    Information:
                    {validated_result}
                    """

        validated_result = call_llm(prompt)
        
        # Convert LLM output into list
        state["validated_results"] = [
            line.strip() for line in validated_result.split("\n") if line.strip()
        ]
        state["logs"].append("ValidationAgent validated results.")
        return state

class ResearchAgent:
    def __init__(self):
        try:
            self.memory = Memory()
            self.memory_available = True
        except Exception as e:
            print(f"Memory initialization failed: {e}")
            self.memory = None
            self.memory_available = False
    
    def run(self, state: AgentState) -> AgentState:
        strategy = state["plan"]["strategy"]
        query = state["query"]
        results = []

        # Handle different strategies
        if strategy == "direct_answer":
            state["logs"].append("ResearchAgent: Skipped (direct answer).")
            state["research_results"] = []
            return state

        elif strategy == "memory_retrieval":
            # Only retrieve from memory
            if self.memory_available:
                try:
                    memory_results = self.memory.search(query, k=5)
                    if memory_results:
                        results.extend(memory_results)
                        state["logs"].append(f"ResearchAgent: Retrieved {len(memory_results)} results from memory.")
                    else:
                        state["logs"].append("ResearchAgent: No memory results found.")
                except Exception as e:
                    state["logs"].append(f"ResearchAgent: Memory search failed, falling back to web.")
                    # Fallback to web search
                    web_results = search_web(query, max_results=5)
                    if web_results:
                        results.extend(web_results)
                        state["logs"].append(f"ResearchAgent: Collected {len(web_results)} web results.")
            else:
                state["logs"].append("ResearchAgent: Memory not available, using web search.")
                web_results = search_web(query, max_results=5)
                if web_results:
                    results.extend(web_results)
                    state["logs"].append(f"ResearchAgent: Collected {len(web_results)} web results.")

        elif strategy == "web_research":
            # Only web search
            web_results = search_web(query, max_results=5)
            if web_results:
                results.extend(web_results)
                state["logs"].append(f"ResearchAgent: Collected {len(web_results)} web results.")
                
                # Save to memory for future use (if available)
                if self.memory_available:
                    try:
                        self.memory.save(web_results, query=query)
                        state["logs"].append(f"ResearchAgent: Saved {len(web_results)} results to memory.")
                    except Exception as e:
                        state["logs"].append(f"ResearchAgent: Could not save to memory.")
            else:
                state["logs"].append("ResearchAgent: No web results found.")

        elif strategy == "hybrid":
            # Both memory and web
            if self.memory_available:
                try:
                    memory_results = self.memory.search(query, k=3)
                    if memory_results:
                        results.extend(memory_results)
                        state["logs"].append(f"ResearchAgent: Retrieved {len(memory_results)} results from memory.")
                except Exception as e:
                    state["logs"].append(f"ResearchAgent: Memory search failed.")
            
            web_results = search_web(query, max_results=3)
            if web_results:
                results.extend(web_results)
                state["logs"].append(f"ResearchAgent: Collected {len(web_results)} web results.")
                
                # Save new web results to memory (if available)
                if self.memory_available:
                    try:
                        self.memory.save(web_results, query=query)
                        state["logs"].append(f"ResearchAgent: Saved {len(web_results)} results to memory.")
                    except Exception as e:
                        state["logs"].append(f"ResearchAgent: Could not save to memory.")

        state["research_results"] = results[:5]  # Limit to 5 total results
        return state
        
class SummaryAgent:
    def run(self, state: AgentState) -> AgentState:
        validated_result = state["validated_results"]

        prompt = f"""
            Create a clear and short answer using the validated facts below.
            Keep it under 5 bullet points.

            Validated facts:
            {validated_result}
            """
        summary = call_llm(prompt)
        state["final_answer"] = summary
        state["logs"].append("Summary agent generated a final answer.")
        return state

"""
LangGraph Workflow for Insurance RAG System.
Implements deterministic, compliance-focused retrieval with specialized nodes.
"""

from langgraph.graph import StateGraph, END
from agents.states import AgentState
from agents.nodes import nodes


def build_rag_workflow() -> StateGraph:
    """
    Builds the LangGraph workflow with the following flow:
    
    query_rewriter → query_classifier → entity_extractor → retrieval_router
                              ↓
                    [conditional routing by intent]
                              ↓
        ┌─────────────────────────────────────────────────────┐
        │ list_plans: listing_agent → guardrail              │
        │ plan_details: retriever → aggregator → retrieval → guardrail │
        │ compare_plans: retriever → aggregator → comparison → guardrail │
        │ recommendation: retriever → aggregator → advisory → guardrail │
        │ general_query: retriever → aggregator → faq → guardrail │
        └─────────────────────────────────────────────────────┘
    """
    
    workflow = StateGraph(AgentState)
    
    # =========================================================================
    # Add all nodes
    # =========================================================================
    
    # Pre-processing nodes
    workflow.add_node("query_rewriter", nodes.query_rewriter_node)
    workflow.add_node("query_classifier", nodes.query_classifier_node)
    workflow.add_node("entity_extractor", nodes.entity_extractor_node)
    workflow.add_node("retrieval_router", nodes.retrieval_router_node)
    
    # Retrieval nodes
    workflow.add_node("retriever", nodes.retriever_node)
    workflow.add_node("plan_aggregator", nodes.plan_aggregator_node)
    
    # Agent nodes
    workflow.add_node("listing_agent", nodes.listing_agent)
    workflow.add_node("retrieval_agent", nodes.retrieval_agent)
    workflow.add_node("advisory_agent", nodes.advisory_agent)
    workflow.add_node("faq_agent", nodes.faq_agent)
    
    # Post-processing
    workflow.add_node("guardrail", nodes.guardrail_node)
    
    # =========================================================================
    # Define edges
    # =========================================================================
    
    # Entry point
    workflow.set_entry_point("query_rewriter")
    
    # Linear pre-processing chain
    workflow.add_edge("query_rewriter", "query_classifier")
    workflow.add_edge("query_classifier", "entity_extractor")
    workflow.add_edge("entity_extractor", "retrieval_router")
    
    # Conditional routing based on intent
    def route_by_intent(state: AgentState) -> str:
        """Route to appropriate handler based on classified intent."""
        intent = state.get("intent", "plan_details")
        
        if intent == "list_plans":
            return "listing_agent"
        else:
            return "retriever"
    
    workflow.add_conditional_edges(
        "retrieval_router",
        route_by_intent,
        {
            "listing_agent": "listing_agent",
            "retriever": "retriever"
        }
    )
    
    # Listing agent goes directly to guardrail
    workflow.add_edge("listing_agent", "guardrail")
    
    # Retriever always goes to aggregator
    workflow.add_edge("retriever", "plan_aggregator")
    
    # Aggregator routes to appropriate agent based on intent
    def route_to_agent(state: AgentState) -> str:
        """Route from aggregator to the appropriate agent."""
        intent = state.get("intent", "plan_details")
        
        route_map = {
            "plan_details": "retrieval_agent",
            "recommendation": "advisory_agent",
            "general_query": "faq_agent"
        }
        
        return route_map.get(intent, "retrieval_agent")
    
    workflow.add_conditional_edges(
        "plan_aggregator",
        route_to_agent,
        {
            "retrieval_agent": "retrieval_agent",
            "advisory_agent": "advisory_agent",
            "faq_agent": "faq_agent"
        }
    )
    
    # All agents end at guardrail
    workflow.add_edge("retrieval_agent", "guardrail")
    workflow.add_edge("advisory_agent", "guardrail")
    workflow.add_edge("faq_agent", "guardrail")
    
    # Guardrail ends the workflow
    workflow.add_edge("guardrail", END)
    
    return workflow


# Build and compile the workflow
workflow = build_rag_workflow()
app = workflow.compile()


if __name__ == "__main__":
    # Test the graph
    print("Graph compiled successfully!")
    
    # Test cases
    test_queries = [
        "List all term plans from Tata AIA",
        "Explain the TATA AIA Smart Value Income plan",
        "Compare Tata AIA vs Edelweiss term plans",
        "Suggest a plan for 30-year-old non-smoker with 1Cr cover"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        initial_state = {
            "input": query,
            "chat_history": [],
            "intent": "",
            "extracted_entities": {},
            "metadata_filters": {},
            "retrieval_strategy": "",
            "context": [],
            "retrieved_chunks": {},
            "reasoning_output": "",
            "answer": "",
            "next_step": ""
        }
        
        try:
            result = app.invoke(initial_state)
            print(f"Intent: {result.get('intent')}")
            print(f"Answer: {result.get('answer', '')[:500]}...")
        except Exception as e:
            print(f"Error: {e}")

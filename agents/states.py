from typing import TypedDict, Annotated, List, Dict, Any, Optional
import operator


class UserProfile(TypedDict, total=False):
    """User profile for recommendation intent."""
    age: Optional[int]
    gender: Optional[str]  # "male", "female"
    income: Optional[str]
    smoker: Optional[bool]
    dependents: Optional[int]
    goal: Optional[str]  # "protection", "savings", "retirement", "wealth"
    cover_amount: Optional[str]  # e.g., "1 Cr", "50 Lakh"
    premium_amount: Optional[str]  # e.g., "1 Lakh", "50000"
    policy_term: Optional[str] # PT
    payment_term: Optional[str] # PPT
    payment_mode: Optional[str] # Mode (Monthly, Annual, etc.)


class ExtractedEntities(TypedDict, total=False):
    """Entities extracted from user query."""
    provider: Optional[List[str]]  # ["TATA AIA", "Edelweiss Life"]
    insurance_type: Optional[List[str]]  # ["Term Insurance", "ULIP"]
    plan_names: Optional[List[str]]  # Specific plan names mentioned
    user_profile: Optional[UserProfile]


class AgentState(TypedDict):
    """
    Enhanced state for the LangGraph RAG workflow.
    Supports deterministic, compliance-focused retrieval.
    """
    # Input
    input: str
    chat_history: List[str]
    
    # Query Classification
    intent: str  # 'list_plans', 'plan_details', 'compare_plans', 'recommendation', 'general_query'
    query_complexity: str # 'low' | 'high'
    
    # Entity Extraction
    extracted_entities: ExtractedEntities
    
    # Retrieval Configuration
    metadata_filters: Dict[str, Any]  # Filters for vector store
    retrieval_strategy: str  # 'metadata_only', 'plan_level', 'section_specific', 'cross_plan'
    
    # Retrieved Content
    context: List[str]  # accumulated context strings
    retrieved_chunks: Dict[str, List[Dict]]  # Grouped by plan_id: {plan_id: [chunks]}
    
    # Reasoning & Output
    reasoning_output: str  # Structured comparison/recommendation data
    answer: str  # Final answer to user
    
    # Internal Routing
    next_step: str  # For conditional edges

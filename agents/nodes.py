import re
from typing import Dict, List, Any, Optional
from collections import defaultdict
from agents.states import AgentState, ExtractedEntities
from rag.retriever import RAGRetriever
from langchain_core.messages import HumanMessage, SystemMessage
from models.llm import LLMFactory


# Compliance disclaimer to append to all answers
COMPLIANCE_DISCLAIMER = (
    "\n\n---\n"
)

# Prompting rules for all agents
COMPLIANCE_RULES = """
CRITICAL RULES:
- ❌ NO invented plan names - only use plans from the provided context
- ❌ NO assumptions beyond documents - if info is missing, say so explicitly
- ❌ NO meta-commentary. DO NOT mention "the provided context", "the documents", "the text", or "internal state".
- ✅ CIS overrides brochure for: exclusions, charges, conditions
- ✅ Use structured output (markdown tables) for comparisons
- ✅ Simple, clear language for end users
- ✅ Provide "OUTPUT ONLY" - start answering the user's question directly.
"""


class AgentNodes:
    """
    Enhanced LangGraph nodes implementing the full RAG specification.
    """
    
    def __init__(self):
        self.retriever = None

    def _get_retriever(self) -> Optional[RAGRetriever]:
        """Lazy initialization of retriever."""
        if not self.retriever:
            try:
                self.retriever = RAGRetriever()
            except Exception:
                return None
        return self.retriever

    def reload_retriever(self):
        """Triggers a reload of the retriever's index."""
        retriever = self._get_retriever()
        if retriever:
            retriever.reload()

    # =========================================================================
    # NODE 1: Query Rewriter
    # =========================================================================
    def query_rewriter_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Rewrites query to be self-contained based on chat history.
        Resolves pronouns and references.
        """
        llm = LLMFactory.get_llm("small")
        query = state["input"]
        history = state.get("chat_history", [])
        
        if not history:
            return {"input": query}
            
        system_prompt = (
            "You are a query rewriter for an insurance RAG system. "
            "Your task is to rewrite the latest question to be self-contained.\n\n"
            "RULES:\n"
            "1. ALWAYS resolve pronouns (it, they, these) or vague terms (the plan, previous one) using the previous context.\n"
            "2. If the user asks a follow-up about 'it' or 'the plan', replace it with the specific plan name mentioned last.\n"
            "3. If the user asks 'is it good for me' or similar, rewrite it to '[Plan Name] recommendation for [user details if any]'.\n"
            "4. If the query is already very specific and names a plan, keep it mostly as-is but ensure insurer names are present.\n"
            "5. Do NOT cross-pollinate unrelated queries. If the user switches topics completely, ignore the history.\n"
            "6. NEVER return a conversational response, suggestion, or question. If you cannot resolve a reference, return the original 'Latest' query as is.\n"
            "7. Return ONLY the rewritten query text."
        )
        
        history_str = "\n".join([f"- {h}" for h in history[-5:]])  # Last 5 turns
        prompt = f"History:\n{history_str}\n\nLatest: {query}"
        
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=prompt)])
        rewritten = getattr(response, 'content', str(response)).strip()
        
        return {"input": rewritten}

    # =========================================================================
    # NODE 2: Query Classifier
    # =========================================================================
    def query_classifier_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Classifies user intent into:
        - list_plans: User wants to see available plans
        - plan_details: User asks about a specific plan
        - compare_plans: User wants to compare multiple plans
        - recommendation: User seeks personalized advice
        - general_query: General insurance questions
        """
        llm = LLMFactory.get_llm("small")
        query = state["input"].lower()
        
        # Fast keyword-based classification first
        if any(kw in query for kw in ["list", "which plans", "what plans", "all plans", "available plans", "show me plans"]):
            return {"intent": "list_plans"}
        
        if any(kw in query for kw in ["compare", "vs", "versus", "difference between", "which is better"]):
            return {"intent": "compare_plans"}
        
        if any(kw in query for kw in ["suggest", "recommend", "best for", "should i", "suitable for"]):
            return {"intent": "recommendation"}
        
        # LLM-based classification for ambiguous cases
        system_prompt = (
            "Classify the user's insurance query into ONE of:\n"
            "- 'plan_details': Asking about features, benefits, eligibility of a SPECIFIC plan\n"
            "- 'list_plans': Wants to know WHICH plans are available\n"
            "- 'compare_plans': Wants to COMPARE 2+ plans side-by-side\n"
            "- 'recommendation': Seeks personalized advice based on their profile\n"
            "- 'general_query': General insurance terminology or concepts\n\n"
            "Return ONLY the category name."
        )
        
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=query)])
        intent = getattr(response, 'content', str(response)).lower().strip()
        
        valid_intents = ['list_plans', 'plan_details', 'compare_plans', 'recommendation', 'general_query']
        if intent not in valid_intents:
            intent = "plan_details"  # Default fallback
            
        return {"intent": intent}

    # =========================================================================
    # NODE 3: Entity Extractor
    # =========================================================================
    def entity_extractor_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Extracts structured entities from the query:
        - provider (insurer names)
        - insurance_type (term, ulip, savings, etc.)
        - plan_names (specific plan names mentioned)
        - user_profile (age, income, smoker, dependents, goal)
        """
        query = state["input"].lower()
        
        # Extract providers
        provider_map = {
            "edelweiss": "Edelweiss Life",
            "tata": "TATA AIA",
            "tata aia": "TATA AIA",
            "generali": "Generali Central",
            "central": "Generali Central",
            "pramerica": "PRAMERICA"
        }
        providers = []
        for keyword, name in provider_map.items():
            if keyword in query and name not in providers:
                providers.append(name)
        
        # Extract insurance types
        type_map = {
            "term": ["Term Insurance", "Term Plan"],
            "ulip": ["Unit Linked Insurance Plan", "ULIP Plan"],
            "wealth": ["Unit Linked Insurance Plan"],
            "savings": ["Savings Plan", "Guaranteed Return"],
            "retirement": ["Retirement and Pension"],
            "pension": ["Retirement and Pension"],
            "health": ["Health Insurance"],
            "group": ["Group Plan"]
        }
        insurance_types = []
        for keyword, types in type_map.items():
            if keyword in query:
                for t in types:
                    if t not in insurance_types:
                        insurance_types.append(t)
        
        # Extract specific plan names using LLM
        plan_names = self._extract_plan_names_from_query(state["input"])
        
        # Extract user profile for recommendation intent
        user_profile = {}
        if state.get("intent") == "recommendation":
            user_profile = self._extract_user_profile(state["input"])
        
        entities: ExtractedEntities = {
            "provider": list(set(providers)) if providers else [],
            "insurance_type": list(set(insurance_types)) if insurance_types else [],
            "plan_names": list(set(plan_names)) if plan_names else [],
            "user_profile": user_profile or {}
        }
        
        # Build metadata filters from entities
        filters = {}
        if providers:
            filters["insurer"] = providers
        if insurance_types:
            filters["insurance_type"] = insurance_types
        
        return {
            "extracted_entities": entities,
            "metadata_filters": filters
        }

    def _extract_plan_names_from_query(self, query: str) -> List[str]:
        """Use LLM to extract specific plan names mentioned in query."""
        llm = LLMFactory.get_llm("small")
        
        system_prompt = (
            "Extract EXACT insurance plan names from the query.\n"
            "If the user is asking to compare, extract BOTH plan names.\n"
            "RULES:\n"
            "- Return one plan name per line\n"
            "- Include insurer prefix if mentioned (e.g., 'TATA AIA Smart Value Income', 'Edelweiss Saral Jeevan Bima')\n"
            "- Return EMPTY if no specific plan names found\n"
            "- Do NOT invent plan names"
        )
        
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=query)])
        result = getattr(response, 'content', str(response)).strip()
        
        # Validation: If LLM returns a sentence instead of names, skip it
        if "mentioned in" in result.lower() or "referring to" in result.lower() or len(result) > 200:
            return []
        
        if not result or result.lower() in ['none', 'empty', 'n/a']:
            return []
        
        # Parse response
        plan_names = []
        for line in result.split('\n'):
            line = re.sub(r'^[\d\.\-\*\u2022]\s*', '', line).strip().strip('"\'')
            if len(line) > 5:
                plan_names.append(line)
        
        return plan_names

    def _extract_user_profile(self, query: str) -> Dict[str, Any]:
        """Extract user profile information for recommendations."""
        llm = LLMFactory.get_llm("small")
        
        system_prompt = (
            "Extract user profile from the insurance query.\n"
            "Return in format:\n"
            "age: <number or null>\n"
            "smoker: <yes/no or null>\n"
            "cover_amount: <amount or null>\n"
            "goal: <protection/savings/retirement/wealth or null>\n"
            "dependents: <number or null>"
        )
        
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=query)])
        result = getattr(response, 'content', str(response))
        
        profile = {}
        for line in result.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip().lower()
                if value not in ['null', 'none', 'n/a', '']:
                    if key == 'age':
                        try:
                            profile['age'] = int(re.search(r'\d+', value).group())
                        except:
                            pass
                    elif key == 'smoker':
                        profile['smoker'] = 'yes' in value
                    elif key == 'cover_amount':
                        profile['cover_amount'] = value
                    elif key == 'goal':
                        profile['goal'] = value
                    elif key == 'dependents':
                        try:
                            profile['dependents'] = int(re.search(r'\d+', value).group())
                        except:
                            pass
        
        return profile

    # =========================================================================
    # NODE 4: Retrieval Router
    # =========================================================================
    def retrieval_router_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Determines retrieval strategy based on intent.
        """
        intent = state.get("intent", "plan_details")
        
        strategy_map = {
            "list_plans": "metadata_only",
            "plan_details": "plan_level",
            "compare_plans": "cross_plan",
            "recommendation": "section_specific",
            "general_query": "plan_level"
        }
        
        return {"retrieval_strategy": strategy_map.get(intent, "plan_level")}

    # =========================================================================
    # NODE 5: Retriever
    # =========================================================================
    def retriever_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Retrieves documents with:
        - Metadata filtering
        - CIS boosting for exclusions/charges/conditions
        - Deduplication by similarity
        """
        retriever = self._get_retriever()
        if not retriever:
            return {"context": [], "retrieved_chunks": {}}
        
        query = state["input"]
        filters = state.get("metadata_filters", {})
        entities = state.get("extracted_entities", {})
        strategy = state.get("retrieval_strategy", "plan_level")
        
        # If specific plan names were extracted, use them for precise retrieval
        plan_names = entities.get("plan_names") or []
        matched_plans = []
        if plan_names:
            # Resolve to actual plan names in index
            all_plans = self._list_plans_from_index()
            for name in plan_names:
                match = self._find_closest_plan_name(name, all_plans)
                if match:
                    matched_plans.append(match)
            
            # Update filters for non-comparison queries (for comparison, _retrieve_for_comparison handles it)
            if matched_plans and strategy != "cross_plan":
                filters = filters.copy()
                filters["product_name"] = matched_plans[0] if len(matched_plans) == 1 else matched_plans
        
        boost_cis = any(kw in query.lower() for kw in 
                       ["exclusion", "excluded", "not covered", "charges", "fee", "condition", "waiting"])
        
        # Retrieve documents
        if strategy == "cross_plan":
            # For comparisons, retrieve for each plan separately
            # Pass matched_plans if we have them, otherwise it will try to find them from filters
            chunks_by_plan = self._retrieve_for_comparison(query, filters, entities, matched_plans=matched_plans)
        else:
            docs = retriever.search(query, filters=filters, k=8)
            chunks_by_plan = self._group_by_plan_id(docs)
        
        # Boost CIS documents if needed
        if boost_cis:
            chunks_by_plan = self._boost_cis_chunks(chunks_by_plan)
        
        # Format context strings
        limit_per_plan = 5 if strategy == "cross_plan" else 3
        context = self._format_context(chunks_by_plan, limit=limit_per_plan)
        
        return {
            "context": context,
            "retrieved_chunks": chunks_by_plan
        }

    def _format_context(self, chunks_by_plan: Dict[str, List[Dict]], limit: int = 3) -> List[str]:
        """Helper to format chunks into LLM-readable context strings."""
        context = []
        for plan_id, chunks in chunks_by_plan.items():
            for chunk in chunks[:limit]:
                content = chunk.get("content", "")
                plan_name = chunk.get("product_name", "Unknown")
                doc_type = chunk.get("document_type", "brochure")
                section = chunk.get("section", "General")
                context.append(f"[{plan_name} - {doc_type.upper()} - {section}] {content}")
        return context

    def _retrieve_for_comparison(self, query: str, filters: Dict, entities: Dict, matched_plans: List[str] = None) -> Dict[str, List]:
        """Retrieve chunks for each plan separately in comparison mode."""
        retriever = self._get_retriever()
        if not retriever:
            return {}
        
        if not matched_plans:
            plan_names = entities.get("plan_names") or []
            all_index_plans = self._list_plans_from_index()
            matched_plans = []
            for name in plan_names:
                match = self._find_closest_plan_name(name, all_index_plans)
                if match:
                    matched_plans.append(match)
        
        if not matched_plans:
            # Plan A: Deterministic "List & Match" Discovery
            # For each provider, list all their plans and see if any match the query
            providers = entities.get("provider") or []
            
            if not providers:
                search_providers = [None]
            else:
                search_providers = providers
                
            discovered_names = []
            all_plans_in_index = self._list_plans_from_index()
            
            for prov in search_providers:
                prov_filter = {"insurer": prov} if prov else {}
                prov_plans = self._list_plans_from_index(filters=prov_filter)
                self._log_debug(f"Provider: {prov}, Plans found: {len(prov_plans)}")
                
                # Try to find which plan from this insurer is mentioned in the query
                match = self._find_closest_plan_name(query, prov_plans)
                self._log_debug(f"Match for {prov}: {match} (In list: {match in prov_plans})")
                
                if match and match in prov_plans and match not in discovered_names:
                    discovered_names.append(match)
            
            matched_plans = discovered_names
        
        if not matched_plans:
            # Plan B: Fall back to broad similarity-based discovery as a last resort
            discovery_docs = retriever.search(query, k=20)
            for d in discovery_docs:
                p_name = d.metadata.get("product_name")
                if p_name and p_name not in matched_plans:
                    matched_plans.append(p_name)
            matched_plans = matched_plans[:3]
        
        if not matched_plans:
            # Plan B: Fall back to listing plans matching filters (metadata-only)
            matched_plans = self._list_plans_from_index(filters)[:5]
        
        chunks_by_plan = defaultdict(list)
        for matched in matched_plans:
            
            # Use a focused query for each plan instead of the broad comparison query
            # This helps the retriever find relevant feature chunks for the specific plan
            focused_query = f"features, benefits, eligibility and exclusions of {matched}"
            
            # Use a fresh, strictly focused filter for each plan
            # IMPORTANT: Search by insurer and manually filter by product_name
            # This is more robust than passing a combined filter to the vector store
            matched_insurer = None
            if hasattr(self, "_cached_plans") and self._cached_plans:
                for p_meta in self._cached_plans:
                    if p_meta["product_name"] == matched:
                        matched_insurer = p_meta.get("insurer")
                        break
            
            search_filters = {"insurer": matched_insurer} if matched_insurer else {}
            
            # Search only by insurer and then manually filter by product_name
            # This is more robust than passing a combined filter to the vector store
            docs = retriever.search(focused_query, filters=search_filters, k=50)
            
            plan_chunks = []
            for doc in docs:
                doc_product = doc.metadata.get("product_name", "")
                # Use fuzzy match for manual filter consistency
                if self._find_closest_plan_name(doc_product, [matched]) == matched:
                    plan_chunks.append(doc)
            
            for doc in plan_chunks[:10]:
                plan_id = doc.metadata.get("plan_id", matched)
                chunks_by_plan[plan_id].append({
                    "content": doc.page_content,
                    "product_name": doc.metadata.get("product_name"),
                    "document_type": doc.metadata.get("document_type", "brochure"),
                    "section": doc.metadata.get("section", "General")
                })
        
        return dict(chunks_by_plan)

    def _group_by_plan_id(self, docs: List) -> Dict[str, List]:
        """Group retrieved documents by plan_id."""
        grouped = defaultdict(list)
        for doc in docs:
            plan_id = doc.metadata.get("plan_id", doc.metadata.get("product_name", "unknown"))
            grouped[plan_id].append({
                "content": doc.page_content,
                "product_name": doc.metadata.get("product_name"),
                "document_type": doc.metadata.get("document_type", "brochure"),
                "section": doc.metadata.get("section", "General")
            })
        return dict(grouped)

    def _boost_cis_chunks(self, chunks_by_plan: Dict[str, List]) -> Dict[str, List]:
        """Boost CIS documents to appear first for each plan."""
        boosted = {}
        for plan_id, chunks in chunks_by_plan.items():
            cis_chunks = [c for c in chunks if c.get("document_type") == "cis"]
            brochure_chunks = [c for c in chunks if c.get("document_type") != "cis"]
            boosted[plan_id] = cis_chunks + brochure_chunks
        return boosted

    # =========================================================================
    # NODE 6: Plan Aggregator
    # =========================================================================
    def plan_aggregator_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Aggregates chunks by plan_id, merging brochure and CIS context.
        CIS overrides brochure for exclusions, charges, conditions.
        """
        chunks_by_plan = state.get("retrieved_chunks", {})
        
        # Already grouped, just ensure proper ordering
        aggregated = {}
        for plan_id, chunks in chunks_by_plan.items():
            # Separate by document type
            cis_chunks = [c for c in chunks if c.get("document_type") == "cis"]
            brochure_chunks = [c for c in chunks if c.get("document_type") != "cis"]
            
            # For exclusions/charges sections, prefer CIS
            override_sections = ["Exclusions", "Charges", "Waiting Period", "Conditions"]
            
            final_chunks = []
            covered_sections = set()
            
            # Add CIS chunks first for override sections
            for chunk in cis_chunks:
                section = chunk.get("section", "General")
                if section in override_sections:
                    final_chunks.append(chunk)
                    covered_sections.add(section)
            
            # Add brochure chunks, skipping overridden sections
            for chunk in brochure_chunks:
                section = chunk.get("section", "General")
                if section not in covered_sections:
                    final_chunks.append(chunk)
            
            # Add remaining CIS chunks
            for chunk in cis_chunks:
                if chunk not in final_chunks:
                    final_chunks.append(chunk)
            
            aggregated[plan_id] = final_chunks
        
        # Refresh context strings based on aggregated chunks
        intent = state.get("intent", "plan_details")
        limit = 5 if intent == "compare_plans" else 3
        context = self._format_context(aggregated, limit=limit)
        
        return {
            "retrieved_chunks": aggregated,
            "context": context
        }

    # =========================================================================
    # NODE 7: Listing Agent
    # =========================================================================
    def listing_agent(self, state: AgentState) -> Dict[str, Any]:
        """
        Lists available plans based on filters.
        Uses direct index access for accuracy.
        """
        llm = LLMFactory.get_llm("small")
        query = state["input"]
        filters = state.get("metadata_filters", {})
        
        plans = self._list_plans_from_index(filters)
        plans = sorted(list(set(plans)))
        
        if not plans:
            filter_desc = ", ".join([str(v) for v in filters.values()]) if filters else "your criteria"
            answer = f"I couldn't find any plans matching {filter_desc}. Please try a different search."
            return {"context": [], "answer": answer}
        
        plans_str = "\n".join([f"- {p}" for p in plans])
        
        # Describe the filters
        filter_parts = []
        if filters.get("insurer"):
            insurer_list = filters["insurer"] if isinstance(filters["insurer"], list) else [filters["insurer"]]
            filter_parts.append(f"from {', '.join(insurer_list)}")
        if filters.get("insurance_type"):
            type_list = filters["insurance_type"] if isinstance(filters["insurance_type"], list) else [filters["insurance_type"]]
            filter_parts.append(f"in {', '.join(type_list)} category")
        
        filter_desc = " ".join(filter_parts) if filter_parts else ""
        
        system_prompt = (
            "Present the following insurance plans in a clear, friendly manner.\n"
            "RULES:\n"
            "- ONLY include plans from the list below\n"
            "- Group by insurer if multiple insurers present\n"
            "- Use bullet points for clarity\n"
            "- Do NOT mention technical details about data retrieval"
        )
        
        prompt = f"User asked: {query}\n\nAvailable plans {filter_desc}:\n{plans_str}"
        
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=prompt)])
        answer = getattr(response, 'content', str(response))
        
        return {"context": [f"Plans: {plans}"], "answer": answer}

    # =========================================================================
    # NODE 8: Plan Details Agent (Retrieval Agent)
    # =========================================================================
    def retrieval_agent(self, state: AgentState) -> Dict[str, Any]:
        """
        Provides detailed information about a specific plan.
        Grounds all responses in retrieved documents.
        """
        llm = LLMFactory.get_llm("medium")
        query = state["input"]
        context = state.get("context", [])
        
        if not context:
            # Fallback retrieval
            retriever = self._get_retriever()
            if retriever:
                docs = retriever.search(query, k=5)
                context = [f"[{d.metadata.get('product_name')}] {d.page_content}" for d in docs]
        
        context_str = "\n\n".join(context)
        
        system_prompt = f"""You are an Insurance Policy Specialist providing accurate information.

{COMPLIANCE_RULES}

Answer the user's question using ONLY the Policy Context provided to you.
If information is not in the context, say "I don't have that specific information in our documents."
DO NOT mention that you are looking at documents or context. Just provide the answer.
Be warm and helpful while maintaining accuracy."""
        
        prompt = f"Policy Context:\n{context_str}\n\nUser Question: {query}"
        
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=prompt)])
        answer = getattr(response, 'content', str(response))
        
        return {"answer": answer}

    # =========================================================================
    # NODE 9: Comparison Agent
    # =========================================================================
    def comparison_agent(self, state: AgentState) -> Dict[str, Any]:
        """
        Generates structured side-by-side comparisons.
        Normalizes attributes across plans.
        """
        llm = LLMFactory.get_llm("medium")
        query = state["input"]
        context = state.get("context", [])
        chunks_by_plan = state.get("retrieved_chunks", {})
        
        # Get plan names being compared
        plan_names = list(chunks_by_plan.keys()) if chunks_by_plan else []
        
        if not context and not plan_names:
            return {"answer": "I couldn't find the plans you want to compare. Please specify the plan names."}
        
        context_str = "\n\n".join(context)
        plans_info = f"\n\nPlans to compare: {', '.join(plan_names)}" if plan_names else ""
        
        system_prompt = f"""You are an Insurance Comparison Expert.

{COMPLIANCE_RULES}

COMPARISON FORMAT:
- Return comparison as a Markdown TABLE
- Columns: Features | Plan 1 | Plan 2 | ...
- Rows: Plan Type, Eligibility, Sum Assured, Premium Terms, Key Benefits, Exclusions
- If a detail is missing, put "Not specified"
- Include ALL plans mentioned in the context
- Be objective and factual"""
        
        prompt = f"Policy Context:\n{context_str}{plans_info}\n\nUser Question: {query}"
        
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=prompt)])
        answer = getattr(response, 'content', str(response))
        
        return {"answer": answer, "reasoning_output": f"Compared {len(plan_names)} plans"}

    # =========================================================================
    # NODE 10: Recommendation Agent (Advisory)
    # =========================================================================
    def advisory_agent(self, state: AgentState) -> Dict[str, Any]:
        """
        Provides personalized recommendations based on user profile.
        Grounds all advice in retrieved documents.
        """
        llm = LLMFactory.get_llm("large")
        query = state["input"]
        context = state.get("context", [])
        entities = state.get("extracted_entities", {})
        user_profile = entities.get("user_profile", {})
        
        context_str = "\n\n".join(context) if context else "No specific plans found matching your criteria."
        
        profile_info = ""
        if user_profile:
            profile_parts = []
            if user_profile.get("age"):
                profile_parts.append(f"Age: {user_profile['age']}")
            if user_profile.get("smoker") is not None:
                profile_parts.append(f"Smoker: {'Yes' if user_profile['smoker'] else 'No'}")
            if user_profile.get("cover_amount"):
                profile_parts.append(f"Cover needed: {user_profile['cover_amount']}")
            if user_profile.get("goal"):
                profile_parts.append(f"Goal: {user_profile['goal']}")
            if profile_parts:
                profile_info = f"\n\nUser Profile: {', '.join(profile_parts)}"
        
        system_prompt = f"""You are an Expert Insurance Advisor.

{COMPLIANCE_RULES}

RECOMMENDATION RULES:
- Base recommendations ONLY on plans in the context
- Consider user's age, smoking status, cover requirement if provided
- Explain WHY a plan suits them based on document features
- List 2-3 suitable options if available
- Be clear about eligibility criteria
- DO NOT reference the "context" or "documents" in your answer. Provide the advice directly."""
        
        prompt = f"Policy Context:\n{context_str}{profile_info}\n\nUser Question: {query}"
        
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=prompt)])
        answer = getattr(response, 'content', str(response))
        
        return {"answer": answer}

    # =========================================================================
    # NODE 11: General Query Agent (FAQ)
    # =========================================================================
    def faq_agent(self, state: AgentState) -> Dict[str, Any]:
        """
        Handles general insurance questions.
        Still attempts to ground in documents when possible.
        """
        llm = LLMFactory.get_llm("small")
        query = state["input"]
        context = state.get("context", [])
        
        context_str = "\n\n".join(context) if context else ""
        
        system_prompt = f"""You are an Insurance Helpdesk Assistant.

{COMPLIANCE_RULES}

For general insurance terminology questions:
- Provide accurate, helpful explanations
- If context is available, use it to give specific examples
- Keep explanations simple and jargon-free"""
        
        prompt = f"Context (if relevant):\n{context_str}\n\nUser Question: {query}" if context_str else f"User Question: {query}"
        
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=prompt)])
        answer = getattr(response, 'content', str(response))
        
        return {"answer": answer}

    # =========================================================================
    # NODE 12: Guardrail
    # =========================================================================
    def guardrail_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Final validation and compliance disclaimer.
        - Validates answer is grounded
        - Adds compliance disclaimer
        - Blocks hallucinated content
        """
        answer = state.get("answer", "")
        
        if not answer:
            answer = "I apologize, but I couldn't generate a response. Please try rephrasing your question."
        
        # Add compliance disclaimer
        if COMPLIANCE_DISCLAIMER not in answer:
            answer = answer + COMPLIANCE_DISCLAIMER
        
        return {"answer": answer}

    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    def _list_plans_from_index(self, filters: Dict = None) -> List[str]:
        """Returns unique product names matching filters. Optimized with caching."""
        retriever = self._get_retriever()
        if not retriever:
            return []
        
        try:
            # Use a simple cache attribute on the instance if it doesn't exist
            if not hasattr(self, "_cached_plans") or self._cached_plans is None:
                store = retriever.vector_store
                plans_metadata = []
                for doc in store.docstore._dict.values():
                    p_name = doc.metadata.get('product_name')
                    insurer = doc.metadata.get('insurer')
                    i_type = doc.metadata.get('insurance_type')
                    if p_name:
                        plans_metadata.append({
                            "product_name": p_name,
                            "insurer": insurer,
                            "insurance_type": i_type
                        })
                self._cached_plans = plans_metadata

            # Filter from cache
            plans = set()
            for meta in self._cached_plans:
                if filters:
                    match = True
                    for k, v in filters.items():
                        doc_val = str(meta.get(k, "")).lower().strip()
                        if not doc_val:
                            match = False
                            break
                            
                        # Standardize filter values to list of lowercase strings
                        filter_values = v if isinstance(v, list) else [v]
                        filter_values = [str(fv).lower().strip() for fv in filter_values]
                        
                        # Robust match: any filter item matches or is matched by doc_val
                        val_match = False
                        for fv in filter_values:
                            if k == "product_name":
                                if fv in doc_val or doc_val in fv:
                                    val_match = True
                                    break
                            elif k == "insurer": # Strictly match insurer names
                                if fv == doc_val:
                                    val_match = True
                                    break
                            else: # For other keys like insurance_type, allow exact match
                                if fv == doc_val:
                                    val_match = True
                                    break
                        
                        if not val_match:
                            match = False
                            break
                            
                    if not match:
                        continue
                plans.add(meta["product_name"])
            
            return sorted(list(plans))
        except Exception:
            return []

    def _find_closest_plan_name(self, query_plan: str, all_plans: List[str]) -> Optional[str]:
        """Finds closest matching plan name using fuzzy matching."""
        if not all_plans:
            return query_plan
            
        def normalize(s):
            return s.lower().replace(" ", "").replace("-", "").replace("_", "").replace("edelweisslife", "edelweiss")
            
        query_norm = normalize(query_plan)
        
        # 1. Exact match (case insensitive)
        for plan in all_plans:
            if plan.lower() == query_plan.lower():
                return plan
                
        # 2. Normalized containment match (High Confidence)
        # Check if the plan name is mentioned in the query
        for plan in all_plans:
            plan_norm = normalize(plan)
            if plan_norm in query_norm or query_norm in plan_norm:
                return plan
        
        # 3. Word overlap (Lower Confidence fallback)
        query_words = set(query_plan.lower().split())
        stop_words = {"tata", "aia", "edelweiss", "life", "generali", "central", "plan", "insurance", "the", "a", "of", "with", "compare"}
        query_significant = query_words - stop_words
        
        best_match = None
        max_overlap = 0
        
        for plan in all_plans:
            plan_words = set(plan.lower().split())
            plan_significant = plan_words - stop_words
            
            # Count significant word overlap
            overlap = len(query_significant.intersection(plan_significant))
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = plan
                
        # Return best match if we found significant overlap (at least 2 words)
        return best_match if max_overlap >= 2 else query_plan


# Singleton instance
nodes = AgentNodes()

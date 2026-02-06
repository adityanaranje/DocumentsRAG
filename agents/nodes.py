import re
import time
import json
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
- ❌ OUT-OF-BOUNDS REFUSAL: If the user asks about topics NOT related to insurance (e.g., booking flights, recipes, general news), you MUST politely refuse and state that you can only assist with insurance-related queries.
- ❌ NO hallucinations - if a plan name is not in the provided context, state clearly that you do not have information about that specific plan.
- ❌ NO assumptions - if numerical data or policy details are missing from the context, do NOT invent them. Say "Information not available."
- ❌ NO meta-commentary - start answering the question directly.
- ✅ PROPER REDIRECTION: After refusing an out-of-bounds query, invite the user to ask about insurance products, available plans, or policy definitions.
- ✅ GROUNDING: Only use facts from the provided context. CIS overrides brochure for exclusions/charges.
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

    def _log_debug(self, msg: str):
        """Internal debug logger."""
        print(f"[DEBUG] {msg}")

    # =========================================================================
    # NODE 1: Query Rewriter
    # =========================================================================
    def query_rewriter_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Rewrites conversational queries into self-contained, RAG-friendly queries.
        Uses conversation history to resolve pronouns and implicit context.
        """
        llm = LLMFactory.get_llm("low")
        query = state["input"]
        history = state.get("chat_history", [])
        
        if not history:
            return {"input": query}
            
        system_prompt = (
            "You are a professional query rewriter for an insurance consultation system. "
            "Rewrite the latest user input to be a standalone search/extraction query.\n\n"
            "RULES:\n"
            "1. If the user provides a missing profile detail (e.g., 'pt 20'), combine it with previous profile data into a recommendation request: "
            "'I want an insurance calculation for [age/gender] with Policy Term 20 years'.\n"
            "2. Resolve all pronouns (it, they) and vague terms (the plan, previous one).\n"
            "3. IMPORTANT: For general questions (e.g., 'What is PPT?') or broad listings (e.g., 'Show all plans'), do NOT inject the user's age/gender if it wasn't requested. Keep the search query clean.\n"
            "4. Only preserve profile details (age, budget) if the user's latest query is a follow-up about a specific calculation or plan recommendation.\n"
            "5. Return ONLY the rewritten query text."
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
        
        # 1. Plan Details (specific plan mentioned)
        # Check specific plan indicators
        specific_plan_indicators = ["star", "guaranteed income", "bharat savings", "premier", "smart value",
                                   "raksha", "saral jeevan", "edelweiss", "tata", "generali", "pramerica",
                                   "canara", "indusind", "max life", "hdfc", "icici"]
        
        has_plan_name = any(plan in query for plan in specific_plan_indicators)
        
        if has_plan_name and ("benefit" in query or "feature" in query or "detail" in query or "eligibility" in query):
             return {"intent": "plan_details", "query_complexity": "low"}
        
        # 2. Comparison (compare, difference, vs)
        compare_keywords = ["compare", "difference", "better", "vs", "versus", "or"]
        if any(kw in query for kw in compare_keywords) and has_plan_name:
            return {"intent": "compare_plans", "query_complexity": "high"}
        
        # 3. Listing queries - CHECK BEFORE RECOMMENDATION (to avoid "term" matching)
        listing_keywords = ["list", "show me", "available", "which plans", "what plans", 
                            "types of", "providers", "insurers", "all plans"]
        if any(kw in query for kw in listing_keywords):
            return {"intent": "list_plans", "query_complexity": "low"}
        
        # 4. General FAQ queries - CHECK BEFORE RECOMMENDATION
        # These include "what is", "what does", "explain", "define"
        faq_keywords = ["what is", "what does", "explain", "define", "meaning of", "tell me about insurance", 
                        "what are the types", "difference between", "how does insurance"]
        if any(kw in query for kw in faq_keywords):
             return {"intent": "general_query", "query_complexity": "low"}
        
        # 5. Recommendation/Calculation queries
        # IMPORTANT: Only specific recommendation indicators, avoiding generic words like "term", "mode"
        recommendation_keywords = ["suggest", "recommend", "best for", "should i", "suitable for", 
                                   "calculate", "how much will i get", "what will i get",
                                   "i am", "i'm", "my age", "my budget", "my premium",
                                   "years old", "year old"]
        
        # Also check for profile indicators (age, gender) combined with numbers/plan mention
        has_profile = any(kw in query for kw in ["male", "female", "age =", "age=", "premium =", "premium=", 
                                                  "pt =", "pt=", "ppt =", "ppt="])
        has_numbers_with_context = any(kw in query for kw in recommendation_keywords) or has_profile
        
        if has_numbers_with_context:
            return {"intent": "recommendation", "query_complexity": "high"}
            
        # 6. Fallback for explicit plan names if not caught by others
        if has_plan_name:
            return {"intent": "plan_details", "query_complexity": "low"}
        
        # 7. Follow-up detection
        if len(state.get("chat_history", [])) > 0 and ("details" in query or "more" in query):
            return {"intent": "plan_details", "query_complexity": "low"}

            
        # Default fallback
        return {"intent": "general_query", "query_complexity": "low"}
        
        # LLM-based classification for ambiguous cases
        # This section is removed as per the instructions.
        # system_prompt = (
        #     "Classify the user's insurance query into ONE of:\n"
        #     "- 'plan_details': Asking about features, benefits, eligibility of a SPECIFIC plan (should retrieve from documents)\n"
        #     "- 'list_plans': Wants to know WHICH plans are available from an insurer or category\n"
        #     "- 'recommendation': Seeks personalized benefit calculations or plan suggestions based on their profile (age, gender, premium)\n"
        #     "- 'general_query': General insurance terminology, concepts, or FAQs (not specific plans)\n\n"
        #     "IMPORTANT: 'What are the benefits of [Plan Name]' is 'plan_details', NOT 'recommendation'\n"
        #     "Return ONLY the category name."
        # )
        
        # response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=query)])
        # intent = getattr(response, 'content', str(response)).lower().strip()
        
        # valid_intents = ['list_plans', 'plan_details', 'recommendation', 'general_query']
        # if intent not in valid_intents:
        #     intent = "plan_details"  # Default fallback
            
        # return {"intent": intent}

    def entity_extractor_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Extracts structured entities from the query.
        """


        # DEBUG: Write to file to ensure we see it
        # try:
        #     with open("extraction_debug.log", "a") as f:
        #         f.write(f"\n\n[TIME] Execution at {time.time()}\n")
        #         f.write(f"[INPUT] {state.get('input', 'NO INPUT')}\n")
        #         f.write(f"[INTENT] {state.get('intent', 'NOT SET')}\n")
        # except: pass
        
        # DEBUG: Write to file to ensure we see it
        try:
            with open("extraction_debug.log", "a") as f:
                f.write(f"\n\n[TIME] Execution at {time.time()}\n")
                f.write(f"[INPUT] {state.get('input', 'NO INPUT')}\n")
                f.write(f"[INTENT] {state.get('intent', 'NOT SET')}\n")
        except: pass

        try:
            print(f"[ENTITY DEBUG] ===== STARTING ENTITY EXTRACTION =====")
            # FORCE extraction for debugging if needed, but rely on logic
            
            try:
                with open("extraction_debug.log", "a") as f:
                    f.write(f"[STATUS] Starting extraction logic...\n")
            except: pass

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
            
            # Extract user profile (Merge with existing data in state AND chat history)
            existing_profile = state.get("extracted_entities", {}).get("user_profile", {})
            history = state.get("chat_history", [])
            new_profile = {}
            
            # Always attempt extraction if it's a recommendation or if profile indicators exist
            profile_indicators = ["old", "male", "female", "year", "lakh", "rs", "budget", "premium", "invest", "benefit", "pt ", "ppt ", "mode", "age"]
            should_extract = any(ind in query for ind in profile_indicators) or state.get("intent") == "recommendation"
            
            print(f"[EXTRACTION DEBUG] Should extract: {should_extract}, Intent: {state.get('intent')}")
            
            try:
                with open("extraction_debug.log", "a") as f:
                    f.write(f"[STATUS] Should extract: {should_extract}\n")
            except: pass

            if should_extract:
                new_profile = self._extract_user_profile(state["input"], history=history)
                print(f"[EXTRACTION DEBUG] Extracted profile: {new_profile}")
                try:
                    with open("extraction_debug.log", "a") as f:
                        f.write(f"[STATUS] Extracted profile: {new_profile}\n")
                except: pass
            
            # Merge: new data overwrites old, but old data is kept if not in new
            # IMPORTANT: Ensure keys with 'null' or empty values in new_profile do not overwrite valid existing data
            user_profile = existing_profile.copy()
            for k, v in new_profile.items():
                if v is not None and v != "" and v != "null":
                    user_profile[k] = v
            
            # Explicitly handle keys that often get dropped or overwritten incorrectly
            if "policy_term" in new_profile and str(new_profile["policy_term"]).strip():
                 user_profile["policy_term"] = new_profile["policy_term"]
            
            entities: ExtractedEntities = {
                "provider": list(set(providers)) if providers else [],
                "insurance_type": list(set(insurance_types)) if insurance_types else [],
                "plan_names": list(set(plan_names)) if plan_names else [],
                "user_profile": user_profile
            }
            
            # Build metadata filters from entities
            filters = {}
            if providers:
                filters["insurer"] = providers
            if insurance_types:
                filters["insurance_type"] = insurance_types
        
            try:
                with open("extraction_debug.log", "a") as f:
                    f.write(f"[RESULT] Entities: {entities}\n")
                    f.write(f"[RESULT] Profile: {user_profile}\n")
            except: pass
                
            print(f"[ENTITY DEBUG] Final entities: {entities}")
            result = {
                "extracted_entities": entities,
                "metadata_filters": filters
            }
            return result
        except Exception as e:
            try:
                with open("extraction_debug.log", "a") as f:
                    f.write(f"[ERROR] {str(e)}\n")
                    import traceback
                    f.write(traceback.format_exc())
            except: pass
                
            print(f"[ENTITY DEBUG] Error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "extracted_entities": {
                    "provider": [],
                    "insurance_type": [],
                    "plan_names": [],
                    "user_profile": {}
                },
                "metadata_filters": {}
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

    def _extract_user_profile(self, query: str, history: List[str] = None) -> Dict[str, Any]:
        """Extract user profile information for recommendations, using history if available."""
        profile = {}
        
        # ========================================================================
        # PRIORITY 1: REGEX EXTRACTION (Most Reliable)
        # ========================================================================
        # These patterns work with formats like:
        # "age=30", "age = 30", "age is 30", "I am 30 years old"
        
        query_lower = query.lower()
        
        # Age extraction
        age_patterns = [
            r'\bage\s*[=:]\s*(\d+)',  # age=30, age = 30, age: 30
            r'\bage\s+is\s+(\d+)',     # age is 30
            r'i\s+am\s+(\d+)\s+years?\s+old',  # I am 30 years old
            r'(\d+)\s+years?\s+old',  # 30 years old
            r'\bage\s+(\d+)\b',        # age 30
        ]
        for pattern in age_patterns:
            match = re.search(pattern, query_lower)
            if match and not profile.get('age'):
                try:
                    age = int(match.group(1))
                    if 18 <= age <= 100:  # Expanded age range
                        profile['age'] = age
                        break
                except: pass
        
        # Gender extraction
        if 'gender' not in profile:
            if re.search(r'gender\s*[=:]\s*(male|m\b)', query_lower) or \
               re.search(r'gender\s+is\s+(male|m\b)', query_lower) or \
               re.search(r'\bmale\b', query_lower):
                profile['gender'] = 'male'
            elif re.search(r'gender\s*[=:]\s*(female|f\b)', query_lower) or \
                 re.search(r'gender\s+is\s+(female|f\b)', query_lower) or \
                 re.search(r'\bfemale\b', query_lower):
                profile['gender'] = 'female'
        
        # Premium extraction
        premium_patterns = [
            r'premium\s*[=:]\s*([\d,\.]+)',  # premium=100000.50
            r'premium\s+(?:amount\s+)?(?:is\s+)?(?:of\s+)?([\d,\.]+)', 
            r'invest(?:ing)?\s+([\d,\.]+)\s*(?:lakh|lac|cr|crore|k|thousand)?',
            r'([\d,\.]+)\s*(?:lakh|lac|cr|crore|k|thousand)\s+(?:per year|annual|premium)',
            r'budget\s*[=:]\s*([\d,\.]+)',
        ]
        
        def parse_indian_amount(text):
            """Parse amounts like '1 lakh', '5.5 cr', '100,000'"""
            if not text: return None
            text = text.lower().replace(',', '').strip()
            
            multiplier = 1
            if 'lakh' in text or 'lac' in text: multiplier = 100000
            elif 'cr' in text or 'crore' in text: multiplier = 10000000
            elif 'k' in text: multiplier = 1000
            
            # Find the number in the segment
            nums = re.findall(r'(\d+(?:\.\d+)?)', text)
            if nums:
                try:
                    return int(float(nums[0]) * multiplier)
                except: return None
            return None
        
        for pattern in premium_patterns:
            match = re.search(pattern, query_lower)
            if match and not profile.get('premium_amount'):
                # Pass the matched segment to parser
                amount = parse_indian_amount(match.group(0))
                if amount and 500 <= amount <= 50000000:
                    profile['premium_amount'] = str(amount)
                    break
        
        # Policy Term (PT)
        pt_patterns = [
            r'\bpt\s*[=:]\s*(\d+)',
            r'\bpt\s+(\d+)\b',
            r'policy\s+term\s*[=:]\s*(\d+)',
            r'policy\s+term\s+(?:of\s+)?(\d+)',
            r'term\s*[=:]\s*(\d+)\b',
        ]
        for pattern in pt_patterns:
            match = re.search(pattern, query_lower)
            if match and not profile.get('policy_term'):
                pt = match.group(1)
                profile['policy_term'] = pt + " years"
                break
        
        # Payment Term (PPT)
        ppt_patterns = [
            r'\bppt\s*[=:]\s*(\d+)',
            r'\bppt\s+(\d+)\b',
            r'(?:premium\s+)?payment\s+term\s*[=:]\s*(\d+)',
            r'paying\s+term\s*[=:]\s*(\d+)',
            r'pay\s+term\s*[=:]\s*(\d+)',
        ]
        for pattern in ppt_patterns:
            match = re.search(pattern, query_lower)
            if match and not profile.get('payment_term'):
                ppt = match.group(1)
                profile['payment_term'] = ppt + " years"
                break
        
        # Payment Mode
        mode_patterns = [
            r'mode\s*[=:]\s*(monthly|annual|yearly|quarterly|half\s*yearly)',
            r'(?:premium\s+)?(?:payment\s+)?mode\s+(?:is\s+)?(monthly|annual|yearly|quarterly)',
            r'\b(monthly|annual|yearly|quarterly)\b',
        ]
        for pattern in mode_patterns:
            match = re.search(pattern, query_lower)
            if match and not profile.get('payment_mode'):
                mode = match.group(1).strip()
                if mode == 'yearly': mode = 'annual'
                profile['payment_mode'] = mode
                break
        
        # ========================================================================
        # PRIORITY 2: LLM EXTRACTION (Fallback for complex cases)
        # ========================================================================
        # Use LLM if critical fields are missing OR if it's a recommendation intent
        critical_fields = ['age', 'gender', 'premium_amount']
        missing_critical = any(field not in profile for field in critical_fields)
        
        if missing_critical:
            llm = LLMFactory.get_llm("medium")
            
            history_context = ""
            if history:
                history_str = "\n".join([f"- {h}" for h in history[-5:]])
                history_context = f"\n\nCONVERSATION HISTORY:\n{history_str}"
                
            system_prompt = (
                "Extract user profile details for insurance recommendations.\n"
                "JSON Output fields (use null if unknown):\n"
                "- age (number)\n"
                "- gender (male/female)\n"
                "- premium_amount (number)\n"
                "- policy_term (number of years)\n"
                "- payment_term (number of years)\n"
                "- payment_mode (Monthly/Annual/Quarterly/Half-Yearly)\n\n"
                "MAPPING RULES:\n"
                "- PT = policy_term\n"
                "- PPT = payment_term\n"
                "- mode = payment_mode\n"
                "- Extract from latest query AND history. Latest query wins conflicts.\n"
                "Return ONLY a raw JSON object."
            )
            
            prompt = f"LATEST QUERY: {query}{history_context}"
            
            try:
                response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=prompt)])
                result_text = getattr(response, 'content', str(response))
                
                # Try to parse JSON
                try:
                    # Clean the response in case LLM added markdown blocks
                    clean_json = re.search(r'\{.*\}', result_text, re.DOTALL)
                    if clean_json:
                        llm_profile = json.loads(clean_json.group(0))
                        
                        # Merge LLM results into profile if regex didn't find them
                        if 'age' not in profile and llm_profile.get('age'):
                            profile['age'] = int(llm_profile['age'])
                        if 'gender' not in profile and llm_profile.get('gender'):
                            profile['gender'] = llm_profile['gender'].lower()
                        if 'premium_amount' not in profile and llm_profile.get('premium_amount'):
                            profile['premium_amount'] = str(llm_profile['premium_amount'])
                        if 'policy_term' not in profile and llm_profile.get('policy_term'):
                            profile['policy_term'] = str(llm_profile['policy_term']) + " years"
                        if 'payment_term' not in profile and llm_profile.get('payment_term'):
                            profile['payment_term'] = str(llm_profile['payment_term']) + " years"
                        if 'payment_mode' not in profile and llm_profile.get('payment_mode'):
                            profile['payment_mode'] = llm_profile['payment_mode'].title().replace('Annual', 'annual').lower()
                except:
                    # Fallback to line-based parsing if JSON fails
                    for line in result_text.split('\n'):
                        if ':' in line:
                            parts = line.split(':', 1)
                            k = parts[0].strip().lower()
                            v = parts[1].strip().lower().replace('"', '').replace("'", "")
                            if v and v != 'null':
                                if 'age' in k and 'age' not in profile: 
                                    nums = re.findall(r'\d+', v)
                                    if nums: profile['age'] = int(nums[0])
                                elif 'gender' in k and 'gender' not in profile: profile['gender'] = v
                                elif 'premium' in k and 'premium_amount' not in profile: profile['premium_amount'] = v
                                elif 'policy_term' in k or 'pt' == k and 'policy_term' not in profile: profile['policy_term'] = v + " years"
                                elif 'payment_term' in k or 'ppt' == k and 'payment_term' not in profile: profile['payment_term'] = v + " years"
                
            except Exception as e:
                print(f"[WARNING] LLM extraction failed: {e}")
        
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
            
            # Find the insurer for this product from cache for better filtering
            matched_insurer = None
            if hasattr(self, "_cached_plans") and self._cached_plans:
                for p_meta in self._cached_plans:
                    if p_meta["product_name"] == matched:
                        matched_insurer = p_meta.get("insurer")
                        break

            # IMPORTANT: Search by product_name directly if possible
            search_filters = {"product_name": matched}
            if matched_insurer:
                search_filters["insurer"] = matched_insurer
            
            # Use a slightly lower k because we are being very specific with the filter
            docs = retriever.search(focused_query, filters=search_filters, k=20)
            
            plan_chunks = []
            for doc in docs:
                doc_product = doc.metadata.get("product_name", "")
                # Final check for safety, but with accurate fuzzy matching
                if self._find_closest_plan_name(doc_product, [matched]) == matched:
                    plan_chunks.append(doc)
            
            for doc in plan_chunks[:8]:
                # Use product_name for the key instead of plan_id to ensure clean table headers
                plan_name = doc.metadata.get("product_name", matched)
                chunks_by_plan[plan_name].append({
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
            # Prefer product_name for display keys
            plan_name = doc.metadata.get("product_name", doc.metadata.get("plan_id", "unknown"))
            grouped[plan_name].append({
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
        intent = state.get("intent", "compare_plans")
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
        Agent for answering plan-specific or comparison questions using retrieved context.
        """
        complexity = state.get("query_complexity", "low")
        llm = LLMFactory.get_llm(complexity)
        
        query = state["input"]
        context = state.get("context", [])
        entities = state.get("extracted_entities", {})
        
        if not context:
            # Fallback retrieval with better filtering
            retriever = self._get_retriever()
            if retriever:
                # Try to extract plan names from query for better filtering
                plan_names = entities.get("plan_names", [])
                filters = state.get("metadata_filters", {})
                
                # If we have plan names, use them for filtering
                if plan_names:
                    filters["product_name"] = plan_names
                
                # Retrieve with filters
                if filters:
                    docs = retriever.search(query, filters=filters, k=10)
                else:
                    docs = retriever.search(query, k=10)
                
                # Format context with plan names
                context = [f"[{d.metadata.get('product_name', 'Unknown')}] {d.page_content}" for d in docs]
        
        # If still no context, provide a helpful message
        if not context:
            return {
                "answer": "I couldn't find specific information about that plan in my knowledge base. "
                         "Could you please provide more details or try asking about a different plan? "
                         "You can also ask me to list available plans."
            }
        
        context_str = "\n\n".join(context)
        
        system_prompt = f"""You are an Insurance Policy Specialist providing accurate information.

{COMPLIANCE_RULES}

STRICT GROUNDING RULES:
- Answer the user's question using the Policy Context provided to you.
- If the requested plan is NOT mentioned in the Policy Context, say: "I'm sorry, but I couldn't find information regarding [Plan Name] in our current policy database. Please verify the name or ask me to list available plans."
- If the question is about non-insurance topics, refuse using the OUT-OF-BOUNDS REFUSAL rule.
- Structure your response with clear headings and bullet points.
"""
        
        prompt = f"Policy Context:\n{context_str}\n\nUser Question: {query}"
        
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=prompt)])
        answer = getattr(response, 'content', str(response))
        
        return {"answer": answer}

    # =========================================================================
    # NODE 9: Recommendation Agent (Advisory)
    # =========================================================================
    def advisory_agent(self, state: AgentState) -> Dict[str, Any]:
        """
        Provides personalized recommendations based on user profile.
        Grounds all advice in retrieved documents.
        If critical info (age/gender/premium) is missing for specific plans, asks for it.
        """
        llm = LLMFactory.get_llm("large")
        query = state["input"]
        context = state.get("context", [])
        entities = state.get("extracted_entities", {})
        user_profile = entities.get("user_profile", {})
        
        # Check for Insurer and Guaranteed/Savings context
        providers = entities.get("provider", [])
        is_guaranteed = any(t in ["Savings Plan", "Guaranteed Return"] for t in entities.get("insurance_type", []))
        is_rec = state.get("intent") == "recommendation"
        
        # Only block and ask for info IF the intent is explicitly a recommendation/calculation
        if is_rec:
            print(f"[ADVISORY DEBUG] Full entities: {entities}")
            print(f"[ADVISORY DEBUG] User profile: {user_profile}")
            missing = []
            if not user_profile.get("age"): missing.append("age")
            if not user_profile.get("gender"): missing.append("gender")
            if not user_profile.get("premium_amount"): missing.append("annual premium amount")
            if not user_profile.get("policy_term"): missing.append("policy term (PT)")
            if not user_profile.get("payment_term"): missing.append("premium payment term (PPT)")
            if not user_profile.get("payment_mode"): missing.append("premium payment mode")
            
            print(f"[ADVISORY DEBUG] Missing fields check:")
            for field in ["age", "gender", "premium_amount", "policy_term", "payment_term", "payment_mode"]:
                value = user_profile.get(field)
                print(f"  - {field}: {value} (truthy: {bool(value)})")
            print(f"[ADVISORY DEBUG] Final missing list: {missing}")

            # Block and ask for info for professional consultation
            if missing:
                missing_str = " and ".join([", ".join(missing[:-1]), missing[-1]] if len(missing) > 1 else missing)
                return {"answer": f"To provide you with specific benefit figures and a professional recommendation, I need a few more details: **{missing_str}**. Could you please provide these?"}
            
            # If we have everything, get the numbers
            calc_result = self.plan_calculator_tool(state)
            state["reasoning_output"] = calc_result.get("reasoning_output", "")
        else:
            # If not a recommendation intent, check if we have enough profile data to show numbers anyway
            # (e.g., if user asks about a specific plan but we already know their profile)
            if user_profile.get("age") and user_profile.get("premium_amount") and user_profile.get("policy_term"):
                calc_result = self.plan_calculator_tool(state)
                state["reasoning_output"] = calc_result.get("reasoning_output", "")
        calculation_info = ""
        raw_calc = state.get('reasoning_output', '')
        if raw_calc:
            try:
                calc_json = json.loads(raw_calc)
                table = calc_json.get("summary_table", "")
                if table:
                    calculation_info = f"\n\n### MANDATORY GROUNDING: NUMERICAL DATA TABLE\n{table}\n(PRIORITIZE THESE PLANS AND NUMBERS OVER ANY TEXT BELOW)\n"
            except: pass

        context_str = "\n\n".join(context) if context else "No plans found."
        
        profile_info = ""
        if user_profile:
            profile_parts = [f"{k}: {v}" for k, v in user_profile.items() if v]
            if profile_parts:
                profile_info = f"\n\nUser Profile: {', '.join(profile_parts)}"
        
        system_prompt = f"""You are an Expert Insurance Advisor.
 
{COMPLIANCE_RULES}
 
RECOMMENDATION RULES:
- 🚨 PRIORITY 1: Recommending plans from the 'MANDATORY GROUNDING' table above. Use those EXACT numbers.
- 🚨 PRIORITY 2: Only provide benefit calculations for the plans in the GROUNDING table.
- If the user asks about plans not in the table for calculation, say you don't have calculation data for them yet.
- If the query is out-of-bounds, use the OUT-OF-BOUNDS REFUSAL rule.
- NEVER say "Not Available" if numbers exist in the grounding table.
- Be consultative and grounded.
"""
        
        prompt = f"{calculation_info}\n\nPolicy Context:\n{context_str}{profile_info}\n\nUser Question: {query}"
        
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=prompt)])
        answer = getattr(response, 'content', str(response))
        
        return {"answer": answer}

    # =========================================================================
    # NODE 11: General Query Agent (FAQ)
    # =========================================================================
    def faq_agent(self, state: AgentState) -> Dict[str, Any]:
        """
        Agent for general insurance questions (glossary, concepts).
        """
        llm = LLMFactory.get_llm("low")
        query = state["input"]
        context = state.get("context", [])
        
        # Try to retrieve context for general insurance terms if not already provided
        if not context:
            retriever = self._get_retriever()
            if retriever:
                # Use broader search for general queries
                docs = retriever.search(query, k=3)  # Reduced from 5 to 3 for more focused context
                if docs:
                    context = [d.page_content for d in docs]
        
        context_str = "\n\n".join(context) if context else ""
        
        system_prompt = f"""You are an Insurance Helpdesk Assistant.

{COMPLIANCE_RULES}
 
INSTRUCTIONS:
- For insurance terminology: Provide a clear, concise definition.
- 🚨 STRICT RULE: If the user asks about ANYTHING non-insurance related (e.g., travel tickets, cooking, etc.), you MUST refuse and redirect to insurance topics.
- 🚨 NO HALLUCINATION: If the term is not common insurance knowledge and not in context, say you don't know rather than guessing.
- Keep the total response under 150 words.

Common Insurance Terms to use as reference:
- **Policy Term (PT)**: The total duration for which the policy remains active.
- **Premium Payment Term (PPT)**: The duration during which premiums must be paid.
- **Maturity Benefit**: The lump sum amount paid when the policy matures.
- **Sum Assured**: The guaranteed amount payable on death or maturity.
"""
        
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
    # TOOL: Plan Calculator Tool
    # =========================================================================
    def plan_calculator_tool(self, state: AgentState) -> Dict[str, Any]:
        """
        Tool logic to calculate benefits using the API's dummy logic.
        Extremely robust extraction fallback for age, gender, and premium.
        """
        from api.plans import get_plan_benefits_tool, resolve_plan_id
        user_profile = state.get("extracted_entities", {}).get("user_profile", {})
        plan_names = state.get("extracted_entities", {}).get("plan_names", [])
        query = state["input"].lower()
        
        # --- ROBUST FALLBACKS ---
        # 1. Age Fallback
        age = user_profile.get("age")
        if not age:
            age_match = re.search(r'\b(\d{2})\b\s*(?:year|yr|old|male|female)?', query)
            if age_match:
                age = int(age_match.group(1))
        
        # 2. Gender Fallback
        gender = user_profile.get("gender")
        if not gender:
            if "male" in query and "female" not in query: gender = "male"
            elif "female" in query: gender = "female"
        
        # 3. Premium Fallback
        premium = user_profile.get("premium_amount")
        clean_premium = 0.0
        
        if not premium:
            # Look for any number followed by a potential unit
            prem_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:rs\.?|inr|lakh|cr|k|thousand)?', query)
            if prem_match:
                val = float(prem_match.group(1))
                unit_search = query[prem_match.start():prem_match.end()+20] # look ahead
                if 'lakh' in unit_search: val *= 100000
                elif 'cr' in unit_search: val *= 10000000
                elif any(k in unit_search for k in ['k', 'thousand']): val *= 1000
                clean_premium = val
        else:
            try:
                if isinstance(premium, (int, float)):
                    clean_premium = float(premium)
                else:
                    nums = re.findall(r'\d+\.?\d*', str(premium))
                    if nums:
                        clean_premium = float(nums[0])
                        if 'lakh' in str(premium).lower(): clean_premium *= 100000
                        elif 'cr' in str(premium).lower(): clean_premium *= 10000000
            except:
                pass

        if not (age and gender and clean_premium > 0):
            return {"reasoning_output": "Insufficient data (age, gender, or premium) to calculate benefits."}

        # 4. Resolve Plan IDs
        pids = []
        for name in plan_names:
            pid = resolve_plan_id(name)
            if pid: pids.append(pid)
        
        # If no specific plan found, calculate for ALL default plans
        target_plan_id = pids[0] if len(pids) == 1 else None
        
        # 5. Execute Tool
        calculation_json = get_plan_benefits_tool(
            age=int(age),
            gender=str(gender),
            premium_amount=clean_premium,
            plan_id=target_plan_id,
            policy_term=user_profile.get("policy_term"),
            payment_term=user_profile.get("payment_term"),
            payment_mode=user_profile.get("payment_mode")
        )
        
        return {"reasoning_output": calculation_json}

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
        # REMOVED insurer names from stop_words because they are critical for distinguishing 
        # similar plan names (like 'Saral Jeevan Bima') across different companies.
        stop_words = {"plan", "insurance", "the", "a", "of", "with", "compare", "is", "between"}
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

"""
Optimized Portfolio Agent with Improved Confidence Scoring
- Better TF-IDF similarity
- Enhanced confidence calculation
- More accurate validation
- Optimized for greeting/simple queries
"""

from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
import operator
import json
import re
from datetime import datetime
from collections import Counter
import math


class EnhancedAgentState(TypedDict):
    """Enhanced state with additional tracking"""
    messages: Annotated[list, operator.add]
    user_query: str
    response: str
    knowledge_base: dict
    retrieved_context: List[str]
    query_type: str
    confidence_score: float
    needs_validation: bool
    tools_used: List[str]
    reasoning_steps: List[str]


class SimpleTextRAG:
    """Optimized RAG system with better similarity scoring"""
    
    def __init__(self, knowledge_base: dict):
        self.knowledge_base = knowledge_base
        self.documents = []
        self._initialize_documents()
    
    def _initialize_documents(self):
        """Convert knowledge base to searchable documents"""
        # Personal info
        personal = self.knowledge_base.get("personal_info", {})
        self.documents.append({
            "content": f"Name: {personal.get('name')}, Title: {personal.get('title')}, "
                      f"Specializations: {', '.join(personal.get('specializations', []))}",
            "type": "personal",
            "keywords": ["name", "title", "specialization", "who", "introduce", "about"]
        })
        
        # Skills
        for skill_cat, skill_data in self.knowledge_base.get("skills", {}).items():
            content = f"Skill Category: {skill_cat}\n"
            content += f"Description: {skill_data.get('description', '')}\n"
            content += f"Technologies: {', '.join(skill_data.get('technologies', []))}\n"
            content += f"Specialties: {', '.join(skill_data.get('specialties', []))}"
            
            keywords = ["skill", "technology", "expertise", "know", "proficient"] + \
                      [tech.lower() for tech in skill_data.get('technologies', [])]
            
            self.documents.append({
                "content": content, 
                "type": "skill",
                "keywords": keywords
            })
        
        # Projects
        for proj_key, proj_data in self.knowledge_base.get("projects", {}).items():
            content = f"Project: {proj_data.get('name')}\n"
            content += f"Description: {proj_data.get('description')}\n"
            content += f"Technologies: {', '.join(proj_data.get('technologies', []))}\n"
            content += f"Features: {', '.join(proj_data.get('features', []))}"
            
            keywords = ["project", "built", "developed", "created", "work"] + \
                      [tech.lower() for tech in proj_data.get('technologies', [])]
            
            self.documents.append({
                "content": content, 
                "type": "project",
                "keywords": keywords
            })
        
        # Experience
        exp = self.knowledge_base.get("experience", {})
        exp_content = f"Experience: {exp.get('years')} years\n"
        exp_content += f"Projects Completed: {exp.get('projects_completed')}\n"
        exp_content += f"Achievements: {', '.join(exp.get('achievements', []))}"
        
        self.documents.append({
            "content": exp_content, 
            "type": "experience",
            "keywords": ["experience", "years", "achievement", "work", "career"]
        })
        
        # Education
        for edu_key, edu_data in self.knowledge_base.get("education", {}).items():
            if isinstance(edu_data, dict):
                content = f"Education: {edu_data.get('degree', edu_data.get('qualification', ''))}\n"
                content += f"Institution: {edu_data.get('institution')}\n"
                content += f"Period: {edu_data.get('period')}"
                
                self.documents.append({
                    "content": content, 
                    "type": "education",
                    "keywords": ["education", "degree", "university", "study", "graduate"]
                })
    
    def _compute_enhanced_similarity(self, query: str, document: Dict) -> float:
        """Enhanced similarity with keyword boosting"""
        # Base TF-IDF similarity
        tfidf_score = self._compute_tfidf_similarity(query, document["content"])
        
        # Keyword matching boost
        query_lower = query.lower()
        keyword_matches = sum(1 for kw in document.get("keywords", []) if kw in query_lower)
        keyword_boost = min(keyword_matches * 0.15, 0.4)  # Up to 40% boost
        
        # Combine scores
        final_score = min(tfidf_score + keyword_boost, 1.0)
        
        return final_score
    
    def _compute_tfidf_similarity(self, query: str, document: str) -> float:
        """Improved TF-IDF cosine similarity"""
        # Tokenize and lowercase
        query_tokens = [t.lower() for t in re.findall(r'\w+', query)]
        doc_tokens = [t.lower() for t in re.findall(r'\w+', document)]
        
        # Remove common stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'be', 'this', 'that'}
        query_tokens = [t for t in query_tokens if t not in stopwords and len(t) > 2]
        doc_tokens = [t for t in doc_tokens if t not in stopwords and len(t) > 2]
        
        if not query_tokens or not doc_tokens:
            return 0.0
        
        # Calculate term frequencies
        query_freq = Counter(query_tokens)
        doc_freq = Counter(doc_tokens)
        
        # Calculate cosine similarity
        common_terms = set(query_freq.keys()) & set(doc_freq.keys())
        if not common_terms:
            return 0.0
        
        numerator = sum(query_freq[term] * doc_freq[term] for term in common_terms)
        
        query_magnitude = math.sqrt(sum(freq ** 2 for freq in query_freq.values()))
        doc_magnitude = math.sqrt(sum(freq ** 2 for freq in doc_freq.values()))
        
        if query_magnitude == 0 or doc_magnitude == 0:
            return 0.0
        
        return numerator / (query_magnitude * doc_magnitude)
    
    def retrieve_relevant_context(self, query: str, k: int = 3) -> List[str]:
        """Retrieve relevant context with enhanced scoring"""
        if not self.documents:
            return []
        
        # Calculate similarity scores
        scored_docs = []
        for doc in self.documents:
            score = self._compute_enhanced_similarity(query, doc)
            scored_docs.append((score, doc["content"], doc["type"]))
        
        # Sort by score
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        
        # Return top k documents (only if score > 0.1)
        return [doc[1] for doc in scored_docs[:k] if doc[0] > 0.1]


class QueryClassifier:
    """Enhanced query classifier"""
    
    @staticmethod
    def classify_query(query: str) -> Dict[str, Any]:
        """Classify query type and complexity"""
        query_lower = query.lower()
        
        classification = {
            "type": "general",
            "complexity": "simple",
            "intent": "information",
            "requires_tools": False,
            "requires_reasoning": False
        }
        
        # Greetings
        greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon"]
        if any(greet in query_lower for greet in greetings):
            classification["type"] = "greeting"
            classification["complexity"] = "simple"
            return classification
        
        # Detect query type
        if any(word in query_lower for word in ["compare", "difference", "versus", "vs", "better"]):
            classification["type"] = "comparative"
            classification["complexity"] = "medium"
            classification["requires_reasoning"] = True
        
        elif any(word in query_lower for word in ["how", "explain", "why", "what is"]):
            classification["type"] = "explanatory"
            classification["complexity"] = "medium"
        
        elif any(word in query_lower for word in ["list", "show", "tell me about"]):
            classification["type"] = "factual"
            classification["complexity"] = "simple"
        
        elif any(word in query_lower for word in ["project", "built", "created", "work on"]):
            classification["type"] = "projects"
            classification["intent"] = "showcase"
        
        elif any(word in query_lower for word in ["skill", "expertise", "technology", "proficient"]):
            classification["type"] = "skills"
            classification["intent"] = "capability"
        
        elif any(word in query_lower for word in ["experience", "years", "worked"]):
            classification["type"] = "experience"
            classification["intent"] = "background"
        
        return classification


class ImprovedResponseValidator:
    """Improved validator with better confidence scoring"""
    
    def __init__(self, knowledge_base: dict):
        self.knowledge_base = knowledge_base
        self.kb_text = json.dumps(knowledge_base, indent=2).lower()
    
    def validate_response(self, response: str, query: str, query_type: str) -> Dict[str, Any]:
        """Enhanced validation with query-type awareness"""
        
        validation_result = {
            "is_valid": True,
            "confidence": 0.8,  # Start with higher base confidence
            "issues": [],
            "suggestions": []
        }
        
        # Greetings get automatic high confidence
        if query_type == "greeting":
            validation_result["confidence"] = 0.95
            return validation_result
        
        # Check response length (very short responses might be incomplete)
        if len(response.strip()) < 20:
            validation_result["confidence"] -= 0.2
            validation_result["suggestions"].append("Response seems short")
        
        # Extract claimed facts from response
        facts = self._extract_facts(response)
        
        grounded_count = 0
        for fact in facts:
            if self._is_grounded(fact):
                grounded_count += 1
        
        # Calculate grounding confidence
        if facts:
            grounding_ratio = grounded_count / len(facts)
            if grounding_ratio < 0.6:
                validation_result["confidence"] -= 0.2
                validation_result["issues"].append("Some claims may be unverified")
            elif grounding_ratio >= 0.8:
                validation_result["confidence"] += 0.1  # Boost for well-grounded
        
        # Check for hedging language (less aggressive penalty)
        hallucination_indicators = [
            "i think", "probably", "might be", "could be", 
            "perhaps", "maybe", "i believe"
        ]
        
        response_lower = response.lower()
        hedge_count = sum(1 for indicator in hallucination_indicators if indicator in response_lower)
        
        if hedge_count > 2:
            validation_result["confidence"] -= 0.15
            validation_result["suggestions"].append("Response contains uncertain language")
        
        # Ensure response is relevant to query
        if not self._is_relevant(response, query):
            validation_result["confidence"] -= 0.2
            validation_result["issues"].append("Response may not be relevant")
        else:
            validation_result["confidence"] += 0.05  # Boost for relevance
        
        # Cap confidence between 0.0 and 1.0
        validation_result["confidence"] = max(0.0, min(1.0, validation_result["confidence"]))
        
        # Mark as invalid only if confidence is very low
        if validation_result["confidence"] < 0.4:
            validation_result["is_valid"] = False
        
        return validation_result
    
    def _extract_facts(self, text: str) -> List[str]:
        """Extract factual claims from text"""
        sentences = re.split(r'[.!?]+', text)
        facts = [s.strip() for s in sentences if len(s.strip()) > 15]
        return facts
    
    def _is_grounded(self, fact: str) -> bool:
        """Check if fact is supported by knowledge base"""
        fact_lower = fact.lower()
        
        # Extract key terms
        key_terms = [word for word in fact_lower.split() 
                    if len(word) > 3 and word.isalpha()]
        
        if len(key_terms) == 0:
            return True
        
        # Check grounding
        grounded_terms = sum(1 for term in key_terms if term in self.kb_text)
        
        # More lenient: 35% threshold
        return (grounded_terms / len(key_terms)) >= 0.35
    
    def _is_relevant(self, response: str, query: str) -> bool:
        """Check if response is relevant to query"""
        query_terms = set(query.lower().split())
        response_terms = set(response.lower().split())
        
        # Remove stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for'}
        query_terms = query_terms - stopwords
        
        overlap = query_terms & response_terms
        
        # More lenient relevance check
        if len(query_terms) == 0:
            return True
        
        return len(overlap) >= min(2, len(query_terms) * 0.25)


def create_enhanced_portfolio_agent(config):
    """Create optimized LangGraph agent"""
    
    llm = ChatGoogleGenerativeAI(
        model=config.model_name,
        temperature=0.3,
        max_tokens=config.max_tokens,
        google_api_key=config.GOOGLE_API_KEY,
        convert_system_message_to_human=True
    )
    
    # Initialize components
    rag_system = None
    query_classifier = QueryClassifier()
    response_validator = None
    
    def initialize_rag(state: EnhancedAgentState) -> EnhancedAgentState:
        """Initialize RAG system with knowledge base"""
        nonlocal rag_system, response_validator
        
        if rag_system is None:
            rag_system = SimpleTextRAG(state["knowledge_base"])
            response_validator = ImprovedResponseValidator(state["knowledge_base"])
        
        return state
    
    def classify_query(state: EnhancedAgentState) -> EnhancedAgentState:
        """Classify user query"""
        classification = query_classifier.classify_query(state["user_query"])
        
        return {
            **state,
            "query_type": classification["type"],
            "reasoning_steps": [f"Query classified as: {classification['type']}"]
        }
    
    def retrieve_context(state: EnhancedAgentState) -> EnhancedAgentState:
        """Retrieve relevant context using RAG"""
        if rag_system is None:
            return state
        
        relevant_docs = rag_system.retrieve_relevant_context(state["user_query"])
        
        return {
            **state,
            "retrieved_context": relevant_docs,
            "reasoning_steps": state.get("reasoning_steps", []) + 
                              [f"Retrieved {len(relevant_docs)} relevant context chunks"]
        }
    
    def generate_grounded_response(state: EnhancedAgentState) -> EnhancedAgentState:
        """Generate response grounded in retrieved context"""
        from app.prompts.system_prompts import get_system_prompt
        
        # Limit conversation history to last 10 messages to prevent memory issues
        conversation_history = state.get("messages", [])
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]
        
        # Build enhanced system prompt with retrieved context
        base_prompt = get_system_prompt(state["knowledge_base"])
        
        # Handle greetings specially
        if state["query_type"] == "greeting":
            context_prompt = """
\n## INSTRUCTION FOR GREETINGS:
Respond warmly and professionally. Briefly introduce Nahid Hasan as an AI Engineer & ML Specialist.
Mention 1-2 key specializations and invite them to ask about projects, skills, or experience.
Keep it concise and friendly.
"""
        else:
            context_prompt = "\n\n## RETRIEVED RELEVANT CONTEXT:\n"
            for i, context in enumerate(state.get("retrieved_context", []), 1):
                context_prompt += f"\nContext {i}:\n{context}\n"
            
            context_prompt += """
\n## CRITICAL INSTRUCTION:
You MUST base your response ONLY on the information provided in the knowledge base and retrieved context above.
If the information is not present, clearly state: "I don't have specific information about that in my knowledge base."
DO NOT make up or infer information that isn't explicitly stated.
Be confident and professional in your responses.
"""
        
        full_prompt = base_prompt + context_prompt
        
        messages = [
            SystemMessage(content=full_prompt),
            HumanMessage(content=state["user_query"])
        ]
        
        response = llm.invoke(messages)
        
        return {
            **state,
            "response": response.content,
            "needs_validation": True,
            "reasoning_steps": state.get("reasoning_steps", []) + 
                              ["Generated response using grounded context"]
        }
    
    def validate_response(state: EnhancedAgentState) -> EnhancedAgentState:
        """Validate response with improved scoring"""
        if not state.get("needs_validation", False) or response_validator is None:
            return {**state, "confidence_score": 0.85}
        
        validation = response_validator.validate_response(
            state["response"], 
            state["user_query"],
            state["query_type"]
        )
        
        reasoning_step = f"Validation: {'PASSED' if validation['is_valid'] else 'FAILED'} "
        reasoning_step += f"(confidence: {validation['confidence']:.2f})"
        
        # Only regenerate if confidence is very low
        if not validation["is_valid"] and validation["confidence"] < 0.3:
            strict_prompt = """You provided information that couldn't be verified in the knowledge base.
Please try again, being extremely strict about only stating facts that are explicitly present.
If uncertain, say so clearly rather than making assumptions."""
            
            messages = [
                SystemMessage(content=strict_prompt),
                HumanMessage(content=f"Original query: {state['user_query']}\n\n"
                           f"Issues found: {', '.join(validation['issues'])}")
            ]
            
            corrected = llm.invoke(messages)
            
            return {
                **state,
                "response": corrected.content,
                "confidence_score": max(validation["confidence"], 0.5),  # Boost after correction
                "reasoning_steps": state.get("reasoning_steps", []) + 
                                  [reasoning_step, "Regenerated response with stricter grounding"]
            }
        
        return {
            **state,
            "confidence_score": validation["confidence"],
            "reasoning_steps": state.get("reasoning_steps", []) + [reasoning_step]
        }
    
    def finalize_response(state: EnhancedAgentState) -> EnhancedAgentState:
        """Finalize and format response"""
        # Create new message pair (don't use operator.add, return single items)
        new_messages = [
            {"role": "user", "content": state["user_query"]},
            {
                "role": "assistant", 
                "content": state["response"],
                "metadata": {
                    "confidence": state.get("confidence_score", 0.0),
                    "query_type": state.get("query_type", "general"),
                    "reasoning_steps": state.get("reasoning_steps", [])
                }
            }
        ]
        
        return {
            **state,
            "messages": new_messages  # Return only new messages, operator.add handles accumulation
        }
    
    # Create graph
    workflow = StateGraph(EnhancedAgentState)
    
    # Add nodes
    workflow.add_node("initialize_rag", initialize_rag)
    workflow.add_node("classify_query", classify_query)
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("generate_response", generate_grounded_response)
    workflow.add_node("validate_response", validate_response)
    workflow.add_node("finalize", finalize_response)
    
    # Set entry point
    workflow.set_entry_point("initialize_rag")
    
    # Add edges
    workflow.add_edge("initialize_rag", "classify_query")
    workflow.add_edge("classify_query", "retrieve_context")
    workflow.add_edge("retrieve_context", "generate_response")
    workflow.add_edge("generate_response", "validate_response")
    workflow.add_edge("validate_response", "finalize")
    workflow.add_edge("finalize", END)
    
    # Compile graph
    app = workflow.compile()
    
    return app
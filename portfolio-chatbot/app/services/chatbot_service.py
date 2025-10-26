"""
Enhanced Chatbot Service - Fixed MemoryError & Ambiguous Query Issues
Integrates:
- RAG-based agent
- Improved validation
- Advanced query handling with defense-in-depth for ambiguity
- Response quality monitoring
- Fallback strategies
- FIXED: Proper conversation history management
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
import logging
import uuid
import re

# Import enhanced components
from app.agents.portfolio_agent import (
    create_enhanced_portfolio_agent,
    EnhancedAgentState
)
from app.services.improved_input_validation import ImprovedInputValidator
from app.services.knowledge_base import get_knowledge_base
from app.config import get_settings

# Configure logging
logger = logging.getLogger(__name__)


class EnhancedChatbotService:
    """
    Production-ready chatbot service with advanced capabilities
    """
    
    def __init__(self):
        """Initialize enhanced chatbot service"""
        self.config = get_settings()
        self.knowledge_base = get_knowledge_base()
        self.agent = create_enhanced_portfolio_agent(self.config)
        self.validator = ImprovedInputValidator()
        
        # Conversation management
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
        self.conversation_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Quality monitoring
        self.quality_metrics = {
            "total_queries": 0,
            "successful_responses": 0,
            "validation_failures": 0,
            "low_confidence_responses": 0,
            "average_confidence": 0.0,
            "total_confidence_sum": 0.0
        }
        
        logger.info("EnhancedChatbotService initialized successfully")
    
    def _create_conversation_metadata(self, conversation_id: str) -> None:
        """Initialize conversation metadata"""
        self.conversation_metadata[conversation_id] = {
            "created_at": datetime.utcnow().isoformat(),
            "message_count": 0,
            "query_types": [],
            "confidence_scores": [],
            "last_activity": datetime.utcnow().isoformat(),
            "validation_failures": 0,
            "user_satisfaction_indicators": {
                "follow_up_questions": 0,
                "query_refinements": 0,
                "positive_feedback_signals": 0,
                "negative_feedback_signals": 0
            }
        }
        logger.debug(f"Created new conversation metadata for {conversation_id}")
    
    def _update_conversation_metadata(
        self,
        conversation_id: str,
        query_type: str,
        confidence: float,
        is_valid: bool = True
    ) -> None:
        """Update conversation tracking"""
        if conversation_id not in self.conversation_metadata:
            self._create_conversation_metadata(conversation_id)
        
        metadata = self.conversation_metadata[conversation_id]
        metadata["message_count"] += 1
        metadata["query_types"].append(query_type)
        metadata["confidence_scores"].append(confidence)
        metadata["last_activity"] = datetime.utcnow().isoformat()
        
        if not is_valid:
            metadata["validation_failures"] += 1
        logger.debug(f"Updated metadata for {conversation_id}: msg_count={metadata['message_count']}, conf={confidence:.2f}")

    def _update_quality_metrics(self, confidence: float, success: bool) -> None:
        """Update service-wide quality metrics"""
        self.quality_metrics["total_queries"] += 1
        
        if success:
            self.quality_metrics["successful_responses"] += 1
        
        if confidence < 0.7:
            self.quality_metrics["low_confidence_responses"] += 1
        
        self.quality_metrics["total_confidence_sum"] += confidence
        self.quality_metrics["average_confidence"] = (
            self.quality_metrics["total_confidence_sum"] / self.quality_metrics["total_queries"]
        ) if self.quality_metrics["total_queries"] > 0 else 0.0
        logger.debug(f"Updated service metrics: total_queries={self.quality_metrics['total_queries']}, avg_conf={self.quality_metrics['average_confidence']:.2f}")

    def _handle_ambiguous_query(self, message: str, validation_metadata: Dict) -> str:
        """Provide helpful response for ambiguous or gibberish queries"""
        suggestions = [
            "Try asking about specific projects (e.g., 'Tell me about your Orani AI Assistant project')",
            "Ask about technical skills (e.g., 'What NLP technologies do you use?')",
            "Inquire about my experience (e.g., 'How many years of ML experience do you have?')"
        ]
        
        response = "I'm not sure how to answer that. To get the best response, could you please ask a more specific question about my skills, projects, or professional experience?\n\nFor example, you could ask:\n"
        response += "\n".join(f"• {suggestion}" for suggestion in suggestions)
        
        return response
    
    def _get_fallback_response(self, error_type: str, context: Dict) -> str:
        """Generate appropriate fallback response"""
        fallbacks = {
            "validation_error": "I didn't quite understand that. Could you rephrase your question about my AI/ML work, skills, or projects?",
            "processing_error": "I encountered an issue processing your request. Please try rephrasing or ask about my experience, skills, or projects.",
            "low_confidence": "I'm not entirely certain about that. To give you the most accurate information, could you ask a more specific question about my AI/ML expertise?",
            "out_of_scope": "That's outside my knowledge base about Nahid's portfolio. I can tell you about his AI/ML skills, projects, experience, and education. What would you like to know?"
        }
        
        return fallbacks.get(error_type, fallbacks["processing_error"])
    
    async def get_response(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        enable_markdown_stripping: bool = True,
        enable_advanced_features: bool = True
    ) -> Dict[str, Any]:
        """
        Get enhanced chatbot response with full features
        """
        try:
            # Step 1: Sanitize and Validate Input
            sanitized_message = self.validator.sanitize_input(message)
            
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
            if conversation_id not in self.conversations:
                self._create_conversation_metadata(conversation_id)
            
            is_valid, reason, validation_metadata = self.validator.is_valid_message(sanitized_message)
            
            if not is_valid:
                logger.warning(f"Input validation failed for conv_id {conversation_id}: {reason}")
                self._update_conversation_metadata(conversation_id, "invalid_input", 0.0, is_valid=False)
                self.quality_metrics["validation_failures"] += 1
                feedback = self.validator.get_validation_feedback(is_valid, reason, validation_metadata)
                return {
                    "response": feedback, "conversation_id": conversation_id, "error": "invalid_input",
                    "reason": reason, "metadata": validation_metadata, "timestamp": datetime.utcnow().isoformat()
                }
            
            # Step 2: Prepare Agent State with Conversation History
            conversation_history = self.conversations.get(conversation_id, [])
            if len(conversation_history) > 20: # Keep last 10 pairs
                conversation_history = conversation_history[-20:]

            initial_state = EnhancedAgentState(
                messages=conversation_history,
                user_query=sanitized_message,
                response="", knowledge_base=self.knowledge_base,
                retrieved_context=[], query_type="general",
                confidence_score=0.0, needs_validation=True,
                tools_used=[], reasoning_steps=[]
            )
            
            # Step 3: Invoke the Agent
            logger.debug(f"Invoking enhanced agent for conv_id: {conversation_id}")
            result = self.agent.invoke(initial_state)
            
            # Step 4: Post-processing and Defensive Checks
            retrieved_context = result.get("retrieved_context", [])
            query_type = result.get("query_type", "general")
            confidence = result.get("confidence_score", 0.8)
            response_text = result.get("response", "")

            # DEFENSE-IN-DEPTH: Catch ambiguous queries that slipped past validation
            if not retrieved_context and query_type == "general" and len(sanitized_message.split()) < 4:
                logger.warning(f"Query '{sanitized_message[:50]}' passed validation but was ambiguous. Overriding LLM response.")
                response_text = self._handle_ambiguous_query(sanitized_message, {})
                confidence = 0.4 # Manually lower confidence for this case
            elif confidence < 0.5:
                logger.warning(f"Low confidence response ({confidence:.2f}) for conv_id {conversation_id}: {sanitized_message[:50]}...")
                response_text = self._get_fallback_response("low_confidence", result)
            
            # Step 5: Finalize and Return Response
            if enable_markdown_stripping:
                response_text = self._strip_markdown(response_text)
            
            self.conversations[conversation_id] = result.get("messages", [])
            logger.debug(f"Conversation history for {conversation_id} updated with {len(self.conversations[conversation_id])} messages.")

            self._update_conversation_metadata(conversation_id, query_type, confidence, is_valid=True)
            self._update_quality_metrics(confidence, success=True)
            
            response_data = {
                "response": response_text,
                "conversation_id": conversation_id,
                "confidence": confidence,
                "query_type": query_type,
                "message_count": self.conversation_metadata[conversation_id]["message_count"],
                "reasoning_steps": result.get("reasoning_steps", []) if enable_advanced_features else [],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Response generated successfully for conv_id: {conversation_id} (confidence: {confidence:.2f})")
            return response_data
            
        except Exception as e:
            logger.error(f"Error in get_response for conv_id {conversation_id}: {str(e)}", exc_info=True)
            self._update_quality_metrics(0.0, success=False)
            fallback = self._get_fallback_response("processing_error", {})
            return {
                "response": f"I apologize, but an internal error occurred: {fallback}",
                "conversation_id": conversation_id or str(uuid.uuid4()),
                "error": str(e),
                "confidence": 0.0,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _strip_markdown(self, text: str) -> str:
        """Remove common markdown formatting for clean text display"""
        text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'\*([^\*]+)\*', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)
        text = re.sub(r'^\s*#{1,6}\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        text = re.sub(r'```(?:\w+)?\n([\s\S]*?)\n```', r'\1', text)
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*>\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    def get_quality_report(self) -> Dict[str, Any]:
        """Generate quality metrics report"""
        total = self.quality_metrics["total_queries"]
        if total == 0:
            return {"status": "no_data", "message": "No queries processed yet"}
        
        success_rate = (self.quality_metrics["successful_responses"] / total * 100)
        validation_failure_rate = (self.quality_metrics["validation_failures"] / total * 100)
        low_confidence_rate = (self.quality_metrics["low_confidence_responses"] / total * 100)
        
        return {
            "total_queries": total,
            "success_rate": f"{success_rate:.2f}%",
            "average_confidence": f"{self.quality_metrics['average_confidence']:.2f}",
            "validation_failure_rate": f"{validation_failure_rate:.2f}%",
            "low_confidence_rate": f"{low_confidence_rate:.2f}%",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_conversation_insights(self, conversation_id: str) -> Optional[Dict]:
        """Get detailed insights for a conversation"""
        if conversation_id not in self.conversation_metadata:
            return None
        
        metadata = self.conversation_metadata[conversation_id]
        confidence_scores = metadata.get("confidence_scores", [])
        avg_confidence = (sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0)
        
        return {
            "conversation_id": conversation_id, "created_at": metadata["created_at"],
            "last_activity": metadata["last_activity"], "message_count": metadata["message_count"],
            "query_types_distribution": self._get_distribution(metadata["query_types"]),
            "average_confidence": f"{avg_confidence:.2f}", "validation_failures": metadata["validation_failures"],
            "duration_minutes": self._calculate_duration(metadata), "engagement_level": self._assess_engagement(metadata),
            "user_satisfaction_indicators": metadata["user_satisfaction_indicators"], "timestamp": datetime.utcnow().isoformat()
        }
    
    def _get_distribution(self, items: list) -> Dict[str, int]:
        """Calculate distribution of items"""
        return {item: items.count(item) for item in set(items)}
    
    def _calculate_duration(self, metadata: Dict) -> float:
        """Calculate conversation duration in minutes"""
        try:
            created = datetime.fromisoformat(metadata["created_at"])
            last_activity = datetime.fromisoformat(metadata["last_activity"])
            return round((last_activity - created).total_seconds() / 60, 2)
        except Exception:
            logger.warning(f"Could not calculate duration for conv_id: {metadata.get('conversation_id', 'N/A')}")
            return 0.0
    
    def _assess_engagement(self, metadata: Dict) -> str:
        """Assess user engagement level"""
        msg_count = metadata["message_count"]
        duration = self._calculate_duration(metadata)
        if msg_count >= 5 and duration > 2: return "high"
        if msg_count >= 3: return "medium"
        return "low" if msg_count > 0 else "no_engagement"
    
    def get_conversation_history(self, conversation_id: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieve the full conversation history for a given ID."""
        return self.conversations.get(conversation_id)

    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear conversation data"""
        removed_conv = self.conversations.pop(conversation_id, None)
        removed_meta = self.conversation_metadata.pop(conversation_id, None)
        if removed_conv is not None or removed_meta is not None:
            logger.info(f"Cleared conversation: {conversation_id}")
            return True
        return False
    
    def cleanup_old_conversations(self, max_age_hours: int = 24) -> int:
        """Cleanup old conversations"""
        try:
            current_time = datetime.utcnow()
            to_remove = [
                conv_id for conv_id, metadata in self.conversation_metadata.items()
                if (current_time - datetime.fromisoformat(metadata.get("last_activity", ""))).total_seconds() / 3600 > max_age_hours
            ]
            for conv_id in to_remove:
                self.clear_conversation(conv_id)
            logger.info(f"Cleaned up {len(to_remove)} conversations older than {max_age_hours} hours.")
            return len(to_remove)
        except Exception as e:
            logger.error(f"Error in cleanup_old_conversations: {str(e)}", exc_info=True)
            return 0


# Singleton instance
_enhanced_service_instance = None

def get_enhanced_chatbot_service() -> EnhancedChatbotService:
    """Get singleton instance of enhanced service"""
    global _enhanced_service_instance
    if _enhanced_service_instance is None:
        _enhanced_service_instance = EnhancedChatbotService()
    return _enhanced_service_instance
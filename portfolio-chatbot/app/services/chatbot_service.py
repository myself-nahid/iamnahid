"""
Enhanced Chatbot Service - Fixed MemoryError Issue
Integrates:
- RAG-based agent
- Improved validation
- Advanced query handling
- Response quality monitoring
- Fallback strategies
- FIXED: Proper conversation history management
"""

from typing import Optional, Dict, Any
from datetime import datetime
import logging
import uuid

# Import enhanced components
from app.agents.portfolio_agent import (
    create_enhanced_portfolio_agent,
    EnhancedAgentState
)
from app.services.improved_input_validation import ImprovedInputValidator
from app.services.knowledge_base import get_knowledge_base
from app.config import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
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
        self.conversations = {}
        self.conversation_metadata = {}
        
        # Quality monitoring
        self.quality_metrics = {
            "total_queries": 0,
            "successful_responses": 0,
            "validation_failures": 0,
            "low_confidence_responses": 0,
            "average_confidence": 0.0
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
                "positive_feedback_signals": 0
            }
        }
    
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
    
    def _update_quality_metrics(self, confidence: float, success: bool) -> None:
        """Update service-wide quality metrics"""
        self.quality_metrics["total_queries"] += 1
        
        if success:
            self.quality_metrics["successful_responses"] += 1
        
        if confidence < 0.7:
            self.quality_metrics["low_confidence_responses"] += 1
        
        # Update rolling average confidence
        total = self.quality_metrics["total_queries"]
        old_avg = self.quality_metrics["average_confidence"]
        self.quality_metrics["average_confidence"] = (
            (old_avg * (total - 1) + confidence) / total
        )
    
    def _handle_ambiguous_query(self, message: str, validation_metadata: Dict) -> str:
        """Provide helpful response for ambiguous queries"""
        suggestions = [
            "Try asking about specific projects (e.g., 'Tell me about your Orani AI Assistant project')",
            "Ask about technical skills (e.g., 'What NLP technologies do you use?')",
            "Inquire about experience (e.g., 'How many years of ML experience do you have?')",
            "Learn about specific domains (e.g., 'What computer vision projects have you built?')"
        ]
        
        response = "I'd love to help! Here are some things you can ask me about:\n\n"
        response += "\n".join(f"• {suggestion}" for suggestion in suggestions[:3])
        
        return response
    
    def _get_fallback_response(self, error_type: str, context: Dict) -> str:
        """Generate appropriate fallback response"""
        fallbacks = {
            "validation_error": "I didn't quite understand that. Could you rephrase your question about my AI/ML work, skills, or projects?",
            "processing_error": "I encountered an issue processing your request. Please try rephrasing or ask about my experience, skills, or projects.",
            "low_confidence": "I'm not entirely certain about that. Let me share what I do know, or feel free to ask about specific aspects of my AI/ML expertise.",
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
        
        Args:
            message: User input
            conversation_id: Optional conversation ID
            enable_markdown_stripping: Strip markdown formatting
            enable_advanced_features: Enable RAG and validation
            
        Returns:
            Response dictionary with metadata
        """
        try:
            # Sanitize input
            sanitized_message = self.validator.sanitize_input(message)
            
            # Generate conversation ID
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
                self._create_conversation_metadata(conversation_id)
            
            # Validate message
            is_valid, reason, validation_metadata = self.validator.is_valid_message(
                sanitized_message
            )
            
            if not is_valid:
                self._update_conversation_metadata(
                    conversation_id, "invalid", 0.0, is_valid=False
                )
                self.quality_metrics["validation_failures"] += 1
                
                feedback = self.validator.get_validation_feedback(
                    is_valid, reason, validation_metadata
                )
                
                return {
                    "response": feedback,
                    "conversation_id": conversation_id,
                    "error": "invalid_input",
                    "reason": reason,
                    "metadata": validation_metadata,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # CRITICAL FIX: Get conversation history but DON'T pass it in initial state
            # Let the agent manage its own state accumulation
            conversation_history = self.conversations.get(conversation_id, [])
            
            # Limit history to prevent memory issues (keep last 10 messages)
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]
                self.conversations[conversation_id] = conversation_history
            
            # Create enhanced agent state
            # IMPORTANT: Pass empty list for messages - agent will manage accumulation
            initial_state = EnhancedAgentState(
                messages=[],  # FIXED: Empty list instead of conversation_history
                user_query=sanitized_message,
                response="",
                knowledge_base=self.knowledge_base,
                retrieved_context=[],
                query_type="general",
                confidence_score=0.0,
                needs_validation=True,
                tools_used=[],
                reasoning_steps=[]
            )
            
            # Invoke enhanced agent
            logger.info(f"Processing query with enhanced agent: {conversation_id}")
            result = self.agent.invoke(initial_state)
            
            # Extract response and metadata
            response_text = result.get("response", "")
            confidence = result.get("confidence_score", 0.8)
            query_type = result.get("query_type", "general")
            reasoning_steps = result.get("reasoning_steps", [])
            
            # Handle low confidence responses
            if confidence < 0.5:
                logger.warning(f"Low confidence response ({confidence:.2f}): {sanitized_message[:50]}")
                response_text = self._get_fallback_response("low_confidence", result)
                response_text += f"\n\nOriginal attempt: {result.get('response', '')}"
            
            # Strip markdown if enabled
            if enable_markdown_stripping:
                response_text = self._strip_markdown(response_text)
            
            # CRITICAL FIX: Manually update conversation history
            # Extract the new messages from result and append to our history
            result_messages = result.get("messages", [])
            if result_messages:
                # Append only the new message pair from this turn
                conversation_history.extend(result_messages[-2:])  # Last 2 messages (user + assistant)
                self.conversations[conversation_id] = conversation_history
            
            # Update metadata and metrics
            self._update_conversation_metadata(
                conversation_id, query_type, confidence, is_valid=True
            )
            self._update_quality_metrics(confidence, success=True)
            
            # Prepare response
            response_data = {
                "response": response_text,
                "conversation_id": conversation_id,
                "confidence": confidence,
                "query_type": query_type,
                "message_count": self.conversation_metadata[conversation_id]["message_count"],
                "reasoning_steps": reasoning_steps if enable_advanced_features else [],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Response generated successfully (confidence: {confidence:.2f})")
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error in get_response: {str(e)}", exc_info=True)
            
            self._update_quality_metrics(0.0, success=False)
            
            fallback = self._get_fallback_response("processing_error", {})
            
            return {
                "response": fallback,
                "conversation_id": conversation_id or str(uuid.uuid4()),
                "error": str(e),
                "confidence": 0.0,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _strip_markdown(self, text: str) -> str:
        """Remove markdown formatting"""
        import re
        
        # Remove bold
        text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        
        # Remove italic
        text = re.sub(r'\*([^\*]+)\*', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)
        
        # Remove headers
        text = re.sub(r'#{1,6}\s+', '', text)
        
        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Remove links but keep text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # Remove list markers
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        
        # Clean up extra whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        
        return text
    
    def get_quality_report(self) -> Dict[str, Any]:
        """Generate quality metrics report"""
        total = self.quality_metrics["total_queries"]
        
        if total == 0:
            return {
                "status": "no_data",
                "message": "No queries processed yet"
            }
        
        success_rate = (
            self.quality_metrics["successful_responses"] / total * 100
        )
        
        validation_failure_rate = (
            self.quality_metrics["validation_failures"] / total * 100
        )
        
        low_confidence_rate = (
            self.quality_metrics["low_confidence_responses"] / total * 100
        )
        
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
        
        # Calculate average confidence
        confidence_scores = metadata.get("confidence_scores", [])
        avg_confidence = (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores else 0.0
        )
        
        return {
            "conversation_id": conversation_id,
            "message_count": metadata["message_count"],
            "query_types_distribution": self._get_distribution(
                metadata["query_types"]
            ),
            "average_confidence": f"{avg_confidence:.2f}",
            "validation_failures": metadata["validation_failures"],
            "duration_minutes": self._calculate_duration(metadata),
            "engagement_level": self._assess_engagement(metadata),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _get_distribution(self, items: list) -> Dict[str, int]:
        """Calculate distribution of items"""
        distribution = {}
        for item in items:
            distribution[item] = distribution.get(item, 0) + 1
        return distribution
    
    def _calculate_duration(self, metadata: Dict) -> float:
        """Calculate conversation duration in minutes"""
        try:
            created = datetime.fromisoformat(metadata["created_at"])
            last_activity = datetime.fromisoformat(metadata["last_activity"])
            duration = (last_activity - created).total_seconds() / 60
            return round(duration, 2)
        except:
            return 0.0
    
    def _assess_engagement(self, metadata: Dict) -> str:
        """Assess user engagement level"""
        msg_count = metadata["message_count"]
        duration = self._calculate_duration(metadata)
        
        if msg_count == 0:
            return "no_engagement"
        elif msg_count >= 5 and duration > 2:
            return "high"
        elif msg_count >= 3:
            return "medium"
        else:
            return "low"
    
    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear conversation data"""
        try:
            if conversation_id in self.conversations:
                del self.conversations[conversation_id]
            if conversation_id in self.conversation_metadata:
                del self.conversation_metadata[conversation_id]
            
            logger.info(f"Cleared conversation: {conversation_id}")
            return True
        except Exception as e:
            logger.error(f"Error clearing conversation: {str(e)}")
            return False
    
    def cleanup_old_conversations(self, max_age_hours: int = 24) -> int:
        """Cleanup old conversations"""
        try:
            current_time = datetime.utcnow()
            to_remove = []
            
            for conv_id, metadata in self.conversation_metadata.items():
                last_activity = datetime.fromisoformat(metadata["last_activity"])
                age_hours = (current_time - last_activity).total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    to_remove.append(conv_id)
            
            for conv_id in to_remove:
                self.clear_conversation(conv_id)
            
            logger.info(f"Cleaned up {len(to_remove)} conversations")
            return len(to_remove)
        except Exception as e:
            logger.error(f"Error in cleanup: {str(e)}")
            return 0


# Singleton instance
_enhanced_service_instance = None

def get_enhanced_chatbot_service() -> EnhancedChatbotService:
    """Get singleton instance of enhanced service"""
    global _enhanced_service_instance
    if _enhanced_service_instance is None:
        _enhanced_service_instance = EnhancedChatbotService()
    return _enhanced_service_instance
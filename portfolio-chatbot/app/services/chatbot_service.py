from app.agents.portfolio_agent import create_portfolio_agent, AgentState
from app.services.knowledge_base import get_knowledge_base
from app.config import get_settings
import uuid
import re
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatbotService:
    """
    Enhanced chatbot service with conversation management,
    intent detection, input validation, and response formatting
    """
    
    def __init__(self):
        """Initialize chatbot service with configuration and agent"""
        self.config = get_settings()
        self.knowledge_base = get_knowledge_base()
        self.agent = create_portfolio_agent(self.config)
        self.conversations = {}
        self.conversation_metadata = {}
        
        logger.info("ChatbotService initialized successfully")
    
    def _strip_markdown(self, text: str) -> str:
        """
        Remove markdown formatting from text for clean display
        
        Args:
            text: Text with potential markdown formatting
            
        Returns:
            Clean text without markdown symbols
        """
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
    
    def _is_valid_message(self, message: str) -> Tuple[bool, str]:
        """
        Validate if message is meaningful and not gibberish
        
        Args:
            message: User input message
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check minimum length
        if len(message) < 2:
            return False, "too_short"
        
        # Remove punctuation and spaces for analysis
        clean_text = ''.join(char for char in message if char.isalpha())
        
        if not clean_text:
            return False, "no_letters"
        
        # Check for vowels (basic language check)
        vowels = set('aeiouAEIOU')
        has_vowels = any(char in vowels for char in message)
        
        if not has_vowels and len(clean_text) > 3:
            return False, "no_vowels"
        
        # Count vowels and consonants
        vowel_count = sum(1 for char in clean_text if char.lower() in vowels)
        total_letters = len(clean_text)
        
        if total_letters > 0:
            vowel_ratio = vowel_count / total_letters
            
            # Natural language typically has 35-45% vowels
            # Allow some flexibility for acronyms and short words
            if total_letters > 5 and (vowel_ratio < 0.15 or vowel_ratio > 0.7):
                return False, "abnormal_vowel_ratio"
        
        # Check for excessive consecutive consonants (keyboard smashing)
        max_consecutive_consonants = 0
        current_consecutive = 0
        
        for char in message.lower():
            if char.isalpha() and char not in vowels:
                current_consecutive += 1
                max_consecutive_consonants = max(max_consecutive_consonants, current_consecutive)
            else:
                current_consecutive = 0
        
        # Most natural words don't have more than 5-6 consecutive consonants
        if max_consecutive_consonants > 7:
            return False, "excessive_consonants"
        
        # Check for repeated patterns (like "aaaaaa" or "kjkjkjkj")
        words = message.split()
        gibberish_words = 0
        
        for word in words:
            # Check if word is just repeated characters
            if len(word) > 3 and len(set(word.lower())) <= 2:
                gibberish_words += 1
            
            # Check for alternating character patterns
            if len(word) > 6:
                is_pattern = True
                pattern_length = 2
                for i in range(0, len(word) - pattern_length, pattern_length):
                    if word[i:i+pattern_length].lower() != word[0:pattern_length].lower():
                        is_pattern = False
                        break
                if is_pattern:
                    gibberish_words += 1
        
        # If more than 50% of words are gibberish
        if words and gibberish_words / len(words) > 0.5:
            return False, "repetitive_patterns"
        
        # Check for common question words or greetings (indicates valid input)
        common_words = {
            'what', 'who', 'where', 'when', 'why', 'how', 'tell', 'explain',
            'describe', 'show', 'can', 'could', 'would', 'should', 'hello',
            'hi', 'hey', 'greetings', 'help', 'about', 'your', 'you', 'skills',
            'experience', 'project', 'work', 'know', 'do', 'does', 'is', 'are'
        }
        
        message_lower = message.lower()
        has_common_word = any(word in message_lower for word in common_words)
        
        # If message is very short and has no common words, might be gibberish
        if len(words) <= 2 and not has_common_word and total_letters > 8:
            return False, "no_recognizable_words"
        
        return True, "valid"
    
    def _detect_intent(self, message: str) -> str:
        """
        Detect user intent from message to customize response style
        
        Args:
            message: User's input message
            
        Returns:
            Intent category string
        """
        message_lower = message.lower()
        
        # Technical questions
        technical_keywords = [
            'how do', 'how does', 'architecture', 'implement', 'algorithm', 'model',
            'framework', 'technical', 'code', 'deploy', 'optimize', 'build',
            'pipeline', 'api', 'database', 'performance', 'tensorflow', 'pytorch',
            'neural', 'training', 'mlops', 'kubernetes', 'docker'
        ]
        if any(keyword in message_lower for keyword in technical_keywords):
            return "technical_question"
        
        # Project inquiries
        project_keywords = [
            'project', 'built', 'created', 'developed', 'work on', 'worked',
            'portfolio', 'example', 'showcase', 'case study', 'recommendation',
            'chatbot', 'detection', 'vehicle'
        ]
        if any(keyword in message_lower for keyword in project_keywords):
            return "project_inquiry"
        
        # Collaboration/business
        business_keywords = [
            'hire', 'hiring', 'available', 'collaborate', 'work together',
            'consulting', 'freelance', 'rate', 'cost', 'price', 'salary',
            'contract', 'opportunity', 'job', 'position', 'email', 'contact'
        ]
        if any(keyword in message_lower for keyword in business_keywords):
            return "collaboration"
        
        # Skills and experience
        skills_keywords = [
            'skill', 'expertise', 'experience', 'know', 'familiar', 'proficient',
            'good at', 'specializ', 'capability', 'capable', 'technologies',
            'tools', 'programming', 'language'
        ]
        if any(keyword in message_lower for keyword in skills_keywords):
            return "skills_inquiry"
        
        # Quick facts (short questions)
        if len(message.split()) <= 5 or (message.endswith('?') and len(message) < 50):
            return "quick_question"
        
        return "general"
    
    def _sanitize_input(self, message: str) -> str:
        """
        Sanitize user input to prevent injection attacks
        
        Args:
            message: Raw user input
            
        Returns:
            Sanitized message
        """
        # Remove potential script tags
        message = re.sub(r'<script[^>]*>.*?</script>', '', message, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove HTML tags
        message = re.sub(r'<[^>]+>', '', message)
        
        # Remove null bytes
        message = message.replace('\x00', '')
        
        # Limit length
        max_length = 500
        if len(message) > max_length:
            message = message[:max_length]
        
        # Remove excessive whitespace
        message = ' '.join(message.split())
        
        return message.strip()
    
    def _create_conversation_metadata(self, conversation_id: str) -> None:
        """
        Create metadata for tracking conversation
        
        Args:
            conversation_id: Unique conversation identifier
        """
        self.conversation_metadata[conversation_id] = {
            "created_at": datetime.utcnow().isoformat(),
            "message_count": 0,
            "intents": [],
            "last_activity": datetime.utcnow().isoformat(),
            "invalid_attempts": 0
        }
    
    def _update_conversation_metadata(
        self, 
        conversation_id: str, 
        intent: str,
        is_valid: bool = True
    ) -> None:
        """
        Update conversation metadata after each message
        
        Args:
            conversation_id: Unique conversation identifier
            intent: Detected intent of current message
            is_valid: Whether the message was valid
        """
        if conversation_id in self.conversation_metadata:
            metadata = self.conversation_metadata[conversation_id]
            metadata["message_count"] += 1
            metadata["intents"].append(intent)
            metadata["last_activity"] = datetime.utcnow().isoformat()
            
            if not is_valid:
                metadata["invalid_attempts"] = metadata.get("invalid_attempts", 0) + 1
    
    def _get_prompt_type(self, intent: str) -> str:
        """
        Map intent to prompt type
        
        Args:
            intent: Detected user intent
            
        Returns:
            Prompt type string
        """
        intent_to_prompt = {
            "technical_question": "technical",
            "collaboration": "sales",
            "quick_question": "concise",
            "general": "default",
            "project_inquiry": "default",
            "skills_inquiry": "default"
        }
        return intent_to_prompt.get(intent, "default")
    
    async def get_response(
        self, 
        message: str, 
        conversation_id: Optional[str] = None,
        enable_markdown_stripping: bool = True,
        detect_intent: bool = True
    ) -> Dict[str, Any]:
        """
        Get chatbot response for user message with enhanced features
        
        Args:
            message: User's input message
            conversation_id: Optional existing conversation ID
            enable_markdown_stripping: Whether to remove markdown from response
            detect_intent: Whether to detect intent and adapt response
            
        Returns:
            Dictionary containing response, conversation_id, and metadata
        """
        try:
            # Sanitize input
            sanitized_message = self._sanitize_input(message)
            
            if not sanitized_message:
                return {
                    "response": "I didn't receive a valid message. Could you please try again?",
                    "conversation_id": conversation_id or str(uuid.uuid4()),
                    "error": "empty_message"
                }
            
            # Generate or use existing conversation ID
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
                self._create_conversation_metadata(conversation_id)
            
            # Validate message quality
            is_valid, validation_reason = self._is_valid_message(sanitized_message)
            
            if not is_valid:
                logger.warning(f"Invalid message detected: {validation_reason} - {sanitized_message}")
                
                # Update metadata for invalid attempt
                self._update_conversation_metadata(conversation_id, "invalid", is_valid=False)
                
                # Provide helpful response for invalid input
                invalid_responses = {
                    "too_short": "Could you please provide a bit more detail in your question?",
                    "no_letters": "I need a text-based question. How can I help you learn about my AI/ML expertise?",
                    "no_vowels": "I'm having trouble understanding your message. Could you try asking about my experience, skills, or projects?",
                    "abnormal_vowel_ratio": "That doesn't seem like a valid question. Feel free to ask me about my AI engineering work, technical skills, or projects!",
                    "excessive_consonants": "I didn't quite understand that. Could you rephrase your question about my AI/ML experience, skills, or projects?",
                    "repetitive_patterns": "That looks like a typo or random characters. How can I help you with information about my work?",
                    "no_recognizable_words": "I couldn't process that input. Could you ask me about my skills, projects, or experience?"
                }
                
                default_response = "I didn't understand that. Could you ask me about my skills, projects, or AI/ML experience?"
                
                return {
                    "response": invalid_responses.get(validation_reason, default_response),
                    "conversation_id": conversation_id,
                    "error": "invalid_input",
                    "reason": validation_reason,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Detect intent if enabled
            intent = "general"
            if detect_intent:
                intent = self._detect_intent(sanitized_message)
                logger.info(f"Detected intent: {intent} for message: {sanitized_message[:50]}...")
            
            # Get conversation history
            conversation_history = self.conversations.get(conversation_id, [])
            
            # Create initial state for agent
            initial_state = AgentState(
                messages=conversation_history,
                user_query=sanitized_message,
                response="",
                knowledge_base=self.knowledge_base
            )
            
            # Invoke agent to get response
            logger.info(f"Invoking agent for conversation: {conversation_id}")
            result = self.agent.invoke(initial_state)
            
            # Get raw response
            raw_response = result.get("response", "I'm sorry, I couldn't generate a response.")
            
            # Strip markdown if enabled
            if enable_markdown_stripping:
                clean_response = self._strip_markdown(raw_response)
            else:
                clean_response = raw_response
            
            # Update conversation history
            self.conversations[conversation_id] = result["messages"]
            
            # Update metadata
            self._update_conversation_metadata(conversation_id, intent, is_valid=True)
            
            # Prepare response
            response_data = {
                "response": clean_response,
                "conversation_id": conversation_id,
                "intent": intent,
                "message_count": self.conversation_metadata[conversation_id]["message_count"],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Response generated successfully for conversation: {conversation_id}")
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error in get_response: {str(e)}", exc_info=True)
            
            return {
                "response": "I apologize, but I encountered an error processing your request. Please try again or rephrase your question.",
                "conversation_id": conversation_id or str(uuid.uuid4()),
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def clear_conversation(self, conversation_id: str) -> bool:
        """
        Clear conversation history and metadata
        
        Args:
            conversation_id: Unique conversation identifier
            
        Returns:
            Boolean indicating success
        """
        try:
            if conversation_id in self.conversations:
                del self.conversations[conversation_id]
            
            if conversation_id in self.conversation_metadata:
                del self.conversation_metadata[conversation_id]
            
            logger.info(f"Cleared conversation: {conversation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing conversation {conversation_id}: {str(e)}")
            return False
    
    def get_conversation_metadata(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific conversation
        
        Args:
            conversation_id: Unique conversation identifier
            
        Returns:
            Conversation metadata or None
        """
        return self.conversation_metadata.get(conversation_id)
    
    def get_active_conversations_count(self) -> int:
        """
        Get count of active conversations
        
        Returns:
            Number of active conversations
        """
        return len(self.conversations)
    
    def cleanup_old_conversations(self, max_age_hours: int = 24) -> int:
        """
        Clean up conversations older than specified hours
        
        Args:
            max_age_hours: Maximum age in hours before cleanup
            
        Returns:
            Number of conversations cleaned up
        """
        try:
            current_time = datetime.utcnow()
            conversations_to_remove = []
            
            for conv_id, metadata in self.conversation_metadata.items():
                last_activity = datetime.fromisoformat(metadata["last_activity"])
                age_hours = (current_time - last_activity).total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    conversations_to_remove.append(conv_id)
            
            # Remove old conversations
            for conv_id in conversations_to_remove:
                self.clear_conversation(conv_id)
            
            logger.info(f"Cleaned up {len(conversations_to_remove)} old conversations")
            return len(conversations_to_remove)
            
        except Exception as e:
            logger.error(f"Error in cleanup_old_conversations: {str(e)}")
            return 0
    
    def get_conversation_history(
        self, 
        conversation_id: str, 
        max_messages: int = 50
    ) -> list:
        """
        Get conversation history for a specific conversation
        
        Args:
            conversation_id: Unique conversation identifier
            max_messages: Maximum number of messages to return
            
        Returns:
            List of messages
        """
        history = self.conversations.get(conversation_id, [])
        return history[-max_messages:] if history else []
    
    def export_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Export complete conversation data for analytics or debugging
        
        Args:
            conversation_id: Unique conversation identifier
            
        Returns:
            Complete conversation data or None
        """
        if conversation_id not in self.conversations:
            return None
        
        return {
            "conversation_id": conversation_id,
            "messages": self.conversations[conversation_id],
            "metadata": self.conversation_metadata.get(conversation_id, {}),
            "exported_at": datetime.utcnow().isoformat()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get service statistics for monitoring
        
        Returns:
            Dictionary with service statistics
        """
        total_conversations = len(self.conversations)
        total_messages = sum(
            len(messages) for messages in self.conversations.values()
        )
        
        # Intent distribution
        intent_counts = {}
        total_invalid = 0
        
        for metadata in self.conversation_metadata.values():
            for intent in metadata.get("intents", []):
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
            total_invalid += metadata.get("invalid_attempts", 0)
        
        return {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "total_invalid_attempts": total_invalid,
            "average_messages_per_conversation": (
                total_messages / total_conversations if total_conversations > 0 else 0
            ),
            "intent_distribution": intent_counts,
            "timestamp": datetime.utcnow().isoformat()
        }


# Singleton instance for application-wide use
_chatbot_service_instance = None

def get_chatbot_service() -> ChatbotService:
    """
    Get or create singleton chatbot service instance
    
    Returns:
        ChatbotService instance
    """
    global _chatbot_service_instance
    if _chatbot_service_instance is None:
        _chatbot_service_instance = ChatbotService()
    return _chatbot_service_instance
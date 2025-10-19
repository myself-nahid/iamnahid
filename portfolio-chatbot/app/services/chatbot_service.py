from app.agents.portfolio_agent import create_portfolio_agent, AgentState
from app.services.knowledge_base import get_knowledge_base
from app.config import get_settings
import uuid

class ChatbotService:
    def __init__(self):
        self.config = get_settings()
        self.knowledge_base = get_knowledge_base()
        self.agent = create_portfolio_agent(self.config)
        self.conversations = {}
    
    async def get_response(self, message: str, conversation_id: str = None) -> dict:
        """Get chatbot response for user message"""
        
        # Generate or use existing conversation ID
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        # Get conversation history
        conversation_history = self.conversations.get(conversation_id, [])
        
        # Create initial state
        initial_state = AgentState(
            messages=conversation_history,
            user_query=message,
            response="",
            knowledge_base=self.knowledge_base
        )
        
        # Run agent
        result = self.agent.invoke(initial_state)
        
        # Update conversation history
        self.conversations[conversation_id] = result["messages"]
        
        return {
            "response": result["response"],
            "conversation_id": conversation_id
        }
    
    def clear_conversation(self, conversation_id: str):
        """Clear conversation history"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
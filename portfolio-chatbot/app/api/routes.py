from fastapi import APIRouter, HTTPException, Depends
from app.models import ChatRequest, ChatResponse, HealthResponse
from app.services.chatbot_service import ChatbotService

router = APIRouter()
chatbot_service = ChatbotService()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint to interact with the AI assistant"""
    try:
        result = await chatbot_service.get_response(
            message=request.message,
            conversation_id=request.conversation_id
        )
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="healthy", version="1.0.0")

@router.delete("/conversation/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """Clear conversation history"""
    chatbot_service.clear_conversation(conversation_id)
    return {"message": "Conversation cleared successfully"}
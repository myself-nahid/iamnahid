from fastapi import APIRouter, HTTPException
from app.models import ChatRequest, ChatResponse, HealthResponse
from app.services.chatbot_service import get_chatbot_service

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint to interact with the AI assistant"""
    chatbot_service = get_chatbot_service()
    
    try:
        result = await chatbot_service.get_response(
            message=request.message,
            conversation_id=request.conversation_id,
            enable_markdown_stripping=True,  # Auto-strip markdown
            detect_intent=True  # Auto-detect intent
        )
        
        return ChatResponse(
            response=result["response"],
            conversation_id=result["conversation_id"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing request: {str(e)}"
        )

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="healthy", version="1.0.0")

@router.delete("/conversation/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """Clear conversation history"""
    chatbot_service = get_chatbot_service()
    success = chatbot_service.clear_conversation(conversation_id)
    
    if success:
        return {"message": "Conversation cleared successfully"}
    else:
        raise HTTPException(
            status_code=404, 
            detail="Conversation not found"
        )

@router.get("/stats")
async def get_statistics():
    """Get chatbot service statistics"""
    chatbot_service = get_chatbot_service()
    return chatbot_service.get_statistics()

@router.get("/conversation/{conversation_id}/metadata")
async def get_conversation_metadata(conversation_id: str):
    """Get conversation metadata"""
    chatbot_service = get_chatbot_service()
    metadata = chatbot_service.get_conversation_metadata(conversation_id)
    
    if metadata:
        return metadata
    else:
        raise HTTPException(
            status_code=404, 
            detail="Conversation not found"
        )

@router.get("/conversation/{conversation_id}/export")
async def export_conversation(conversation_id: str):
    """Export conversation data"""
    chatbot_service = get_chatbot_service()
    data = chatbot_service.export_conversation(conversation_id)
    
    if data:
        return data
    else:
        raise HTTPException(
            status_code=404, 
            detail="Conversation not found"
        )
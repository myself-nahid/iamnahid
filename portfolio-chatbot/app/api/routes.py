from fastapi import APIRouter, HTTPException
from app.models import ChatRequest, ChatResponse, HealthResponse
from app.services.chatbot_service import get_enhanced_chatbot_service

router = APIRouter()

import logging
import traceback

logger = logging.getLogger(__name__)

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint with error logging"""
    try:
        logger.info(f"Received message: {request.message[:50]}...")
        
        chatbot_service = get_enhanced_chatbot_service()
        logger.info("Service retrieved")
        
        result = await chatbot_service.get_response(
            message=request.message,
            conversation_id=request.conversation_id,
            enable_markdown_stripping=True,
            enable_advanced_features=True 
        )
        
        logger.info("Response generated successfully")
        
        return ChatResponse(
            response=result["response"],
            conversation_id=result["conversation_id"]
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Error: {str(e)}"
        )

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="healthy", version="1.0.0")

@router.delete("/conversation/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """Clear conversation history"""
    chatbot_service = get_enhanced_chatbot_service()
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
    chatbot_service = get_enhanced_chatbot_service()
    return chatbot_service.get_statistics()

@router.get("/conversation/{conversation_id}/metadata")
async def get_conversation_metadata(conversation_id: str):
    """Get conversation metadata"""
    chatbot_service = get_enhanced_chatbot_service()
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
    chatbot_service = get_enhanced_chatbot_service()
    data = chatbot_service.export_conversation(conversation_id)
    
    if data:
        return data
    else:
        raise HTTPException(
            status_code=404, 
            detail="Conversation not found"
        )
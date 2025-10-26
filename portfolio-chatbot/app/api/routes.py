from fastapi import APIRouter, HTTPException
from app.models import ChatRequest, ChatResponse, HealthResponse
from app.services.chatbot_service import get_enhanced_chatbot_service
import logging
import traceback
import uuid # Import uuid for error IDs

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint with error logging"""
    try:
        logger.info(f"Received message: {request.message[:100]}... (conv_id: {request.conversation_id})")
        
        chatbot_service = get_enhanced_chatbot_service()

        if not request.message or len(request.message.strip()) < 1:
            raise HTTPException(status_code=400, detail="Message cannot be empty.")
        if len(request.message) > 1000:
            raise HTTPException(status_code=400, detail="Message too long. Max 1000 characters.")
        
        result = await chatbot_service.get_response(
            message=request.message,
            conversation_id=request.conversation_id,
            enable_markdown_stripping=True,
            enable_advanced_features=True 
        )
        
        # --- START OF FIX ---
        
        confidence_value = result.get('confidence')
        
        # Check if confidence is a number and format it for logging. Otherwise, use a placeholder.
        if isinstance(confidence_value, (float, int)):
            confidence_log_str = f"{confidence_value:.2f}"
        else:
            # This handles cases where validation fails and 'confidence' is not in the result dict.
            confidence_log_str = "N/A (Validation Failed)"
            
        logger.info(f"Response generated for conv_id: {result['conversation_id']} (confidence: {confidence_log_str})")

        # --- END OF FIX ---
        
        # This handles both successful responses and user-facing error messages from the service
        return ChatResponse(
            response=result["response"],
            conversation_id=result["conversation_id"]
        )
        
    except HTTPException as e:
        logger.warning(f"Client error in chat endpoint: {e.detail}")
        raise e
    except Exception as e:
        error_id = uuid.uuid4()
        logger.error(f"Internal server error in chat endpoint (Error ID: {error_id}): {str(e)}", exc_info=True)
        
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred. Please try again later. (Error ID: {error_id})"
        )

# ... (the rest of the routes.py file remains the same) ...
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
    return chatbot_service.get_quality_report()

@router.get("/conversation/{conversation_id}/metadata")
async def get_conversation_metadata(conversation_id: str):
    """Get conversation metadata"""
    chatbot_service = get_enhanced_chatbot_service()
    metadata = chatbot_service.get_conversation_insights(conversation_id)
    
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
    data = chatbot_service.get_conversation_history(conversation_id)
    
    if data:
        return {"conversation_id": conversation_id, "history": data}
    else:
        raise HTTPException(
            status_code=404, 
            detail="Conversation not found"
        )
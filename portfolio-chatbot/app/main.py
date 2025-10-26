import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.config import get_settings

# Configure basic logging for visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize settings
settings = get_settings()
logger.info("Settings loaded.")

# Create FastAPI app
app = FastAPI(
    title="Portfolio Chatbot API",
    description="AI-powered chatbot for portfolio website",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("CORS middleware configured.")

# Include routers
app.include_router(router, prefix="/api", tags=["chatbot"])
logger.info("API router included.")

@app.get("/")
async def root():
    logger.info("Root endpoint accessed.")
    return {
        "message": "Portfolio Chatbot API",
        "version": "1.0.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server.")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
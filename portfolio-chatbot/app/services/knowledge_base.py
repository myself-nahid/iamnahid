PORTFOLIO_KNOWLEDGE = {
    "personal_info": {
        "name": "Nahid Hasan",
        "title": "AI Engineer & ML Specialist",
        "tagline": "Building intelligent systems that transform data into actionable insights",
        "specializations": [
            "Deep Learning",
            "Natural Language Processing",
            "Computer Vision"
        ]
    },
    
    "skills": {
        "machine_learning": {
            "description": "Expert in developing and deploying ML models",
            "technologies": ["TensorFlow", "PyTorch", "scikit-learn"],
            "specialties": ["Deep Neural Networks", "Reinforcement Learning"]
        },
        "nlp": {
            "description": "Advanced experience with transformer models",
            "technologies": ["BERT", "GPT", "Transformers"],
            "specialties": ["LLM Fine-tuning", "Text Classification", "Named Entity Recognition"]
        },
        "computer_vision": {
            "description": "Proficient in CNNs and object detection",
            "technologies": ["OpenCV", "YOLO", "CNNs"],
            "specialties": ["Object Detection", "Image Segmentation", "Real-time Video Processing"]
        },
        "mlops": {
            "description": "Experienced in model deployment and scaling",
            "technologies": ["AWS", "GCP", "Azure", "Kubernetes", "Docker"],
            "specialties": ["CI/CD Pipelines", "Model Monitoring", "Cloud Infrastructure"]
        },
        "data_engineering": {
            "description": "Strong foundation in data pipelines",
            "technologies": ["Spark", "Airflow", "SQL"],
            "specialties": ["ETL Processes", "Big Data", "Data Warehousing"]
        }
    },
    
    "projects": {
        "AI-Assistant": {
            "name": "Orani-AI-Assistant",
            "description": "Orani is an AI-powered virtual receptionist that automates call handling, transcription, and messaging to streamline business communications Integrating Twilio, Whisper, and ElevenLabs, it offers secure, multilingual telephony with CRM support, enhancing efficiency and customer engagement for SMBs.",
            "impact": "Improved user engagement by 95%",
            "technologies": ["PyTorch", "Twilio", "FastAPI", "VApi", "Elevenlabs", "Whisper"]
        },
        "Multi-Agent": {
            "name": "Multi-Agent Orchestration with LangGraph",
            "description": "Developed a modular multi-agent orchestration system using LangGraph, modeling AI agent workflows as directed stateful graphs to enable flexible, scalable, and maintainable agent collaboration logic.",
            "technologies": ["Langgraph", "Python", "FastAPI", "LLM"]
        },
        "AI-Medical-Solution-app": {
            "name": "AI-Medical-Solution-app",
            "description": "AI-Medical-Solution-app is a privacy-focused AI platform that automates clinical documentation using multimodal inputs and advanced transcription with OpenAI Whisper. Leveraging Google Gemini, it generates structured, multilingual clinical notes while ensuring secure, ephemeral processing. Built with FastAPI and a modular architecture, it delivers scalable, production-grade AI for healthcare workflows.",
            "technologies": ["Whisper", "Python", "Fastapi"]
        }
    },
    
    "experience": {
        "years": "1+ years",
        "industries": ["Software Company", "IT"],
        "team_experience": "Led teams of 200-300 engineers",
        "achievements": [
            "AI Lead",
            "Open source contributor to major ML libraries",
            "Speaker at AI/ML conferences"
        ]
    },
    
    "education": {
        "degree": "B.Sc. in Computer Science",
        "specialization": "Artificial Intelligence and Machine Learning",
        "certifications": [
            "AWS Certified Machine Learning Specialty",
            "Google Cloud Professional ML Engineer",
            "Deep Learning Specialization (Coursera)"
        ]
    }
}

def get_knowledge_base():
    return PORTFOLIO_KNOWLEDGE
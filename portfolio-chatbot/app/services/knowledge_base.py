PORTFOLIO_KNOWLEDGE = {
    "personal_info": {
        "name": "Nahid Hasan",
        "title": "AI Engineer & ML Specialist",
        "tagline": "Building intelligent systems that transform data into actionable insights",
        "email": "nahidhasan.gst.cse@gmail.com",
        "github": "https://github.com/myself-nahid",
        "linkedin": "https://www.linkedin.com/in/nahid-hasan-4b2a27363/",
        "kaggle": "https://www.kaggle.com/nhmishuk",
        "specializations": [
            "Deep Learning",
            "Natural Language Processing",
            "Computer Vision"
        ]
    },
    
    "skills": {
        "machine_learning": {
            "description": "Expert in developing and deploying ML models using TensorFlow, PyTorch, and scikit-learn",
            "technologies": ["TensorFlow", "PyTorch", "scikit-learn"],
            "specialties": ["Deep Neural Networks", "Reinforcement Learning"]
        },
        "nlp": {
            "description": "Advanced experience with transformer models and LLM fine-tuning",
            "technologies": ["BERT", "GPT", "Transformers", "Whisper", "Gemini"],
            "specialties": ["LLM Fine-tuning", "Text Classification", "Named Entity Recognition", "Transcription"]
        },
        "computer_vision": {
            "description": "Proficient in CNNs, object detection, and image segmentation",
            "technologies": ["OpenCV", "YOLO", "CNNs"],
            "specialties": ["Object Detection", "Image Segmentation", "Real-time Video Processing"]
        },
        "mlops": {
            "description": "Experienced in model deployment, monitoring, and scaling on AWS, GCP, and Azure",
            "technologies": ["AWS", "GCP", "Azure", "Kubernetes", "Docker"],
            "specialties": ["CI/CD Pipelines", "Model Monitoring", "Cloud Infrastructure"]
        },
        "data_engineering": {
            "description": "Strong foundation in data pipelines and ETL processes",
            "technologies": ["Spark", "Airflow", "SQL", "Python"],
            "specialties": ["ETL Processes", "Big Data", "Data Warehousing"]
        },
        "research": {
            "description": "Published researcher with experience in experimental design and A/B testing",
            "specialties": ["Experimental Design", "A/B Testing", "Research Innovation"]
        }
    },
    
    "projects": {
        "orani_ai_assistant": {
            "name": "Orani-AI-Assistant",
            "description": "An AI-powered virtual receptionist that automates call handling, transcription, and messaging to streamline business communications. Integrates Twilio, Whisper, and ElevenLabs to offer secure, multilingual telephony with CRM support.",
            "impact": "Enhances efficiency and customer engagement for SMBs",
            "technologies": ["Twilio", "Vapi", "ElevenLabs", "Whisper", "FastAPI"],
            "features": ["Call automation", "Transcription", "Messaging", "CRM integration", "Multilingual support"]
        },
        "multi_agent_orchestration": {
            "name": "Multi-Agent Orchestration with LangGraph",
            "description": "A modular multi-agent orchestration system using LangGraph, modeling AI agent workflows as directed stateful graphs to enable flexible, scalable, and maintainable agent collaboration logic.",
            "technologies": ["LangGraph", "Python", "FastAPI", "LLM"],
            "features": ["Agent workflow modeling", "Stateful graph processing", "Scalable architecture"]
        },
        "ai_medical_solution": {
            "name": "AI-Medical-Solution-app",
            "description": "A privacy-focused AI platform that automates clinical documentation using multimodal inputs and advanced transcription. Leverages Google Gemini for generating structured, multilingual clinical notes with secure, ephemeral processing.",
            "technologies": ["Whisper", "Google Gemini", "FastAPI", "Python"],
            "features": ["Clinical documentation automation", "Multimodal input processing", "Secure processing", "Multilingual support", "Modular architecture"]
        },
        "smart_agriculture": {
            "name": "SMART AGRICULTURE FARMING",
            "description": "A web application that helps farmers make data-driven decisions using machine learning and deep learning. Includes three main modules: Crop Recommendation, Fertilizer Recommendation, and Crop Disease Prediction.",
            "technologies": ["Python", "Flask", "CNN", "Logistic Regression"],
            "features": [
                "Crop Recommendation based on environmental and soil data",
                "Fertilizer Recommendation for soil and crop needs",
                "Crop Disease Prediction using image analysis"
            ]
        }
    },
    
    "experience": {
        "years": "1+ years",
        "industries": ["AI/ML", "Software Development"],
        "projects_completed": "20+",
        "ml_models_deployed": "2+",
        "client_satisfaction": "98%",
        "achievements": [
            "Completed multiple AI research lab projects",
            "Thesis-based AI project completion",
            "End-to-end data analysis expertise",
            "Research publishing experience"
        ]
    },
    
    "education": {
        "university": {
            "degree": "Bachelor of Science in Computer Engineering",
            "institution": "Gopalgonj Science and Technology University",
            "period": "2018 - 2024",
            "specialization": "Artificial Intelligence and Machine Learning",
            "focus": "Deep learning architectures and neural networks",
            "achievements": [
                "Successfully completed a thesis-based AI project",
                "Led multiple AI research lab projects",
                "Executed end-to-end data analysis using Python"
            ]
        },
        "high_school": {
            "qualification": "High School Certificate",
            "institution": "Noakhali Govt. College",
            "period": "2015 - 2017",
            "subjects": ["Science", "Mathematics", "Biology"],
            "achievements": ["Member of Science Club"]
        },
        "secondary": {
            "qualification": "Secondary School Certificate",
            "institution": "Govt. Technical High School",
            "period": "2010 - 2014",
            "subjects": ["Science", "Mathematics", "Biology", "ICT"],
            "achievements": ["Member of Science Club"]
        }
    },
    
    "certifications": {
        "tensorflow": {
            "name": "TensorFlow Developer Certificate",
            "issuer": "Google",
            "year": 2024
        },
        "aws": {
            "name": "AWS Machine Learning Specialty",
            "issuer": "Amazon Web Services",
            "year": 2023
        },
        "deep_learning": {
            "name": "Deep Learning Specialization",
            "issuer": "DeepLearning.AI",
            "year": 2023
        },
        "azure": {
            "name": "Azure AI Engineer Associate",
            "issuer": "Microsoft",
            "year": 2024
        }
    },
    
    "statistics": {
        "projects_completed": 20,
        "ml_models_deployed": 2,
        "client_satisfaction_percent": 98
    }
}

def get_knowledge_base():
    """Returns the complete portfolio knowledge base"""
    return PORTFOLIO_KNOWLEDGE

def get_personal_info():
    """Returns personal information"""
    return PORTFOLIO_KNOWLEDGE["personal_info"]

def get_skills():
    """Returns all skills"""
    return PORTFOLIO_KNOWLEDGE["skills"]

def get_projects():
    """Returns all projects"""
    return PORTFOLIO_KNOWLEDGE["projects"]

def get_education():
    """Returns education information"""
    return PORTFOLIO_KNOWLEDGE["education"]

def get_certifications():
    """Returns certifications"""
    return PORTFOLIO_KNOWLEDGE["certifications"]

def get_experience():
    """Returns experience summary"""
    return PORTFOLIO_KNOWLEDGE["experience"]
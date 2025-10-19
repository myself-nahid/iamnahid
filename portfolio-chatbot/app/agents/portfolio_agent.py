from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    user_query: str
    response: str
    knowledge_base: dict

def create_portfolio_agent(config):
    """Create a LangGraph agent for portfolio chatbot"""
    
    llm = ChatGoogleGenerativeAI(
        model=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        google_api_key=config.GOOGLE_API_KEY,
        convert_system_message_to_human=True
    )
    
    def process_query(state: AgentState) -> AgentState:
        """Process user query and generate response"""
        from app.prompts.system_prompts import get_system_prompt
        
        system_prompt = get_system_prompt(state["knowledge_base"])
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["user_query"])
        ]
        
        response = llm.invoke(messages)
        
        return {
            **state,
            "response": response.content,
            "messages": state.get("messages", []) + [
                {"role": "user", "content": state["user_query"]},
                {"role": "assistant", "content": response.content}
            ]
        }
    
    # Create graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("process_query", process_query)
    
    # Set entry point
    workflow.set_entry_point("process_query")
    
    # Add edges
    workflow.add_edge("process_query", END)
    
    # Compile graph
    app = workflow.compile()
    
    return app
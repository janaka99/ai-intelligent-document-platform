from typing import Annotated
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.config import get_settings
from app.db.mongodb import get_mongo_db
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

@tool
async def generate_search_query(complex_message: str) -> str:
    """
    Use this tool when the user provides a complex, conversational, or vague message.
    It cleans up the message and generates an optimized keyword query to be used for document retrieval.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.openai_api_key)
    prompt = SystemMessage(
        content="You are an expert search query generator. Given the user's complex message, extract the core entities and keywords to create a concise, effective search query for a vector database. Output ONLY the generated query string without any quotes or preamble."
    )
    
    response = await llm.ainvoke([prompt, HumanMessage(content=complex_message)])
    query = response.content.strip()
    logger.info("generate_search_query_tool_used", original=complex_message, generated=query)
    return query

from langgraph.prebuilt import InjectedState
from app.agents.chat_agent.state import ChatState

@tool
async def search_documents(query: str, state: Annotated[ChatState, InjectedState]) -> str:
    """
    Search the document context for the given query.
    Returns the text content of the most relevant documents.
    ALWAYS call this tool if you need to find information from the user's documents.
    """
    logger.info("search_documents_tool_used", query=query)
    print("Search query: ", query)
    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-small", 
        api_key=settings.openai_api_key
    )
    query_vector = await embeddings_model.aembed_query(query)
    
    db = get_mongo_db()
    document_ids = state["chat_session"].document_ids
    
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index", 
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": 100,
                "limit": 5,
                "filter": {
                    "document_id": {"$in": document_ids}
                }
            }
        },
        {
            "$project": {
                "_id": 0,
                "content": 1,
                "document_id": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    cursor = db.document_embeddings.aggregate(pipeline)
    try:
        retrieved_docs = await cursor.to_list(length=5)
        logger.info("retrieved documents", docs_len=len(retrieved_docs))
    except Exception as e:
        logger.error(f"ERROR IN to_list: {e}")
        retrieved_docs = []
        
    # Store the retrieved docs in state using a reducer or we can just format it as string.
    # Actually, ToolNode will append the returned string to messages as a ToolMessage.
    # But we ALSO need to update retrieved_docs in the state if we want to show it elsewhere.
    # To update other state keys from a tool, the tool must return a Command (in langgraph>=0.2) or we just return a dictionary if we are writing a custom node.
    # Since we are using standard ToolNode, returning a dict updates the state if the ToolNode is configured properly, but usually ToolNode expects string.
    # Wait, the easiest is to just return a formatted string of the context to the LLM directly.
    
    if not retrieved_docs:
        return f"Original User Message: {state['user_message']}\n\nNo relevant documents found."
        
    context_text = "\n\n".join([f"Source Document {doc.get('document_id')}:\n{doc.get('content')}" for doc in retrieved_docs])
    
    return f"Original User Message:\n{state['user_message']}\n\nDocument Context:\n{context_text}"

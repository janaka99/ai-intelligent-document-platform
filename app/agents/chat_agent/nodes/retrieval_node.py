from app.agents.chat_agent.state import ChatState
from langchain_openai import OpenAIEmbeddings
from app.core.config import get_settings
from app.db.mongodb import get_mongo_db
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

async def retrieval_node(state: ChatState):
    logger.info("retrieval_node_started", session_id=state["session_id"])
    
    # INJECT YOUR RETRIEVER HERE
    # We use MongoDB Atlas Vector Search since the embeddings are stored there
    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-small", 
        api_key=settings.openai_api_key
    )
    query_vector = await embeddings_model.aembed_query(state["user_message"])
    
    db = get_mongo_db()
    document_ids = state["chat_session"].document_ids
    
    # Atlas Vector Search pipeline filtering by document_ids
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index", # Requires Atlas Vector Search Index creation
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
        logger.info("retrieved documents", docs=retrieved_docs)
    except Exception as e:
        logger.error(f"ERROR IN to_list: {e}")
        retrieved_docs = []
        
    logger.info("retrieval_node_completed", session_id=state["session_id"], docs_retrieved=len(retrieved_docs))
    return {"retrieved_docs": retrieved_docs}

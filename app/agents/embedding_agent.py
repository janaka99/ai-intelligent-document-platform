import asyncio
from langchain_openai import OpenAIEmbeddings
from app.core.config import get_settings
from app.db.mongodb import get_mongo_db
from datetime import datetime, timezone
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

class EmbeddingAgent:
    def __init__(self):
        # We use text-embedding-3-small as an industry standard for efficient retrieval
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", 
            api_key=settings.openai_api_key
        )
        self.db = get_mongo_db()
        self.batch_size = 100

    async def get_remaining_chunks(self, document_id: str):
        """Finds out how many embeddings exist and fetches the remaining chunks."""
        chunks_coll = self.db.document_chunks
        embeds_coll = self.db.document_embeddings
        
        total_chunks = await chunks_coll.count_documents({"document_id": document_id})
        
        if total_chunks == 0:
            return 0, []
            
        # Resumability: find max chunk_index in embeddings
        cursor = embeds_coll.find({"document_id": document_id}).sort("chunk_index", -1).limit(1)
        last_embed = await cursor.to_list(length=1)
        
        start_index = 0
        if last_embed:
            start_index = last_embed[0]["chunk_index"] + 1
            
        # Fetch remaining chunks
        remaining_cursor = chunks_coll.find({
            "document_id": document_id, 
            "chunk_index": {"$gte": start_index}
        }).sort("chunk_index", 1)
        
        remaining_chunks = await remaining_cursor.to_list(length=None)
        
        return total_chunks, remaining_chunks

    async def process_batch(self, document_id: str, batch: list):
        """Generates embeddings for a batch and saves them to MongoDB."""
        if not batch:
            return
            
        texts = [chunk["content"] for chunk in batch]
        
        logger.info(f"Embedding batch of {len(batch)} chunks for {document_id}")
        
        # Call OpenAI asynchronously
        vectors = await self.embeddings.aembed_documents(texts)
        
        # Prepare MongoDB documents
        embed_docs = []
        now = datetime.now(timezone.utc)
        for i, chunk in enumerate(batch):
            embed_docs.append({
                "document_id": document_id,
                "chunk_index": chunk["chunk_index"],
                "content": chunk["content"],
                "embedding": vectors[i],
                "created_at": now
            })
            
        # Insert batch into MongoDB
        embeds_coll = self.db.document_embeddings
        await embeds_coll.insert_many(embed_docs)

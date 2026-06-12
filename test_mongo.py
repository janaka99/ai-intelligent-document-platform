import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

async def test():
    client = AsyncIOMotorClient("mongodb+srv://janakachamith99_db_user:Ruq78CDb30DwJR9q@docintel.hus38bh.mongodb.net/?appName=docintel")
    db = client.get_default_database("docintel")
    
    # Check connection
    try:
        await db.command("ping")
        print("MongoDB connection successful")
    except Exception as e:
        print("Connection failed:", e)
        return

    # Simulate query
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index", 
                "path": "embedding",
                "queryVector": [0.1] * 1536, # Dummy vector
                "numCandidates": 100,
                "limit": 5,
                "filter": {
                    "document_id": {"$in": []}
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
    print("Cursor created:", cursor)
    
    try:
        docs = await cursor.to_list(length=5)
        print("Retrieved docs:", docs)
    except Exception as e:
        print("Error during to_list:", e)

if __name__ == "__main__":
    asyncio.run(test())

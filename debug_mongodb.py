import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# MongoDB Setup
client = MongoClient(os.getenv("MONGODB_URI"))
db = client[os.getenv("MONGODB_DB_NAME")]
memory_collection = db["candidates"]

def debug_mongodb_operations():
    """Debug function to check MongoDB operations"""
    
    print("=== MongoDB Debug Information ===")
    
    # 1. Check connection
    try:
        # Ping the database
        client.admin.command('ping')
        print("✅ MongoDB connection successful")
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        return
    
    # 2. Check database and collection
    print(f"📊 Database: {db.name}")
    print(f"📋 Collection: {memory_collection.name}")
    
    # 3. Count documents in collection
    doc_count = memory_collection.count_documents({})
    print(f"📄 Total documents in collection: {doc_count}")
    
    # 4. List all documents
    print("\n📋 All documents in collection:")
    all_docs = list(memory_collection.find({}))
    if all_docs:
        for i, doc in enumerate(all_docs):
            print(f"  Document {i+1}:")
            print(f"    Session ID: {doc.get('session_id', 'N/A')}")
            print(f"    Has summary: {'Yes' if 'summary' in doc else 'No'}")
            if 'summary' in doc:
                summary = doc['summary']
                if isinstance(summary, list):
                    print(f"    Summary length: {len(summary)} messages")
                    for j, msg in enumerate(summary[:3]):  # Show first 3 messages
                        if isinstance(msg, dict):
                            print(f"      Message {j+1}: {msg.get('type', 'unknown')} - {msg.get('content', '')[:50]}...")
                        else:
                            print(f"      Message {j+1}: {str(msg)[:50]}...")
                else:
                    print(f"    Summary: {str(summary)[:100]}...")
            print()
    else:
        print("  No documents found")
    
def check_recent_saves():
    """Check for recent saves in the last few minutes"""
    print("\n=== Recent Saves Check ===")
    
    # Look for documents with recent timestamps or recent saves
    recent_docs = list(memory_collection.find({}).sort("_id", -1).limit(5))
    
    if recent_docs:
        print("📅 Most recent documents:")
        for i, doc in enumerate(recent_docs):
            print(f"  {i+1}. Session: {doc.get('session_id', 'N/A')}")
            if 'summary' in doc:
                summary = doc['summary']
                if isinstance(summary, list):
                    print(f"     Messages: {len(summary)}")
                else:
                    print(f"     Summary: {str(summary)[:50]}...")
    else:
        print("No recent documents found")

if __name__ == "__main__":
    debug_mongodb_operations()
    check_recent_saves() 
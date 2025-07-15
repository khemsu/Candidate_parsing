import os
from langchain.memory import ConversationSummaryBufferMemory
from pymongo import MongoClient
from dotenv import load_dotenv
from utils.llm_query_helpers import (
    candidate_query_to_cypher,
    run_cypher,
)
from pydantic import SecretStr

load_dotenv()

# === MongoDB Setup ===
client = MongoClient(os.getenv("MONGODB_URI"))
db = client[os.getenv("MONGODB_DB_NAME")]
memory_collection = db["candidates"]

# === Memory Setup ===
from langchain_google_genai import ChatGoogleGenerativeAI
summary_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
    api_key=SecretStr(os.getenv("GOOGLE_API_KEY", ""))
)

memory = ConversationSummaryBufferMemory(
    llm=summary_llm,
    memory_key="history",
    return_messages=True
)

# === Save summary after each interaction ===
def save_summary_to_mongodb(session_id="default"):
    print(f"[DEBUG] save_summary_to_mongodb called for session: {session_id}")
    try:
        summary = memory.load_memory_variables({})["history"]
        print(f"[DEBUG] [save_summary_to_mongodb] Loaded summary: {summary}")
        def serialize_message(msg):
            if hasattr(msg, "to_dict"):
                return msg.to_dict()
            elif hasattr(msg, "content"):
                return {
                    "type": getattr(msg, "type", "unknown"),
                    "content": msg.content,
                }
            return str(msg)
        summary_serialized = [serialize_message(m) for m in summary]
        print(f"[DEBUG] [save_summary_to_mongodb] Serialized summary: {summary_serialized}")
        result = memory_collection.update_one(
            {"session_id": session_id},
            {"$set": {"summary": summary_serialized}},
            upsert=True
        )
        print(f"[DEBUG] [save_summary_to_mongodb] MongoDB update result: {result.modified_count} modified, {result.upserted_id} upserted")
        # print(f"[DEBUG] [save_summary_to_mongodb] ✅ Saved summary for session: {session_id}")
    except Exception as e:
        print(f"[DEBUG] [save_summary_to_mongodb] ❌ Error: {e}")
        import traceback
        traceback.print_exc()

def load_summary_from_mongodb(session_id="default"):
    doc = memory_collection.find_one({"session_id": session_id})
    if doc and "summary" in doc:
        # print(f"[DEBUG] (MongoDB) Loaded summary for session: {session_id} (length: {len(doc['summary']) if isinstance(doc['summary'], list) else 'unknown'})")
        memory.chat_memory.messages = []  # reset memory
        summary = doc["summary"]
        for msg in summary:
            if isinstance(msg, dict):
                if msg.get("type") == "human":
                    memory.chat_memory.add_user_message(msg.get("content", ""))
                elif msg.get("type") == "ai":
                    memory.chat_memory.add_ai_message(msg.get("content", ""))
        print(f"[DEBUG] Memory restored from MongoDB for session: {session_id}")
    else:
        print(f"[DEBUG] No summary found for session: {session_id}")


def add_to_memory(user_query, result, session_id="default"):
    print(f"[DEBUG] add_to_memory called with user_query: {user_query}")
    print(f"[DEBUG] add_to_memory called with result: {result}")
    memory.chat_memory.add_user_message(user_query)
    import json
    def serialize(obj):
        if hasattr(obj, 'items'):
            return {k: serialize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [serialize(x) for x in obj]
        elif hasattr(obj, '__dict__'):
            return dict(obj.items())
        return obj
    try:
        result_serialized = serialize(result)
        result_str = json.dumps(result_serialized, indent=2)
        print(f"[DEBUG] [add_to_memory] result_str: {result_str}")
        memory.chat_memory.add_ai_message(result_str)
        save_summary_to_mongodb(session_id)
    except Exception as e:
        memory.chat_memory.add_ai_message("Could not save result summary.")
        print("[DEBUG] [add_to_memory] Error:", e)
        import traceback
        traceback.print_exc()

def is_followup_query(user_query, llm):
    print(f"[DEBUG] is_followup_query called with user_query: {user_query}")
    prompt = f"""
You are a classifier. Determine if the following user query is a follow-up question that refers to previous results or context (e.g., uses words like 'their', 'those', 'them', 'the above', 'the previous', etc.), or if it is a standalone question.

Example:
- show me their cvs or resume.
- can you return me their emails.
- list me their education details. etc

User query: "{user_query}"

Return only one word: followup or standalone.
"""
    response = llm.invoke(prompt).content.strip().lower()
    print(f"[DEBUG] [is_followup_query] LLM response: '{response}' for user_query: '{user_query}'")
    if response.startswith("```"):
        response = response.strip("``` ").strip()
        print("[DEBUG] [is_followup_query] Cleaned response: {response}")
    is_followup = response == "followup"
    print(f"[DEBUG] [is_followup_query] result: {is_followup}")
    return is_followup


def _detect_requested_field(query: str):
    q = query.lower()
    if any(k in q for k in ["email", "e-mail", "mail"]):
        return "email"
    if "education" in q or "university" in q or "degree" in q:
        return "education"
    if any(k in q for k in ["work", "experience", "company", "position"]):
        return "work"
    if "skill" in q:
        return "skill"
    if "project" in q:
        return "project"
    if "activity" in q or "activities" in q:
        return "activity"
    # default -> let LLM handle
    return None

def build_followup_cypher(names: set[str], user_query: str):
    """Return a Cypher tailored to the requested field(s) for the given names.
    If the field cannot be recognised, return None to signal fallback to LLM."""
    field = _detect_requested_field(user_query)
    if not field:
        return None

    names_literal = ", ".join([f'"{n}"' if "'" in n else f"\'{n}\'" for n in names])
    prefix = (
        f"MATCH (c:Candidate)\n"
        f"WHERE c.name IN [{names_literal}]\n"
    )

    if field == "email":
        return prefix + "RETURN DISTINCT c.name, c.email"

    if field == "education":
        return (
            prefix
            + "OPTIONAL MATCH (c)-[:STUDIED_IN]->(edu:Education)\n"
            + "RETURN c.name, collect(DISTINCT {university: edu.university, degree: edu.degree}) AS education"
        )

    if field == "work":
        return (
            prefix
            + "OPTIONAL MATCH (c)-[:WORKED_IN]->(w:Work)\n"
            + "RETURN c.name, collect(DISTINCT {company: w.company, position: w.position, years: w.years}) AS workExperience"
        )
 
    if field == "skill":
        return (
            prefix
            + "OPTIONAL MATCH (c)-[:HAS_SKILL]->(s:Skill)\n"
            + "RETURN c.name, collect(DISTINCT s.name) AS skills"
        )

    if field == "project":
        return (
            prefix
            + "OPTIONAL MATCH (c)-[:HAS_PROJECT_ON]->(p:Project)\n"
            + "RETURN c.name, collect(DISTINCT p.name) AS projects"
        )
    return None


def handle_followup(user_query, driver, llm, schema, memory):
    history = memory.load_memory_variables({})
    # print(f"[DEBUG] [handle_followup] Raw history: {history}")
    last_result = ""
    names = set()
    if history and 'history' in history and history['history']:
        messages = history['history']
        print(f"[DEBUG] [handle_followup] Messages: {messages}")
       
        # Find the last AI message with candidate names (not emails)
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if hasattr(msg, 'type') and msg.type == 'ai':
                content = getattr(msg, 'content', '')
            elif isinstance(msg, dict) and msg.get("type") == "ai":
                content = msg.get("content", "")
            else:
                continue
            # Heuristic: look for 'c.name' in the content (candidate name result)
            if 'c.name' in content:
                last_result = content
                print(f"[DEBUG] [handle_followup] Found last AI message with candidate names 'from mongodb to memory': {last_result}")
                if last_result.startswith("Query result:"):
                    last_result = last_result[len("Query result:"):].strip()
                break
        print(f"[DEBUG] [handle_followup] last_result after prefix strip: {last_result}")
    else:
        print(f"[DEBUG] [handle_followup] No valid messages in history.")

    import json, re
    # print(f"[DEBUG] [handle_followup] last_result string: {last_result}")
    json_start = None
    for i, ch in enumerate(last_result):
        if ch == '{' or ch == '[':
            json_start = i
            break
    candidate_text = last_result[json_start:] if json_start is not None else last_result
    print(f"[DEBUG] [handle_followup] candidate_text string: {candidate_text}")
    parsed = None
    try:
        parsed = json.loads(candidate_text)
        print(f"[DEBUG] [handle_followup] Successfully parsed candidate_text as JSON.")
    except Exception as e_json:
        print(f"[DEBUG] [handle_followup] JSON parsing failed: {e_json}")

    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                for k, v in item.items():
                    if (k.lower() == 'name' or k.lower().endswith('.name')) and isinstance(v, str):
                        names.add(v)
    if not names:
        for match in re.finditer(r"['\"]c?\.?\.?name['\"]?\s*:\s*['\"]([^'\"]+)['\"]", candidate_text):
            names.add(match.group(1))
    print(f"[DEBUG] [handle_followup] Extracted candidate names: {names}")
    if names:
        print(f"[DEBUG] [handle_followup] Successfully parsed candidate names for followup from history and sending to build_follow_up: {names}")
        cypher = build_followup_cypher(names, user_query)
        if cypher is None:
            base_cypher = candidate_query_to_cypher(user_query, schema, llm)
            names_literal = ", ".join([f'"{n}"' if "'" in n else f"'{n}'" for n in names])
            lines = base_cypher.split("\n")
            injected = False
            for idx, line in enumerate(lines):
                if "MATCH" in line and "(c:Candidate" in line.replace(" ", "") and not injected:
                    if "WHERE" in line:
                        parts = line.split("WHERE",1)
                        lines[idx] = f"{parts[0]}WHERE c.name IN [{names_literal}] AND {parts[1].strip()}"
                    else:
                        lines.insert(idx+1, f"WHERE c.name IN [{names_literal}]")
                    injected = True
            if not injected:
                lines.append(f"WHERE c.name IN [{names_literal}]")
            cypher = "\n".join(lines)
    else:
        print(f"[DEBUG] [handle_followup] No candidate names found; treating as standalone query.")
        cypher = candidate_query_to_cypher(user_query, schema, llm)
    print(f"[DEBUG] [handle_followup] Generated Cypher for follow-up after failing to hanlde from handle_foloowup and sending to base cypher geenrater.:\n{cypher}")
    result = run_cypher(cypher, driver)
    return result
    






import json
from langchain.memory import ConversationBufferMemory
from utils.llm_query_helpers import run_cypher, candidate_query_to_cypher

# 1. Initialize memory (global)
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# 2. After Cypher query result, update memory (call this after result is shown)
def add_to_memory(user_query, result):
    """
    Add a user query and its result to the conversation memory.
    Handles Neo4j Node objects by converting them to dictionaries first.
    """
    try:
        memory.chat_memory.add_user_message(user_query)
        
        # Convert Neo4j Node objects to dictionaries if needed
        def convert_node(node):
            if hasattr(node, 'items'):  # Already a dict
                return {k: convert_node(v) for k, v in node.items()}
            elif isinstance(node, (list, tuple)):
                return [convert_node(item) for item in node]
            elif hasattr(node, '__dict__'):  # Handle Neo4j Node objects
                return dict(node.items())
            return node
            
        # Convert the result to a serializable format JOSN formatted string
        serializable_result = convert_node(result)
        
        # Convert to JSON with proper handling of non-serializable types
        result_str = json.dumps(
            serializable_result,
            indent=2,
            ensure_ascii=False,
            default=lambda o: str(o) if not isinstance(o, (int, float, str, bool, type(None))) else o
        )
        
        # Limit the size of the result to avoid memory issues
        max_result_length = 2000
        if len(result_str) > max_result_length:
            result_str = result_str[:max_result_length] + "... [truncated]"
            
        print("\n[DEBUG] Saving to memory (JSON string):\n", result_str)
        memory.chat_memory.add_ai_message(f"Query result:\n{result_str}")
        
    except Exception as e:
        import traceback
        print("\nError adding to memory: {e}\n{traceback.format_exc()}")
        # Still add the user query even if result serialization fails
        memory.chat_memory.add_ai_message("I found some results, but couldn't save them to the conversation history.")


def is_followup_query(user_query, llm):
    """
    Returns True if the user_query is a follow-up (refers to previous results), else False.
    """
    prompt = f"""
You are a classifier. Determine if the following user query is a follow-up question that refers to previous results or context (e.g., uses words like 'their', 'those', 'them', 'the above', 'the previous', etc.), or if it is a standalone question.

User query: "{user_query}"

Return only one word: followup or standalone.
"""
    response = llm.invoke(prompt).content.strip().lower()

    if response.startswith("```"):
        response = response.strip("``` ").strip()
        print("\n response from whether it is follow up query or not.")
    return response == "followup"


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

    if field == "activity":
        return (
            prefix
            + "OPTIONAL MATCH (c)-[:HAS_ACTIVITY]->(a:Activity)\n"
            + "RETURN c.name, collect(DISTINCT a.name) AS activities"
        )

    return None


def handle_followup(user_query, driver, llm, schema, memory):
    """
    Handles a follow-up query by extracting context from memory and generating a context-aware Cypher query.
    """
    # Get the last AI message (should be the last result)
    history = memory.load_memory_variables({})
    last_result = ""
    if history and 'history' in history and history['history']:
        messages = history['history']
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if hasattr(msg, 'type') and msg.type == 'ai':
                last_result = msg.content
                break

    # Extract candidate names from the last AI message
    import json, ast, re
    names = set()

    # Remove any leading label like "Query result:" and keep JSON part
    json_start = None
    for i, ch in enumerate(last_result):
        if ch == '{' or ch == '[':
            json_start = i
            break
    candidate_text = last_result[json_start:] if json_start is not None else last_result

    parsed = None
    try:
        parsed = json.loads(candidate_text)
    except Exception:
        try:
            parsed = ast.literal_eval(candidate_text)
        except Exception:
            parsed = None

    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                for k, v in item.items():
                    if ('name' in k.lower() or k.lower().endswith('.name')) and isinstance(v, str):
                        names.add(v)

    if not names:
        # Fallback regex to catch patterns like "c.name': 'Arju Thapa'"
        for match in re.finditer(r"['\"]c?\.?.?name['\"]?\s*:\s*['\"]([^'\"]+)['\"]", candidate_text):
            names.add(match.group(1))
    # Decide whether this is a contextual follow-up or a standalone request
    if names:
        print(f"[DEBUG] Names extracted for follow-up: {names}")
        cypher = build_followup_cypher(names, user_query)
        if cypher is None:
            # Field not recognised – fall back to LLM but inject name filter
            base_cypher = candidate_query_to_cypher(user_query, schema, llm)
            names_literal = ", ".join([f'"{n}"' if "'" in n else f"\'{n}\'" for n in names])
            # Try to inject WHERE after first Candidate MATCH
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
        # Stand-alone question – delegate fully to LLM
        cypher = candidate_query_to_cypher(user_query, schema, llm)

    print(f"\n[DEBUG] Generated Cypher for follow-up:\n{cypher}")
    result = run_cypher(cypher, driver)
    print(f"\n[DEBUG] Follow-up query result:\n{result}")
    return result
    






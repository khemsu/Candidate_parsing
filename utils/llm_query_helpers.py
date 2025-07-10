import streamlit as st

def detect_query_type(query, llm):
    prompt = f"""
Your task is to classify the user's input into one of the following categories:

- conversation → for greetings, small talk, or general questions about you, the system, or data storage (e.g., "hello", "how are you", "who are you", "where do you store data?")
- candidate → for technical queries about candidates, their skills, education, work experience, projects, asked to return candidates cv or resume , or any question that would require generating a Cypher query to search the candidate database (e.g., "show me candidates with Python skills", "who studied at Harvard?", "find candidates with 3+ years experience")
User input: "{query}"

Return only one word: **conversation** or **candidate**.

Do not explain. Do not include punctuation or extra words. Return just the category name.
"""
    response = llm.invoke(prompt).content.strip().lower()

    # Remove possible formatting from LLM output
    if response.startswith("```"):
        response = response.strip("```").strip()

    return response

def candidate_query_to_cypher(user_query, schema, llm):
    prompt = f"""
You are an expert in Cypher and Neo4j. You are given a knowledge graph schema and must only use nodes, relationships, and properties that exist in the schema as follows:
{schema}

Your task is to generate a Cypher query to answer the user's question. Follow ALL instructions strictly:
---

Instructions:
1. Use **case-insensitive** and **partial** matching on string properties:
   - Use `toLower(property) CONTAINS toLower("keyword")`
   - Applies to all string filters.

2. Output **only the Cypher query**:
   - No explanations, comments, or markdown
   - Valid, concise, complete Cypher

3. Resume Queries:
   - If user query contains keywords like "resume", "cv", "list candidate", "give me candidate", "who are", "their resumes", "show resumes of", etc:
     - Return only distinct candidate names with filters:
       RETURN DISTINCT c.name
     - Match related nodes as needed for filtering (skills, education, experience)

4. **WITH Clause**
   - Always pass forward all needed variables.
   - Use aliases consistently after aliasing.

5. **Aggregation**
   - Use `sum(w.years)` for total experience when asked.
   - Use `DISTINCT` inside `collect()` to avoid duplicates.

6. **For avoiding reverse duplicates**
   - Use `c1 <> c2` to avoid reverse duplicates.

7. **MATCH Usage**
   - Use `MATCH` for required conditions.
   - Use `OPTIONAL MATCH` only for fetching additional details.

8. **DISTINCT usage**
    - Use `DISTINCT` to remove duplicates from the results.

9. **Filter usage**
    - Use `WHERE` to filter the results based on user query.
    - Use `ORDER BY` to sort the results based on user query.

10. **Limit usage**
    - Use `LIMIT` to limit the number of results returned based on user query.

11. ** Partial Entity Matching**
    - Break multi-word inputs (e.g. "biplav ghale") into parts (["biplav", "ghale"]) and match each part separately using AND or OR conditions for better fuzzy matching.
    - Apply the logic to all relevant nodes, not just Candidate, and to any property being filtered.


12. **return**
    - Use `RETURN` to return the results based on user query.

some examples:
if asked about "any 5 candidate who 2 years of experience in python"
MATCH (c:Candidate)-[:WORKED_IN]->(w:Work)
WHERE w.years = 2 AND toLower(w.position) CONTAINS toLower("python")
RETURN DISTINCT c.name LIMIT 5

if asked about "any candidates who have same skills" , follow same logic for other too.
MATCH (c1:Candidate)-[:HAS_SKILL]->(s:Skill)<-[:HAS_SKILL]-(c2:Candidate)
WHERE toLower(c1.name) < toLower(c2.name)
WITH c1, c2, COLLECT(DISTINCT toLower(s.name)) AS sharedSkills
RETURN c1.name AS candidate1, c2.name AS candidate2, sharedSkills


if asked about cv of candidate who has work for 2 or more years in java
MATCH (c:Candidate)
WHERE (EXISTS {{
    MATCH (c)-[:WORKED_IN]->(w:Work)
    WHERE w.years >= 2 AND toLower(w.position) CONTAINS toLower("jav")
}}) 
OPTIONAL MATCH (c)-[:HAS_SKILL]->(s:Skill)
OPTIONAL MATCH (c)-[:STUDIED_IN]->(edu:Education)
OPTIONAL MATCH (c)-[:WORKED_IN]->(work:Work)
OPTIONAL MATCH (c)-[:HAS_PROJECT_ON]->(proj:Project)
RETURN
    c.name,
    c.email,
    COLLECT(DISTINCT s.name) AS skills,
    COLLECT(DISTINCT {{university: edu.university, degree: edu.degree}}) AS education,
    COLLECT(DISTINCT {{company: work.company, position: work.position, years: work.years}}) AS workExperience,
    COLLECT(DISTINCT proj.name) AS projects

If asked about specific candidate/ or their resume or cv, return all the details of that candidate.


🧠User Query:
{user_query}
"""
    cypher_code = llm.invoke(prompt).content.strip()
    if cypher_code.startswith("```"):
        cypher_code = cypher_code.strip("`")
        if cypher_code.lower().startswith("cypher"):
            cypher_code = cypher_code[6:].strip()
    return cypher_code

def run_cypher(cypher_query, driver):
    # Debug: Print the Cypher query being executed
    # print("[DEBUG] Executing Cypher Query:\n", cypher_query)
    try:
        with driver.session() as session:
            result = session.run(cypher_query)
            data = []
            for record in result:
                # Dynamically build the candidate dict based on available keys
                candidate = {}
                for key in record.keys():
                    candidate[key] = record.get(key)
                data.append(candidate)
            return data
    except Exception as e:
        st.error(f" Error running Cypher query: {e}")
        return []

def display_results_with_llm(result, llm):
    # Return a clear message if no results found
    if not result or (isinstance(result, list) and len(result) == 0):
        return "No matching candidates found in the database."
    
    """
    Format query results in a user-friendly way.
    
    Args:
        result: The query result to format (list of dicts or single dict)
        llm: The language model for generating responses
        
    Returns:
        str: Formatted response string
    """
    if not result:
        return "No results found matching your query."
    
    # Let the LLM do all formatting for non-empty results
    try:
        import json
        result_str = json.dumps(result, indent=2, ensure_ascii=False)
        prompt = f"""
You are a helpful assistant. Format the following database query result for a recruiter.
Present each candidate clearly in conversational natural language, using bullet points or short paragraphs.
If multiple candidates are present, number them. For each candidate, include any available fields such as
email, skills, education, work experience, projects, or activities — only if present in the data.
Use concise sentences and omit any null or empty values.

Data:
{result_str}
"""
        return llm.invoke(prompt).content.strip()
    except Exception as e:
        import traceback, json
        print(f"Error formatting results: {e}\n{traceback.format_exc()}\n")
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        # If LLM fails, return a basic formatted response
        import traceback
        print(f"Error formatting results: {e}\n{traceback.format_exc()}\n")
        return "Here are the results from your query:\n\n" + str(result)

from neo4j import GraphDatabase
import os

from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USER = os.environ["NEO4J_USER"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def candidate_exists(tx, name, email):
    query = "MATCH (c:Candidate {name: $name, email: $email}) RETURN c"
    return tx.run(query, name=name, email=email).single() is not None

def store_candidate(tx, data):
    # Set defaults for missing keys (single values or lists)
    data.setdefault("age", None)
    data.setdefault("skills", [])
    data.setdefault("education", [])
    data.setdefault("work_experience", [])
    data.setdefault("projects", [])
    data.setdefault("activities", [])
    
    # Clean/filter lists of dicts to keep only valid entries with required keys
    data["skills"] = [
        s for s in data["skills"]
        if s and isinstance(s, dict) and s.get("name")
    ]
    data["education"] = [
        edu for edu in data["education"]
        if edu and isinstance(edu, dict) and edu.get("degree") and edu.get("university")
    ]
    data["work_experience"] = [
        exp for exp in data["work_experience"]
        if exp and isinstance(exp, dict) and exp.get("company") and exp.get("position") and exp.get("years")
    ]
    data["projects"] = [
        p for p in data["projects"]
        if p and isinstance(p, dict) and p.get("name")
    ]
    data["activities"] = [
        a for a in data["activities"]
        if a and isinstance(a, dict) and a.get("name")
    ]

    # Cypher query with proper UNWIND instead of FOREACH for MERGE with properties
    query = """
    MERGE (c:Candidate {name: $name, email: $email})
    SET c.age = $age

    WITH c
    UNWIND $skills AS skill
        MERGE (s:Skill {name: skill.name})
        MERGE (c)-[:HAS_SKILL]->(s)

    WITH c
    UNWIND $education AS edu
        MERGE (e:Education {degree: edu.degree, university: edu.university})
        MERGE (c)-[:STUDIED_IN]->(e)

    WITH c
    UNWIND $work_experience AS exp
        MERGE (w:Work {company: exp.company, position: exp.position, years: exp.years})
        MERGE (c)-[:WORKED_IN]->(w)

    WITH c
    UNWIND $projects AS proj
        MERGE (p:Project {name: proj.name})
        MERGE (c)-[:HAS_PROJECT_ON]->(p)

    WITH c
    UNWIND $activities AS act
        MERGE (a:Activity {name: act.name})
        MERGE (c)-[:HAS_ACTIVITY]->(a)
    """

    tx.run(query,
           name=data.get("name"),
           email=data.get("email"),
           age=data.get("age"),
           skills=data["skills"],
           education=data["education"],
           work_experience=data["work_experience"],
           projects=data["projects"],
           activities=data["activities"])



def save_to_neo4j(data):
    with driver.session() as session:

        if data is None or "name" not in data or "email" not in data:
            raise ValueError("Invalid candidate data passed to save_to_neo4j")
        
        # Check if candidate already exists
        exists = session.read_transaction(candidate_exists, data["name"], data["email"])
        if not exists:
            session.write_transaction(store_candidate, data)
            return "Stored successfully"
        return "Candidate already exists"



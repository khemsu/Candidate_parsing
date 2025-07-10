def get_knowledge_graph_schema():
    return {
        "Candidate": {
            "name": "string",
            "email": "string",
            "age": "integer",
            "relationships": {
                "HAS_SKILL": "Skill",
                "STUDIED_IN": "Education",
                "WORKED_IN": "Work",
                "HAS_PROJECT_ON": "Project",
                "HAS_ACTIVITY": "Activity"
            }
        }
    }

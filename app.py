import streamlit as st
import os
from pydantic import SecretStr
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from neo4j import GraphDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.neo4j_ops import save_to_neo4j
from pymongo import MongoClient
import uuid


#database connection
client = MongoClient(os.getenv("MONGODB_URI"))
db = client[os.getenv("MONGODB_DB_NAME")]
collection = db["candidates"]

# Local imports
from utils.extract_cv_data import extract_text_from_pdf, extract_candidate_data

from utils.llm_query_helpers import (
    detect_query_type,
    candidate_query_to_cypher,
    run_cypher,
    display_results_with_llm
)

from memory_cypher_chain import(
    memory,
    add_to_memory,
    is_followup_query,
    handle_followup,
    load_summary_from_mongodb,
    save_summary_to_mongodb
)

# Load environment variables
load_dotenv()

def initialize_llm():
    # """Initialize and return the LLM with API key check."""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        api_key=SecretStr(google_api_key)
    )

from memory_cypher_chain import memory, add_to_memory, is_followup_query, handle_followup

def get_knowledge_graph_schema():
    """Return the schema of the Neo4j knowledge graph."""
    return """
(:Candidate {name, email})
(:Skill {name})
(:Education {university, degree})
(:Work {company, position, years})
(:Project {name})

(:Candidate)-[:HAS_SKILL]->(:Skill)
(:Candidate)-[:STUDIED_IN]->(:Education)
(:Candidate)-[:WORKED_IN]->(:Work)
(:Candidate)-[:HAS_PROJECT_ON]->(:Project)
"""

def handle_basic_conversation(query, conversation_chain):
    # Load the prompt template (only once per process in production)
    with open("cvparser_prompt_template.txt", "r") as f:
        system_prompt = f.read()
    # If the conversation_chain supports a system prompt, set it here
    if hasattr(conversation_chain, 'system_prompt'):
        conversation_chain.system_prompt = system_prompt
        return conversation_chain.predict(input=query)
    else:
        # Use the system prompt as context, but do NOT prepend it to the user query for memory
        # Instead, call the LLM directly with the system prompt and user query as context
        # (Assuming conversation_chain.predict() just sends the input to the LLM)
        # You may need to adjust this depending on your LLM wrapper
        response = conversation_chain.llm.invoke(system_prompt + "\nUSER: " + query).content
        print(response)
        # Optionally, manually add to memory here if needed
        return response

#flow 
def main():
    """Main Streamlit application."""
    # Initialize Streamlit
    st.set_page_config("CV Parser and Knowledge Graph")
    st.title("CV Parser & Neo4j Knowledge Graph")

    # Initialize LLM and Neo4j
    llm = initialize_llm()
    
    # Neo4j configuration
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    if not all([neo4j_uri, neo4j_user, neo4j_password]):
        raise ValueError("NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD environment variables must be set.")
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
    # Initialize conversation chains
    custom_prompt = PromptTemplate(
        input_variables=["history", "input"],
        template="""The following is a friendly conversation between a human and an CV parser. 
        The CV parser is helpful and provides lots of specific details about candidates from its knowledge graph. 
        If the CV parser does not know the answer to a question, it truthfully says it does not know.

        Current conversation:
        {history}
        Human: {input}
        CV parser:"""
    )
    
    # Initialize conversation chain for conversation between user and cv parser
    conversation_chain = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=custom_prompt,
        verbose=False
    )

    # File uploader
    uploaded_files = st.file_uploader("Upload CVs (PDF only)", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        with open("utils/extraction_prompt.txt") as f:
            prompt_template = f.read()
            for uploaded_file in uploaded_files:
                with st.expander(f"Processing: {uploaded_file.name}"):
                    raw_text = extract_text_from_pdf(uploaded_file)
                    candidate_data = extract_candidate_data(raw_text, prompt_template)
                    st.json(candidate_data)

                    result = save_to_neo4j(candidate_data)
                    st.success(result)
    

    # --- Streamlit chat interface ---
    st.divider()
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []  # list of (role, message)

    # Display chat history with avatars and chat bubbles
    for role, msg in st.session_state["chat_history"]:
        with st.chat_message(role):
            st.markdown(msg)

    # --- Chat input handling with st.chat_input ---
    def process_chat_input():
        user_query = st.session_state["chat_input"]
        if not user_query:
            return

        # Ensure session_id is set
        if "session_id" not in st.session_state:
            st.session_state["session_id"] = str(uuid.uuid4())
        session_id = st.session_state["session_id"]

        # Always load the latest summary for this session
        load_summary_from_mongodb(session_id)

        st.session_state["chat_history"].append(("user", user_query))
        schema = get_knowledge_graph_schema()
        with st.spinner("Analyzing query..."):
            query_type = detect_query_type(user_query, llm)


        # Check for follow-up intent for candidate queries
        if query_type == "candidate" and is_followup_query(user_query, llm):
            with st.spinner("Answering follow-up..."):
                result = handle_followup(user_query, driver, llm, schema, memory)
            
            if result:
                display_text = display_results_with_llm(result, llm)
                st.session_state["chat_history"].append(("ai", display_text))
                add_to_memory(user_query, result, session_id=session_id)
                print("\n[DEBUG] Follow-up query result from handle folowup:\n", result)
            else:
                response = "I couldn't find any information for your follow-up query."
                st.session_state["chat_history"].append(("ai", response))
        elif query_type in ("greet", "conversation"):
            with st.spinner("Replying..."):
                response = handle_basic_conversation(user_query, conversation_chain)
            st.session_state["chat_history"].append(("ai", response))

        # Handle vulgar language
        elif query_type == "vulgar":
            response = "Please use appropriate language."
            st.session_state["chat_history"].append(("ai", response))

        # Handle candidate queries
        elif query_type == "candidate":
            with st.spinner("Generating Cypher query..."):
                cypher_query = candidate_query_to_cypher(user_query, schema, llm)
                print("\n[DEBUG] Generated Cypher query:\n", cypher_query)
            with st.spinner("Querying Neo4j..."):
                result = run_cypher(cypher_query, driver)
                print("\n[DEBUG] Normal query result:\n", result)

            # Show debug information if enabled
            if 'show_debug' in st.session_state and st.session_state.show_debug:
                st.markdown("###Generated Cypher Query")
                st.code(cypher_query, language="cypher")
                st.markdown("###Raw Results")
                st.json(result)

            if result:
                display_text = display_results_with_llm(result, llm)
                st.session_state["chat_history"].append(("ai", display_text))
                add_to_memory(user_query, result, session_id=session_id)
            else:
                response = "No matching candidates found."
                st.session_state["chat_history"].append(("ai", response))

        # Clear the input box after processing
        st.session_state["chat_input"] = ""
        st.rerun()


    # User input box at the bottom, using st.chat_input
    if prompt := st.chat_input("Type your question and press Enter"):
        st.session_state["chat_input"] = prompt
        process_chat_input()

    # Sidebar controls
    st.sidebar.title("Controls")
    
    # Debug toggle
    show_debug = st.sidebar.checkbox("Show debug information", value=False)
    st.session_state.show_debug = show_debug

    # Clear memory button
    if st.sidebar.button("Clear Memory"):
        memory.clear()
        st.success("Conversation memory cleared.")



if __name__ == "__main__":
    main()
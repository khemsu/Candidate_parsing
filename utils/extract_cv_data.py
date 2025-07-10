import PyPDF2
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import os


from dotenv import load_dotenv

load_dotenv()


def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return "\n".join(page.extract_text() for page in reader.pages)


def extract_candidate_data(raw_text: str, prompt_template: str):
    # Initialize Mistral AI
    # llm = ChatMistralAI(
    #     model="mistral-medium",
    #     temperature=0.1,
    #     mistral_api_key=os.getenv("MISTRAL_API_KEY")
    # )

    llm=ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY")
    )

    # Format the prompt with the CV's raw text
    prompt = prompt_template.format(text=raw_text)

    # Invoke Mistral model
    response = llm.invoke(prompt).content.strip()
    # Optional cleanup: remove ```json or ``` wrappers
    if response.startswith("```"):
        response = response.strip("```json").strip("```").strip()

    # Try to parse JSON response
    try:
        candidate_data = json.loads(response)
    except json.JSONDecodeError:
        raise ValueError("❌ LLM response could not be parsed as valid JSON:\n" + response)

    return candidate_data


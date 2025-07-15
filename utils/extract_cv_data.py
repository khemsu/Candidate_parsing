import PyPDF2
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import os


from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()


def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return "\n".join(page.extract_text() for page in reader.pages)


def extract_candidate_data(raw_text: str, prompt_template: str):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=SecretStr(os.getenv("GOOGLE_API_KEY", ""))
    )

    # Format the prompt with the CV's raw text
    prompt = prompt_template.format(text=raw_text)

    response = llm.invoke(prompt).content
    if isinstance(response, list):
        response = response[0]
    if isinstance(response, str):
        response = response.strip()
    else:
        response = str(response).strip()

    # Optional cleanup: remove ```json or ``` wrappers
    if response.startswith("```"):
        response = response.strip("```json").strip("```").strip()

    # Try to parse JSON response
    try:
        candidate_data = json.loads(response)
    except json.JSONDecodeError:
        raise ValueError("❌ LLM response could not be parsed as valid JSON:\n" + response)

    return candidate_data


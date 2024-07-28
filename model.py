import vertexai
from langchain_google_vertexai import VertexAI


def initialize_vertexai(project_id, location):
    vertexai.init(project=project_id, location=location)


def create_gemini_llm(project_id, location, model_name, temperature):
    initialize_vertexai(project_id, location)
    return VertexAI(model_name=model_name, temperature=temperature)


def gemini_llm(project_id, location):
    return create_gemini_llm(project_id, location, "gemini-1.0-pro-001", 0.2)


def gemini15_llm(project_id, location):
    return create_gemini_llm(project_id, location, "gemini-1.5-pro-001", 0.2)

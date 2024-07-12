import pandas as pd
from langchain.prompts import PromptTemplate


def knowledge_full_context_template():
    template = """Context information is below, but feel free to use all available knowledge to answer the question.
    ---------------------
    {context}
    ---------------------
    Given the context information and any additional knowledge, provide an answer (a, b, c or d) to the question.
    {question}
    Walk me through in manageable parts step by step, summarizing and analyzing as we go.
    Answer:
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "context",
            "question"
        ]
    )
    return prompt


def knowledge_brief_answer_template():
    template = """Context information is below, but the response should be brief and directly answer the question without additional unnecessary information.
    ---------------------
    {context}
    ---------------------
    {question}
    Answer: 
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "context",
            "question"
        ]
    )
    return prompt


def score_template():
    template = """
    Given the possible solution: {possible_solution} and the code provided by the student: {answer}.
    Please assess their score using the following criteria: 
    ---------------------
    {criteria}
    ---------------------
    Based on these criteria, provide the score obtained.

    """

    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "possible_solution",
            "answer",
            "criteria"
        ]
    )
    return prompt

def answer_with_history_template():
    template = """Role and Goal: Serve as a university professor specializing in Java programming.
              Coding the following question.
    Context:
    {context}
    
    History:
    {history}
    
    Question:
    {question}
    
    Answer:
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "history",
            "context",
            "question"
        ]
    )
    return prompt


def best_compare_template():
    template = """
    You are an expert in analyzing and comparing the content of two texts for their similarity in meaning, context, and factual accuracy. Additionally, you are adept at comparing code snippets for their functional and structural similarity. Your task is to provide a detailed comparison that focuses on the underlying ideas, accuracy of the information presented, and functionality if the text is code. Based on your analysis, assign a similarity score from 0 to 1, where 0 means no similarity at all and 1 means completely identical in meaning, factual content, or functionality.

    Consider the following phrases:

    context: "{context}"
    Phrase 1: "{phrase1}"
    Phrase 2: "{phrase2}"

    Given the these phrases, please only return the similarity score.

    """

    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "context",
            "phrase1",
            "phrase2"
        ]
    )
    return prompt
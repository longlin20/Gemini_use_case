import os
import time

import google.auth
import pandas as pd
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers.bm25 import BM25Retriever

from embedding_documents import get_vertexai_embeddings
from model import gemini_llm, gemini15_llm
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from sentence_transformers import CrossEncoder


def get_retriever(db, k: int):
    """
    Returns a retriever for a given database.

    Args:
        db (Database): The database.
        k (int): Number of documents to retrieve.

    Returns:
        Retriever: The retriever.
    """
    return db.as_retriever(search_kwargs={"k": k})

def get_chroma_db(db_name: str) -> Chroma:
    """
    Returns a Chroma database.

    Args:
        db_name (str): Name of the database.

    Returns:
        Chroma: The Chroma database.
    """
    persist_directory = "./chroma_db/" + db_name
    embedding = get_vertexai_embeddings()
    return Chroma(persist_directory=persist_directory, embedding_function=embedding)

def select_model(model):
    credentials, project_id = google.auth.default()
    LOCATION = "us-central1"

    if model == "gemini1.0":
        return gemini_llm(project_id, LOCATION)
    elif model == "gemini1.5":
        return gemini15_llm(project_id, LOCATION)
    else:
        raise ValueError("Invalid model. Please choose 'gemini1.0' or 'gemini1.5'.")


# Function to convert DataFrame content to documents
def convert_to_documents(data_frame: pd.DataFrame, content_column: str):
    """
    Convert a pandas DataFrame to a list of Document objects.

    Args:
        data_frame (pd.DataFrame): The DataFrame to convert.
        content_column (str): The name of the column containing the content.

    Returns:
        A list of Document objects.
    """
    return [
        Document(page_content=row[content_column])
        for index, row in data_frame.iterrows()
    ]


def get_bm25_retriever(data_file_name: str, num_chunks: int):
    """
    Get a BM25Retriever object from a CSV file.

    Args:
        data_file_name (str): The name of the CSV file.
        num_chunks (int): The number of chunks to retrieve.

    Returns:
        A BM25Retriever object.
    """
    data_frame = pd.read_csv(data_file_name)
    documents = convert_to_documents(data_frame, 'content_chunks')

    retriever = BM25Retriever.from_documents(documents)
    retriever.k = num_chunks
    return retriever


def get_dense_retriever(database_name: str, num_chunks: int):
    """
    Get a Retriever object from a database.

    Args:
        database_name (str): The name of the database.
        num_chunks (int): The number of chunks to retrieve.

    Returns:
        A Retriever object.
    """
    retriever = get_retriever(get_chroma_db(database_name), num_chunks)
    return retriever


def select_retriever(retriever_type: str, data_file_name: str, num_chunks: int, database_name: str):
    """
    Select a retriever based on the retriever type.

    Args:
        retriever_type (str): The type of retriever to select.
        data_file_name (str): The name of the data file.
        num_chunks (int): The number of chunks to retrieve.
        database_name (str): The name of the database.

    Returns:
        A retriever object.
    """
    if retriever_type == "bm25":
        return get_bm25_retriever(data_file_name, num_chunks)
    elif retriever_type == "dense":
        return get_dense_retriever(database_name, num_chunks)
    elif retriever_type == "ensemble":
        bm25_retriever = get_bm25_retriever(data_file_name, num_chunks)
        dense_retriever = get_dense_retriever(database_name, num_chunks)
        return EnsembleRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            weights=[0.5, 0.5]
        )
    else:
        raise ValueError(
            "Invalid retriever type. Please choose 'bm25', 'dense', 'parents', 'ensemble' or 'ensemble_code'."
        )

def generate_test_answer(qa, llm, question, col_name, df, i, rag):
    context_name = "contexts"
    try:
        print(f"Attempting to get llm_answer for question {i}")
        if rag:
            result = qa.invoke(question)
            llm_answer = result["result"]
            answer_context = result['source_documents']
            context = "\n".join(doc.page_content for doc in answer_context)
            df.loc[i, context_name] = context
        else:
            llm_answer = llm.invoke(question)
        print(f"Successfully obtained llm_answer for question {i}: {llm_answer}")
    except Exception as e:
        print(f"Error obtaining llm_answer for question {i}: {e}")
        llm_answer = "Error processing the question"
        df.loc[i, context_name] = "Error processing the question"

    df.loc[i, col_name] = llm_answer


def generate_test_answer_with_rag(model, retriever_type="", data_name="",
                                  number_of_chunks=0, database_name="", prompt_type=None,
                                  input_excel_file="",
                                  col_name="", result_excel_file="", rag=True):
    llm = select_model(model)

    if rag:
        retriever = select_retriever(retriever_type, data_name, number_of_chunks,
                                     database_name)
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": prompt_type,
            })
    else:
        qa = None

    df = pd.read_excel(input_excel_file)

    question_count = 0

    for i, row in df.iterrows():
        question = row['question']

        generate_test_answer(qa, llm, question,
                             col_name, df, i, rag)

        question_count += 1
        if model == "gemini1.5" and question_count % 60 == 0:
            time.sleep(15)
    df.to_excel(result_excel_file, index=False)


def generate_test_answers(files_path, model, retriever_type, data_name,
                          number_of_chunks, database_name, prompt_type, result_folder):
    all_items = os.listdir(files_path)
    files = [item for item in all_items if os.path.isfile(os.path.join(files_path, item))]

    for f in files:
        input_file_path = os.path.join(files_path, f)
        result_file_path = os.path.join(result_folder, f"1.0_{f}")
        generate_test_answer_with_rag(model, retriever_type, data_name,
                                      number_of_chunks, database_name,
                                      prompt_type,
                                      input_file_path,
                                      "answer",
                                      result_file_path,
                                      True)


"""
generate_test_answers("parsed_exam/test", "gemini1.5", "ensemble",
                                   "parsed_pdf_docs/parsed_pdf_docs_256.csv", 6,
                                   "db_256_64", knowledge_full_context_template(),
                                   "all_results_test/results_test_with_context_pdf")

generate_test_answer_with_rag("gemini1.5", input_excel_file="parsed_exam/test/2023_GING_Midterm_p1_Test.xlsx",
                                           col_name="answer", result_excel_file="results/2023_GING_Midterm_p1_Test_wr1.5.xlsx", rag=False)


generate_test_answer_with_rag("gemini1.0", "ensemble", "parsed_pdf_docs/test.csv", 4,
                                           "chroma_db_test_with_pdf", knowledge_full_context_template(),
                                           input_excel_file="parsed_exam/test/2022_GING_Midterm2_Theory.xlsx",
                                           col_name="answer",
                                           result_excel_file="all_results_test/all/1.0_2022_GING_Midterm2_Theory.xlsx",
                                           rag=True)                                   
"""


def generate_code_and_punts(qa, llm, question, ground_truth, criteria,
                            criteria_prompt, col_name, df, i, rag=True):
    gemini_score_col_name = "score " + col_name
    context_name = "contexts"
    criteria_chain = criteria_prompt | select_model("gemini1.5")
    try:
        print(f"Attempting to get llm_answer for question {i}")
        if rag:
            result = qa.invoke(question)
            llm_answer = result["result"]
            answer_context = result['source_documents']
            context = "\n".join(doc.page_content for doc in answer_context)
            df.loc[i, context_name] = context
        else:
            llm_answer = llm.invoke(question)["response"]
        print(f"Successfully obtained llm_answer for question {i}: {llm_answer}")
    except Exception as e:
        print(f"Error obtaining llm_answer for question {i}: {e}")
        llm_answer = "Error processing the question"
        df.loc[i, context_name] = "Error processing the question"

    try:
        llm_score = criteria_chain.invoke(
            {"possible_solution": ground_truth, "answer": llm_answer, "criteria": criteria})
        print(f"Successfully calculated llm_score for question {i}: {llm_score}")
    except Exception as e:
        print(f"Error calculating llm_score for question {i}: {e}")
        llm_score = 0

    df.loc[i, col_name] = llm_answer
    df.loc[i, gemini_score_col_name] = llm_score


def generate_code_and_punts_without_rag(model, criteria_prompt=None, prompt_type=None, input_excel_file="",
                                        col_name="", result_excel_file=""):
    """
    This function generates code and punts without using RAG.

    Args:
        model (str): The model to use.
        criteria_prompt (str): The criteria prompt.
        input_excel_file (str): The input Excel file.
        col_name (str): The column name.
        result_excel_file (str): The result Excel file.
    """
    llm = select_model(model)

    df = pd.read_excel(input_excel_file)

    for i, row in df.iterrows():
        question = row["question"]
        if i > 0 and question != df.at[i - 1, 'question'] or i == 0:
            qa = ConversationChain(
                llm=llm,
                memory=ConversationBufferMemory()
            )
            if row["description"] != "none":
                question = question + "\n" + row["description"]
        else:
            question = row["description"]
        ground_truth = row['ground_truth']
        criteria = row['criteria']

        generate_code_and_punts(qa, qa, question, ground_truth, criteria,
                                criteria_prompt, col_name, df, i, False)

    df.to_excel(result_excel_file, index=False)


"""
generate_code_and_punts_without_rag("gemini1.0", score_template(), answer_with_history_template(),
                                    input_excel_file="parsed_exam/exam/2023_Midterm.xlsx",
                                    col_name="answer",
                                    result_excel_file="all_results_exam/results_exam_wr/wr_1_2023_Midterm2.xlsx")
"""


def generate_code_and_punts_with_rag(model: str,
                                     retriever_type: str = "",
                                     data_name: str = "",
                                     criteria_prompt: str = None,
                                     number_of_chunks: int = 0,
                                     database_name: str = "",
                                     prompt_type: str = None,
                                     input_excel_file: str = "",
                                     col_name: str = "",
                                     result_excel_file: str = "") -> None:
    """
    Generates code and punts using the specified model, retriever, and prompt.

    Args:
        model (str): The model to use.
        retriever_type (str, optional): The type of retriever. Defaults to "".
        data_name (str, optional): The name of the data. Defaults to "".
        criteria_prompt (str, optional): The criteria prompt. Defaults to None.
        number_of_chunks (int, optional): The number of chunks. Defaults to 0.
        database_name (str, optional): The name of the database. Defaults to "".
        prompt_type (str, optional): The type of prompt. Defaults to None.
        input_excel_file (str, optional): The path to the input Excel file. Defaults to "".
        col_name (str, optional): The name of the column. Defaults to "".
        result_excel_file (str, optional): The path to the result Excel file. Defaults to "".
    """
    # Select the model
    llm = select_model(model)

    # Select the retriever
    retriever = select_retriever(retriever_type, data_name, number_of_chunks,
                                 database_name)

    # Read the input Excel file
    df = pd.read_excel(input_excel_file)

    # Iterate over each row in the DataFrame
    for i, row in df.iterrows():
        # Get the question from the current row
        question = row["question"]

        # Check if the question is different from the previous row or if it's the first row
        if i > 0 and question != df.at[i - 1, 'question'] or i == 0:
            # Create a RetrievalQA instance
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff",
                return_source_documents=True,
                chain_type_kwargs={
                    "verbose": True,
                    "prompt": prompt_type,
                    "memory": ConversationBufferMemory(memory_key="history", input_key="question")
                })

            # Append the description to the question if it's not "none"
            if row["description"] != "none":
                question = question + "\n" + row["description"]
        else:
            question = row["description"]

        # Get the ground truth and criteria from the current row
        ground_truth = row['ground_truth']
        criteria = row['criteria']

        # Generate code and punts
        generate_code_and_punts(qa, qa, question, ground_truth, criteria,
                                criteria_prompt, col_name, df, i, True)

    # Save the updated DataFrame to the result Excel file
    df.to_excel(result_excel_file, index=False)



def rename_files(input_excel_file: str = "", result_excel_file: str = "") -> None:
    """
    Renames the files in the input_excel_file directory and saves them to the result_excel_file directory.

    Args:
        input_excel_file (str): The directory containing the Excel files to be renamed.
        result_excel_file (str): The directory to save the renamed Excel files.

    Returns:
        None
    """
    # Get all the items in the input_excel_file directory
    all_items = os.listdir(input_excel_file)

    # Filter out directories
    files = [item for item in all_items if os.path.isfile(os.path.join(input_excel_file, item))]

    # Rename and save each file
    for f in files:
        # Get the path of the input file
        input_file_path = os.path.join(input_excel_file, f)

        # Get the path of the result file
        result_file_path = os.path.join(result_excel_file, f"sim_{f}")

        # Read the Excel file
        df = pd.read_excel(input_file_path)

        # Rename the 'question' column to 'problem'
        df = df.rename(columns={'question': 'problem'})

        # Update the 'question' column based on the 'description' column
        for i, row in df.iterrows():
            question = row["problem"]
            if row["description"] != "none":
                question = question + "\n" + row["description"]
            df.loc[i, 'question'] = question

        # Save the updated Excel file
        df.to_excel(result_file_path, index=False)


def get_similarity(compare_llm, compare_prompt, cross_encoder_model, question, ground_truth,
                   answer, col_name, df, i):
    """
    Calculates the similarity between the question, ground_truth, and answer using the compare_llm,
    compare_prompt, and cross_encoder_model.

    Args:
        compare_llm: The language model to use for comparison.
        compare_prompt: The prompt to use for comparison.
        cross_encoder_model: The cross-encoder model to use for comparison.
        question (str): The question to compare.
        ground_truth (str): The ground truth to compare.
        answer (str): The answer to compare.
        col_name (str): The name of the column to store the similarity.
        df (pandas.DataFrame): The DataFrame to store the similarity.
        i (int): The index of the row in the DataFrame.

    Returns:
        None
    """
    # Create the column names for the similarity scores
    gemini_sim_col_name = "similarity " + col_name
    cross_sim_col_name = "similarity CrossEncoder " + col_name

    # Create the best_compare_chain
    best_compare_chain = compare_prompt | compare_llm

    try:
        # Calculate the llm_answer_similarity
        llm_answer_similarity = best_compare_chain.invoke(
            {"context": question, "phrase1": ground_truth, "phrase2": answer})
        print(f"Successfully calculated llm_answer_similarity for question {i}: {llm_answer_similarity}")
    except Exception as e:
        print(f"Error calculating llm_answer_similarity for question {i}: {e}")
        llm_answer_similarity = 0

    try:
        # Calculate the cross_sim_similarity
        cross_sim_similarity = cross_encoder_model.predict([ground_truth, answer])
    except Exception as e:
        print(f"Error calculating cross_sim_similarity for question {i}: {e}")
        cross_sim_similarity = 0

    # Store the similarity scores in the DataFrame
    df.loc[i, gemini_sim_col_name] = llm_answer_similarity
    df.loc[i, cross_sim_col_name] = cross_sim_similarity


def generate_similarity(model: str, compare_prompt, input_excel_file: str = "", col_name: str = "", result_excel_file: str = "") -> None:
    """
    Generates similarity scores for each row in the input Excel file and saves the results to the result Excel file.

    Args:
        model (str): The name of the model to use for similarity calculation.
        compare_prompt: The prompt to use for comparison.
        input_excel_file (str): The path to the input Excel file.
        col_name (str): The name of the column to store the similarity scores.
        result_excel_file (str): The path to save the result Excel file.

    Returns:
        None
    """
    # Select the model to use
    llm = select_model(model)

    # Initialize the cross-encoder model
    cross_encoder_model = CrossEncoder('cross-encoder/stsb-roberta-large')

    # Read the input Excel file
    df = pd.read_excel(input_excel_file)

    # Calculate similarity scores for each row
    for i, row in df.iterrows():
        question = row["question"]
        ground_truth = row['ground_truth']
        answer = row["answer"]

        # Call get_similarity to calculate and store the similarity scores
        get_similarity(llm, compare_prompt, cross_encoder_model, question, ground_truth, answer, col_name, df, i)

    # Save the updated DataFrame to the result Excel file
    df.to_excel(result_excel_file, index=False)


"""
# Example usage of rename_files function
input_excel_file = result_excel_file = "all_results_exam/results_exam"
rename_files(input_excel_file, result_excel_file)

# Example usage of generate_similarity function
file_path = "all_results_exam/results_exam"
all_items = os.listdir(file_path)
files = [item for item in all_items if os.path.isfile(os.path.join(file_path, item))]

for f in files:
    input_file_path = result_file_path = os.path.join(file_path, f)
    generate_similarity("gemini1.5", compare_prompt=best_compare_template(),
                        input_excel_file=input_file_path,
                        col_name="answer",
                        result_excel_file=result_file_path)
"""
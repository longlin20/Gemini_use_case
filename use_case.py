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


def get_retriever(db, k):
    return db.as_retriever(search_kwargs={"k": k})


def embedding_docs(db_name):
    # "./chroma_db/vertexai/ + dataset_name"
    persist_directory = "./chroma_db_use_case/" + db_name
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
def convert_to_documents(df, content_column):
    documents = [Document(page_content=row[content_column]) for index, row in df.iterrows()]
    return documents


def get_bm25_retriever(data_name, number_of_chunks):
    df_pages = pd.read_csv(data_name)
    documents = convert_to_documents(df_pages, 'content_chunks')

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = number_of_chunks
    return bm25_retriever


def get_dense_retriever(database_name, number_of_chunks):
    dense_retriever = get_retriever(embedding_docs(database_name), number_of_chunks)
    return dense_retriever


def select_retriever(retriever_type, data_name, number_of_chunks, database_name):
    if retriever_type == "bm25":
        return get_bm25_retriever(data_name, number_of_chunks)
    elif retriever_type == "dense":
        return get_dense_retriever(database_name, number_of_chunks)
    elif retriever_type == "ensemble":
        bm25_retriever = get_bm25_retriever(data_name, number_of_chunks)
        dense_retriever = get_dense_retriever(database_name, number_of_chunks)
        return EnsembleRetriever(retrievers=[bm25_retriever, dense_retriever], weights=[0.5, 0.5])
    else:
        raise ValueError(
            "Invalid retriever type. Please choose 'bm25', 'dense', 'parents', 'ensemble' or 'ensemble_code'.")


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
all_items = os.listdir("parsed_exam/exam")
files = [item for item in all_items if os.path.isfile(os.path.join("parsed_exam/exam", item))]

for f in files:
    print(f)
    input_file_path = os.path.join("parsed_exam/exam", f)
    result_file_path = os.path.join("all_results_exam/results_exam_wr", f"wr_15_{f}")
    generate_code_and_punts_without_rag("gemini1.5", score_template(), answer_with_history_template(),
                                     input_excel_file=input_file_path,
                                     col_name="answer",
                                     result_excel_file=result_file_path)



generate_code_and_punts_without_rag("gemini1.0", score_template(), answer_with_history_template(),
                                    input_excel_file="parsed_exam/exam/2023_Midterm.xlsx",
                                    col_name="answer",
                                    result_excel_file="all_results_exam/results_exam_wr/wr_1_2023_Midterm2.xlsx")
"""


def generate_code_and_punts_with_rag(model, retriever_type="", data_name="", criteria_prompt=None,
                                     number_of_chunks=0, database_name="", prompt_type=None,
                                     input_excel_file="",
                                     col_name="", result_excel_file=""):
    llm = select_model(model)

    retriever = select_retriever(retriever_type, data_name, number_of_chunks,
                                 database_name)

    df = pd.read_excel(input_excel_file)

    for i, row in df.iterrows():
        question = row["question"]
        if i > 0 and question != df.at[i - 1, 'question'] or i == 0:
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
            if row["description"] != "none":
                question = question + "\n" + row["description"]
        else:
            question = row["description"]
        ground_truth = row['ground_truth']
        criteria = row['criteria']

        generate_code_and_punts(qa, qa, question, ground_truth, criteria,
                                criteria_prompt, col_name, df, i, True)

    df.to_excel(result_excel_file, index=False)


"""
all_items = os.listdir("parsed_exam/exam")
files = [item for item in all_items if os.path.isfile(os.path.join("parsed_exam/exam", item))]



for f in files:
    input_file_path = os.path.join("parsed_exam/exam", f)
    result_file_path = os.path.join("all_results_exam/results_exam", f"1_6chunks_{f}")
    generate_code_and_punts_with_rag("gemini1.0", "ensemble", "parsed_pdf_docs/parsed_pdf_docs_256.csv",
                                     score_template(), 6, database_name="db_256_64",
                                     prompt_type=answer_with_history_template(), input_excel_file=input_file_path,
                                     col_name="answer", result_excel_file=result_file_path)



generate_code_and_punts_with_rag("gemini1.0", "ensemble", "parsed_pdf_docs/parsed_pdf_docs_256.csv",
                                 score_template(), 6, database_name="db_256_64",
                                 prompt_type=answer_with_history_template(), input_excel_file="parsed_exam/exam/2023_Midterm.xlsx",
                                 col_name="answer", result_excel_file="all_results_exam/results_exam/1_2023_Midterm.xlsx")
"""


def rename_files(input_excel_file="", result_excel_file=""):
    all_items = os.listdir(input_excel_file)
    files = [item for item in all_items if os.path.isfile(os.path.join(input_excel_file, item))]
    for f in files:
        input_file_path = os.path.join(input_excel_file, f)
        result_file_path = os.path.join(result_excel_file, f"sim_{f}")
        df = pd.read_excel(input_file_path)
        df = df.rename(columns={'question': 'problem'})
        for i, row in df.iterrows():
            question = row["problem"]
            if row["description"] != "none":
                question = question + "\n" + row["description"]
            df.loc[i, 'question'] = question

        df.to_excel(result_file_path, index=False)


""""
input_excel_file = result_excel_file = "all_results_exam/results_exam"
rename_files(input_excel_file, result_excel_file)
"""


def get_similarity(compare_llm, compare_prompt, cross_encoder_model, question, ground_truth,
                   answer, col_name, df, i):
    gemini_sim_col_name = "similarity " + col_name
    cross_sim_col_name = "similarity CrossEncoder " + col_name
    best_compare_chain = compare_prompt | compare_llm

    try:
        llm_answer_similarity = best_compare_chain.invoke(
            {"context": question, "phrase1": ground_truth, "phrase2": answer})
        print(f"Successfully calculated llm_answer_similarity for question {i}: {llm_answer_similarity}")
    except Exception as e:
        print(f"Error calculating llm_answer_similarity for question {i}: {e}")
        llm_answer_similarity = 0

    try:
        cross_sim_similarity = cross_encoder_model.predict([ground_truth, answer])
    except Exception as e:
        print(f"Error calculating cross_sim_similarity for question {i}: {e}")
        cross_sim_similarity = 0

    df.loc[i, gemini_sim_col_name] = llm_answer_similarity
    df.loc[i, cross_sim_col_name] = cross_sim_similarity


def generate_similarity(model, compare_prompt, input_excel_file="", col_name="", result_excel_file=""):
    llm = select_model(model)
    cross_encoder_model = CrossEncoder('cross-encoder/stsb-roberta-large')
    df = pd.read_excel(input_excel_file)

    for i, row in df.iterrows():
        question = row["question"]
        ground_truth = row['ground_truth']
        answer = row["answer"]

        get_similarity(llm, compare_prompt, cross_encoder_model, question, ground_truth, answer, col_name, df, i)

    df.to_excel(result_excel_file, index=False)


"""
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
"""
def average_faithfulness(file_path):
    total_answer = 0
    count = 0  # To keep track of the number of valid 'similarity answer' entries
    all_items = os.listdir(file_path)
    files = [item for item in all_items if os.path.isfile(os.path.join(file_path, item))]

    for f in files:
        if f.endswith('.xlsx'):
            df = pd.read_excel(os.path.join(file_path, f))
            # Convert 'similarity answer' to numeric, setting errors to 'coerce' to handle invalid strings by converting them to NaN
            df['similarity answer'] = pd.to_numeric(df['similarity answer'], errors='coerce')
            # Drop rows where 'similarity answer' is NaN after conversion
            valid_answers = df['similarity answer'].dropna()
            total_answer += valid_answers.sum()  # Add the sum of valid 'similarity answer' entries
            count += valid_answers.count()  # Increment the count by the number of valid entries

    if count == 0:  # To avoid division by zero
        return 0
    else:
        return total_answer / count  # Calculate and return the average



file_path = "all_results_exam/results_exam_ragas"
total_answer = average_faithfulness(file_path)
print(total_answer)
file_path = "all_results_exam/results_exam_wr_ragas"
total_answer = average_faithfulness(file_path)
print(total_answer)
"""

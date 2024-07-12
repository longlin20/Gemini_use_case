import os
import time

import google.auth
import pandas as pd
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers.bm25 import BM25Retriever

from custum_template import answer_with_history_template, score_template, best_compare_template, \
    knowledge_full_context_template
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
    documents = convert_to_documents(df_pages, 'content_pages')

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


def generate_llm_answer_and_accuracy(qa, llm, question, col_name, df, i, rag):
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


def generate_test_answer_and_accuracy_with_rag(model, retriever_type="", data_name="",
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
                "verbose": False,
                "prompt": prompt_type,
            })
    else:
        qa = None

    df = pd.read_excel(input_excel_file)

    question_count = 0

    for i, row in df.iterrows():
        question = row['question']

        generate_llm_answer_and_accuracy(qa, llm, question,
                                         col_name, df, i, rag)

        question_count += 1
        if model == "gemini1.5" and question_count % 60 == 0:
            time.sleep(15)
    df.to_excel(result_excel_file, index=False)


def generate_test_answers_and_accuracy(files_path, model, retriever_type, data_name,
                                       number_of_chunks, database_name, prompt_type, result_folder):
    all_items = os.listdir(files_path)
    files = [item for item in all_items if os.path.isfile(os.path.join(files_path, item))]

    for f in files:
        for i in range(1, 3):
            if i == 1:
                input_file_path = os.path.join(files_path, f)
                result_file_path = os.path.join(result_folder, f"1.0_{f}")
            else:
                input_file_path = result_file_path
            generate_test_answer_and_accuracy_with_rag(model, retriever_type, data_name,
                                                       number_of_chunks, database_name,
                                                       prompt_type,
                                                       input_file_path,
                                                       f"answer{i}",
                                                       result_file_path,
                                                       True)




"""
generate_test_answers_and_accuracy("parsed_exam/test", "gemini1.0", "ensemble",
                                   "parsed_pdf_docs/result_context_text.csv", 4,
                                   "chroma_db_test", knowledge_full_context_template(),
                                   "results2")

generate_test_answers_and_accuracy("parsed_exam/test", "gemini1.5", "ensemble",
                                   "parsed_pdf_docs/result_context_text.csv", 4,
                                   "chroma_db_test", knowledge_full_context_template(),
                                   "results2")
"""
"""
generate_test_answer_and_accuracy_with_rag("gemini1.5", input_excel_file="parsed_exam/test/2023_GING_Midterm_p1_Test.xlsx",
                                           col_name="answer", result_excel_file="results/2023_GING_Midterm_p1_Test_wr1.5.xlsx", rag=False)

generate_test_answer_and_accuracy_with_rag("gemini1.5", "ensemble", "parsed_pdf_docs/parsed_pdf_docs_256_without_text.csv", 4,
                                           "chroma_db_256_without_text", knowledge_full_context_template(),
                                           input_excel_file="parsed_exam/test/2023_GING_Midterm_p1_Test.xlsx",
                                           col_name="answer",
                                           result_excel_file="results/2023_GING_Midterm_p1_Test_256wtext.xlsx",
                                           rag=True)                                   
"""
"""
generate_test_answer_and_accuracy_with_rag("gemini1.5", "ensemble", "parsed_pdf_docs/result_context_text.csv", 6,
                                           "chroma_db_test", knowledge_full_context_template(),
                                           input_excel_file="parsed_exam/test/2023_Ordinary-call_Theory_final.xlsx",
                                           col_name="answer",
                                           result_excel_file="results2/1.5_2023_Ordinary-call_Theory_final2.xlsx",
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
            print(f"Successfully obtained result for question {i}: {result}")
            llm_answer = result["result"]
            answer_context = result['source_documents']
            context = "\n".join(doc.page_content for doc in answer_context)
            df.loc[i, context_name] = context
        else:
            llm_answer = llm.invoke(question)["response"]
            print(f"Successfully obtained llm_answer for question {i}: {llm_answer}")
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
            question = question + "\n" + row["description"]
        else:
            question = row["description"]
        ground_truth = row['code']
        criteria = row['criteria']

        generate_code_and_punts(qa, llm, question, ground_truth, criteria,
                                criteria_prompt, col_name, df, i, True)

    df.to_excel(result_excel_file, index=False)


def generate_code_and_punts_without_rag(model, criteria_prompt=None, input_excel_file="",
                                        col_name="", result_excel_file=""):
    llm = select_model(model)

    df = pd.read_excel(input_excel_file)

    for i, row in df.iterrows():
        question = row["question"]
        if i > 0 and question != df.at[i - 1, 'question'] or i == 0:
            qa = ConversationChain(
                llm=llm,
                verbose=True,
                memory=ConversationBufferMemory()
            )
            question = question + "\n" + row["description"]
        else:
            question = row["description"]
        ground_truth = row['code']
        criteria = row['criteria']

        generate_code_and_punts(qa, qa, question, ground_truth, criteria,
                                criteria_prompt, col_name, df, i, False)

    df.to_excel(result_excel_file, index=False)

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
        if i > 0 and question != df.at[i - 1, 'question'] or i == 0:
            question = question + "\n" + row["description"]
        else:
            question = row["description"]
        ground_truth = row['code']
        answer = row["answer"]
        #criteria = row['criteria']

        get_similarity(llm, compare_prompt, cross_encoder_model, question, ground_truth, answer, col_name, df, i)

    df.to_excel(result_excel_file, index=False)

def process_files():
    all_items = os.listdir("similarity_exam2")
    files = [item for item in all_items if os.path.isfile(os.path.join("similarity_exam2", item))]

    for f in files:
        input_file_path = os.path.join("similarity_exam2", f)
        result_file_path = os.path.join("similarity_exam_ragas2", f"ragas_{f}")
        df = pd.read_excel(input_file_path)
        df = df.rename(columns={'question': 'problem', 'code': 'ground_truth'})
        for i, row in df.iterrows():
            question = row["problem"]
            question = question + "\n" + row["description"]
            df.loc[i, 'question'] = question

        df.to_excel(result_file_path, index=False)

#process_files()

"""
generate_similarity("gemini1.5", compare_prompt=best_compare_template(), input_excel_file="results_exam/1_2022_GING_Midterm1_structured_problems.xlsx",
                    col_name="answer", result_excel_file="results_exam/similarity/similar_1_2022_GING_Midterm1_structured_problems.xlsx")
"""

"""
all_items = os.listdir("parsed_exam/exam")
files = [item for item in all_items if os.path.isfile(os.path.join("parsed_exam/exam", item))]

for f in files:
    input_file_path = os.path.join("parsed_exam/exam", f)
    result_file_path = os.path.join("parsed_exam/exam", f"1_{f}")
    generate_code_and_punts_with_rag("gemini1.0", "ensemble", "parsed_pdf_docs/parsed_pdf_docs_256.csv",
                                               score_template(), 6, "chroma_db_256", answer_with_history_template(),
                                               input_excel_file=input_file_path,
                                               col_name="answer",
                                               result_excel_file=result_file_path)
    """
"""
all_items = os.listdir("results_exam2")
files = [item for item in all_items if os.path.isfile(os.path.join("results_exam2", item))]
for f in files:
    input_file_path = os.path.join("results_exam2", f)
    result_file_path = os.path.join("similarity_exam2", f)
    generate_similarity("gemini1.5", compare_prompt=best_compare_template(),
                        input_excel_file=input_file_path,
                        col_name="answer",
                        result_excel_file=result_file_path)
"""
"""
generate_code_and_punts_without_rag("gemini1.5", score_template(),
                                     input_excel_file="parsed_exam/exam/2023_GING_Midterm_structured_problems.xlsx",
                                     col_name="answer",
                                     result_excel_file="parsed_exam/exam"
                                                       "/wr_1.5_2023_GING_Midterm_structured_problems.xlsx")
                                                       """
import time
import json
import numpy as np
from pathlib import Path
from langchain.schema.retriever import BaseRetriever

from repo_rag.components.vectorstore import Vectorstore
from repo_rag.components.constants import EVAL_DATA_PATH


def recall_at_k(retriever: BaseRetriever, queries: list[dict[str, list[str]]], k: int = 10) -> float:
    """
    Computes Recall@K for the given retriever.

    Parameters
    ----------
    retriever : BaseRetriever
        The retriever object used for retrieving documents.
    queries : list of dict[str, list[str]]
        A list of queries, where each query contains a 'question' (str) and a list of relevant file names.
    k : int, optional
        The number of filenames to consider, by default 10.

    Returns
    -------
    float
        The average Recall@K score across all queries.
    """
    recalls = []

    for query in queries:
        question = query['question']
        relevant_files = set(query['files'])

        retrieved_docs = retriever.invoke(question)

        retrieved_files = set()
        for doc in retrieved_docs:
            if len(retrieved_files) < k:
                retrieved_files.add(doc.metadata['file_name'])
            else:
                break

        hits = len(retrieved_files & relevant_files)
        recall = hits / len(relevant_files) if relevant_files else 0
        recalls.append(recall)

    return np.mean(recalls)


def main():
    vectorstore = Vectorstore(1)
    loaded_vectorstore = vectorstore.load()
    retriever = loaded_vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 50})

    eval_data_path = Path(EVAL_DATA_PATH)

    with eval_data_path.open('r', encoding='utf-8') as f:
        queries = json.load(f)

    start_time = time.time()
    recall_score = recall_at_k(retriever, queries, k=10)
    elapsed_time = time.time() - start_time
    len_queries = len(queries)
    mean_time = elapsed_time / len_queries

    print(f'Recall@10: {recall_score:.2f}')  # 0.62
    print(f'Mean execution time: {mean_time:.4f} seconds')  # 0.55


if __name__ == '__main__':
    main()

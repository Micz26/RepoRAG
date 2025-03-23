import time
import json
import numpy as np
from pathlib import Path
from langchain.schema.retriever import BaseRetriever

from repo_rag.components.vectorstore import Vectorstore
from repo_rag.components.constants import EVAL_DATA_PATH
from repo_rag.components.chains import augment_query_chain


def recall_at_k(
    retriever: BaseRetriever,
    queries: list[dict[str, list[str]]],
    k: int = 10,
) -> float:
    """
    Computes Recall@K for the given retriever.

    Args:
        retriever: The retriever object used for retrieving documents.
        queries: List of queries with expected relevant file names.
        k: The number of unique filenames to consider. Defaults to 10.

    Returns:
        The average Recall@K score across all queries.
    """
    recalls = []

    for query in queries:
        question = query['question']
        relevant_files = set(query['files'])

        response = augment_query_chain.invoke(question)
        if 'CALLED_RETRIEVER' in response.content:
            response.content = response.content.replace('CALLED_RETRIEVER', '')
            response = question + response.content
        else:
            response = question

        retrieved_docs = retriever.invoke(response)

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

    print(f'Recall@10: {recall_score:.2f}')  # 0.67
    print(f'Mean execution time: {mean_time:.4f} seconds')  # 1.51


if __name__ == '__main__':
    main()

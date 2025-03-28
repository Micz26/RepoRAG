import time
import json
from typing import Literal
import numpy as np
from pathlib import Path
from langchain.schema.retriever import BaseRetriever

from repo_rag.components.vectorstore import Vectorstore
from repo_rag.components.constants import EVAL_DATA_PATH
from repo_rag.components.reranker import Reranker


def recall_at_k(
    retriever: BaseRetriever,
    reranker_model: Literal['listwise', 'cross-encoder'],
    queries: list[dict[str, list[str]]],
    k: int = 10,
) -> float:
    """
    Computes Recall@K for the given retriever.

    Parameters
    ----------
    retriever : BaseRetriever
        The retriever object used for retrieving documents.
    reranker_model: Literal['listwise', 'cross-encoder']
        The rereanker used for reranking
    queries : list of dict[str, list[str]]
        A list of queries, where each query contains a 'question' (str) and a list of relevant file names.
    k : int, optional
        The number of unique filenames to consider, by default 10.

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

        reranked_docs = Reranker.rerank(reranker_model, question, retrieved_docs)

        retrieved_files = set()
        for doc in reranked_docs:
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
    retriever = loaded_vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 25})

    eval_data_path = Path(EVAL_DATA_PATH)

    with eval_data_path.open('r', encoding='utf-8') as f:
        queries = json.load(f)

    for reranker_model in ['cross-encoder', 'listwise']:
        start_time = time.time()
        recall_score = recall_at_k(retriever, reranker_model, queries, k=10)
        elapsed_time = time.time() - start_time

        print(f'\n{reranker_model} reranker evaluation:')
        len_queries = len(queries)
        mean_time = elapsed_time / len_queries

        print(f'Recall@10: {recall_score:.2f}')
        print(f'Mean execution time: {mean_time:.4f} seconds')
        # 0.57 5.76
        #


if __name__ == '__main__':
    main()

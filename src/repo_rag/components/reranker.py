from typing import Literal
import torch
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import CrossEncoder

from langchain_core.documents import Document


class Reranker:
    @staticmethod
    def rerank(model: Literal['listwise', 'cross-encoder'], query: str, documents: list[Document]) -> list[Document]:
        """
        Reranks the documents based on the specified model.

        args:
            model (Literal['listwise', 'cross-encoder']): The name of the model ('listwise' or 'cross-encoder')
            query (str): The user query
            documents (list[Document]): List of documents to evaluate

        returns:
            sorted_documents (list[Document]): Sorted list of documents based on relevance with their respective scores
        """
        if model == 'listwise':
            return Reranker.listwise_rerank(query, documents)
        elif model == 'cross-encoder':
            return Reranker.cross_encoder_rerank(query, documents)
        else:
            raise ValueError(f'Unsupported model: {model}')

    @staticmethod
    def listwise_rerank(query: str, documents: list[Document]) -> list[Document]:
        """
        Reranks documents using the Listwise model.

        args:
            query (str): The user query
            documents (list[Document]): List of documents

        returns:
            sorted_documents (list[Document]): Sorted list of documents based on similarity with their respective scores
        """
        model_name = 'ByteDance/ListConRanker'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        inputs = tokenizer(
            [query] * len(documents),
            [doc.page_content for doc in documents],
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors='pt',
        )

        with torch.no_grad():
            outputs = model(**inputs)

        embeddings = outputs.last_hidden_state[:, 0, :]
        query_embedding = embeddings[0].unsqueeze(0)
        doc_embeddings = embeddings[1:]

        similarities = torch.nn.functional.cosine_similarity(query_embedding, doc_embeddings)

        return [documents[i] for i in similarities.argsort(descending=True)]

    @staticmethod
    def cross_encoder_rerank(query: str, documents: list[Document]) -> list[Document]:
        """
        Reranks documents using the Cross-Encoder model.

        args:
            query (str): The user query
            documents (list[Document]): List of documents

        returns:
            sorted_documents (list[Document]): Sorted list of documents based on the model's scores with their respective scores
        """
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        scores = model.predict([(query, doc.page_content) for doc in documents])

        return [doc for _, doc in sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)]

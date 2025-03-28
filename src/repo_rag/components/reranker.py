from typing import Literal
import torch
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import CrossEncoder

from langchain_core.documents import Document


class Reranker:
    """
    A component for reranking retrieved documents based on relevance to a given query.

    Supports two reranking models:
    - Listwise ranking using a transformer-based approach.
    - Cross-encoder ranking for pairwise relevance scoring.

    Methods:
        - rerank: Selects the reranking model and applies it to a list of documents.
        - listwise_rerank: Ranks documents using a listwise approach.
        - cross_encoder_rerank: Ranks documents using a cross-encoder model.
    """

    @staticmethod
    def rerank(model: Literal['listwise', 'cross-encoder'], query: str, documents: list[Document]) -> list[Document]:
        """
        Reranks the documents based on the specified model.

        ----------
        model : Literal[&#39;listwise&#39;, &#39;cross
            The name of the model ('listwise' or 'cross-encoder')
        query : str
            The user query
        documents : list[Document]
            List of documents to evaluate

        Returns
        -------
        list[Document]
            Sorted list of documents based on relevance with their respective scores

        Raises
        ------
        ValueError
            _description_
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

        Parameters
        ----------
        query : str
            The user query
        documents : list[Document]
            List of documents

        Returns
        -------
        list[Document]
            Sorted list of documents based on similarity with their respective scores
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

        Parameters
        ----------
        query : str
            The user query
        documents : list[Document]
            List of documents

        Returns
        -------
        list[Document]
            Sorted list of documents based on the model's scores with their respective scores
        """
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        scores = model.predict([(query, doc.page_content) for doc in documents])

        return [doc for _, doc in sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)]

import time
import logging
from pathlib import Path
from uuid import uuid4
import faiss

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from repo_rag.components.constants import VECTORSTORE_PATH
from repo_rag.components.embeddings import openai_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Vectorstore:
    """Vectorstore class"""

    embeddings = openai_embeddings

    def __init__(self, version: int = 0):
        self.vectorstore_path = VECTORSTORE_PATH + f'_v{version}'

    def create(self) -> FAISS:
        """
        Creates a new vectorstore if it does not exist.

        Returns
        -------
        FAISS
            FAISS vectorstore
        """
        vectorstore_path = Path(self.vectorstore_path)
        if vectorstore_path.exists():
            return self.load()

        index = faiss.IndexFlatIP(1536)
        docstore = InMemoryDocstore()
        index_to_docstore_id = {}

        vectorstore = FAISS(
            embedding_function=Vectorstore.embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
        )

        vectorstore.save_local(self.vectorstore_path)
        return vectorstore

    def add_docs(self, docs: list[Document], batch_size: int) -> None:
        """
        Adds documents in batches of batch_size to avoid rate limiting issues.
        If there are more than batch_size documents, the function waits 1 minute before continuing.

        Parameters
        ----------
        docs : list[Document]
            list of documents to be uploaded
        batch_size : int
            batch size
        """
        try:
            vectorstore = self.load()
        except FileNotFoundError:
            vectorstore = self.create()

        total_docs = len(docs)
        for i in range(0, total_docs, batch_size):
            batch = docs[i : i + batch_size]
            ids = [uuid4() for _ in batch]

            vectorstore.add_documents(documents=batch, ids=ids)
            vectorstore.save_local(self.vectorstore_path)

            logger.info(f'Added {len(batch)} documents to vectorstore. ({i + len(batch)}/{total_docs})')

            if i + batch_size < total_docs:
                logger.warning('Rate limit reached. Sleeping for 1 minute...')
                time.sleep(60)

    def load(self) -> FAISS:
        """
        Loads an existing vectorstore, or raises an error if not found.

        Returns
        -------
        FAISS
            loaded FAISS vectorstore

        Raises
        ------
        FileNotFoundError
            if vectorstore was not found
        """
        vectorstore_path = Path(self.vectorstore_path)
        if not vectorstore_path.exists():
            raise FileNotFoundError("Vectorstore not found. Create it using 'Vectorstore.create()'.")

        return FAISS.load_local(
            vectorstore_path,
            Vectorstore.embeddings,
            allow_dangerous_deserialization=True,
        )

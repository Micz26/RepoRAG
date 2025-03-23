from langchain.text_splitter import RecursiveCharacterTextSplitter
from repo_rag.components.loader import Loader
from repo_rag.components.vectorstore import Vectorstore


def main():
    repo_url = input("Enter GitHub repository URL (or press Enter to use default 'escrcpy'): ").strip()
    if not repo_url:
        repo_url = 'https://github.com/viarotel-org/escrcpy'

    print(f'Using repository: {repo_url}')

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64, separators=['\n', ' ', ''])

    documents = Loader.load_and_split(repo_url, text_splitter)

    vectorstore = Vectorstore(1)
    vectorstore.create()
    vectorstore.add_docs(documents, batch_size=900)

    index = vectorstore.load().index
    print(f'Total documents in index: {index.ntotal}')


if __name__ == '__main__':
    main()

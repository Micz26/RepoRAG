from langchain.text_splitter import RecursiveCharacterTextSplitter
from repo_rag.components.loader import Loader
from repo_rag.components.vectorstore import Vectorstore


def main():
    """
    Build index by providing repository url
    """
    repo_url = input("Enter GitHub repository URL (or press Enter to use default 'escrcpy'): ").strip()
    batch_size = input(
        'Enter batch size (number of documents per batch for vector store upload, recommended: 900): '
    ).strip()

    if not batch_size:
        batch_size = 900
    else:
        try:
            batch_size = int(batch_size)
        except ValueError:
            print('Invalid input. Using default batch size: 900.')
            batch_size = 900

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

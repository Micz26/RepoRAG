import os
import logging
import requests
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.text_splitter import TextSplitter

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Loader:
    """
    A component for fetching and processing files from a GitHub repository.

    This class interacts with the GitHub API to retrieve files while filtering out
    unsupported file types. It supports loading raw file content and splitting
    documents into smaller chunks for further processing.
    """

    GITHUB_API = 'https://api.github.com/repos/'
    token = os.getenv('GITHUB_TOKEN')

    IGNORED_EXTENSIONS = {
        '.png',
        '.jpg',
        '.jpeg',
        '.gif',
        '.bmp',
        '.ico',
        '.svg',
        '.tiff',
        '.mp3',
        '.wav',
        '.mp4',
        '.avi',
        '.mov',
        '.mkv',
        '.flv',
        '.pdf',
        '.zip',
        '.tar',
        '.gz',
        '.rar',
        '.7z',
        '.exe',
        '.dll',
        '.so',
        '.bin',
        '.obj',
        '.class',
        '.pyc',
        '.pyo',
        '.woff',
        '.woff2',
        '.ttf',
        '.otf',
        '.eot',
        '.inc',
    }

    @staticmethod
    def _get_repo_files(repo_url: str, token: str | None = None):
        """Fetches files from a repository, filtering out ignored file types.

        Parameters
        ----------
        repo_url : str
            URL of the GitHub repository.
        token : str | None, optional
            GitHub authentication token, by default None.

        Returns
        -------
        list of tuple
            List of tuples containing (file_name, download_url, full_file_url, repo_url).
        """
        repo_name = repo_url.rstrip('/').split('github.com/')[-1]
        api_url = f'{Loader.GITHUB_API}{repo_name}/contents/'

        def fetch_files(url: str, path: str = ''):
            """
            Helper function to recursively fetch files from the repository.
            """
            headers = {'Authorization': f'token {token or Loader.token}'} if token or Loader.token else {}

            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                logger.error(f'Error fetching: {url} | Status: {response.status_code}')
                return []

            files = []
            for item in response.json():
                if item['type'] == 'file':
                    file_ext = os.path.splitext(item['name'])[1].lower()
                    if file_ext in Loader.IGNORED_EXTENSIONS:
                        continue

                    full_file_url = f'{repo_url}/blob/main/{path}{item["name"]}'
                    files.append((path + item['name'], item['download_url'], full_file_url, repo_url))

                elif item['type'] == 'dir':
                    files.extend(fetch_files(item['url'], path + item['name'] + '/'))

            return files

        return fetch_files(api_url)

    @staticmethod
    def _get_file_content(file_url: str, token: str | None = None):
        """Fetches the content of a single file, including the token in the header (if provided).

        Parameters
        ----------
        file_url : str
            URL of the file to fetch.
        token : str | None, optional
            GitHub authentication token, by default None.

        Returns
        -------
        str | None
            Content of the file as a string, or None if an error occurs.
        """
        headers = {'Authorization': f'token {token or Loader.token}'} if token or Loader.token else {}

        response = requests.get(file_url, headers=headers)
        if response.status_code == 200:
            return response.text

        logger.error(f'Error fetching file content: {file_url} | Status: {response.status_code}')
        return None

    @staticmethod
    def load(repo_url: str, token: str | None = None):
        """
        Loads files from the repository and returns them as documents.

        Parameters
        ----------
        repo_url : str
            URL of the GitHub repository.
        token : str | None, optional
            GitHub authentication token, by default None.

        Returns
        -------
        list of Document
            List of Document objects representing the loaded files.
        """
        file_docs = []
        files = Loader._get_repo_files(repo_url, token)
        for file_name, raw_url, full_url, repo_url in files:
            file_content = Loader._get_file_content(raw_url, token)
            if file_content:
                file_doc = Document(
                    page_content=file_content,
                    metadata={'file_name': file_name, 'full_url': full_url, 'repo_url': repo_url},
                )
                file_docs.append(file_doc)
        return file_docs

    @staticmethod
    def load_and_split(repo_url: str, splitter: TextSplitter, token: str | None = None):
        """Loads files from the repository, splits them into chunks, and returns them as documents.

        Parameters
        ----------
        repo_url : str
            URL of the GitHub repository.
        splitter : TextSplitter
            TextSplitter instance used to divide documents into smaller chunks.
        token : str | None, optional
            GitHub authentication token, by default None.

        Returns
        -------
        list of Document
            List of split Document objects.
        """
        file_docs = []
        files = Loader._get_repo_files(repo_url, token)
        for file_name, raw_url, full_url, url in files:
            file_content = Loader._get_file_content(raw_url, token)

            if file_content:
                file_doc = Document(
                    page_content=file_content,
                    metadata={'file_name': file_name, 'full_url': full_url, 'repo_url': url},
                )
                split_file_doc = splitter.split_documents([file_doc])
                file_docs += split_file_doc

        return file_docs

import os

from dotenv import load_dotenv

load_dotenv()

VECTORSTORE_PATH = 'vectorstores/vectorstore'

OPEN_AI_API_KEY = os.getenv('OPEN_AI_API_KEY', '')

EVAL_DATA_PATH = 'data/escrcpy-commits-generated.json'

import os
import shutil
from git import Repo
from sklearn.feature_extraction.text import TfidfVectorizer
import tempfile

from aiclients import openai_client
from db.vector_utils import string_to_vector

from shared import logger

log = logger.get_logger(__name__)



def clone_and_vectorize(repo_url, extensions=['.js', '.py', '.java', '.ts', '.go']):

    top_n = 5
    with tempfile.TemporaryDirectory() as temp_dir:
        # Clone the repository
        Repo.clone_from(repo_url, temp_dir, depth=1)

        # Find the top N largest files of the specified types
        file_sizes = []
        for root, dirs, files in os.walk(temp_dir):
            dirs[:] = [d for d in dirs if d not in ['dist', 'target', 'node_modules'] and not d.startswith('.')]
            for file in files:
                if file.endswith(tuple(extensions)):
                    file_path = os.path.join(root, file)
                    file_sizes.append((file_path, os.path.getsize(file_path)))

        largest_files = sorted(file_sizes, key=lambda x: x[1], reverse=True)[:top_n]

        # Read and concatenate the source code from specific file type
        source_code = ""
        for file_path, _ in largest_files:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                source_code += f.read() + "\n"

        # Vectorize the source code
        log.info("Vectorizing source code")
        log.info(source_code)

        vectorized_source_code = string_to_vector(source_code)
        return vectorized_source_code




# Example usage
# repo_url = 'your-repository-url'  # Replace with the GitHub repository URL
# vectorized_code = clone_and_vectorize(repo_url)

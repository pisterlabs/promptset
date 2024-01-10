print("Loading libraries...")
import datasets
#import faiss
from io import BytesIO
import json
import logging
import numpy as np
import os
import tqdm
from typing import List, Optional, Tuple
import urllib.request
import zipfile

from .model_types import OpenAIModel, InstructorModel, ModelType

# Module constants

# Map from url to what we need to append to get the zip file
supported_remote_repositories = {
    'https://github.com': "/archive/refs/heads/main.zip",
    #'https://gitlab.com': "/-/archive/main/REPONAME-main.zip", # Missing repository name, which would be difficult to get into here.
    'https://bitbucket.org': "/get/main.zip",
}

# Exported functions
def generate_embeddings_for_repository(
        dataset_name: str,
        repo_url_or_path: str,
        embeddings_dir: str,
        model_type: str = 'instructor',
        model_name: Optional[str] = None) -> None:
    """
    Generate embeddings for a repository.

    Parameters
    ----------
    dataset_name : str
        A name for the newly generated dataset.
    repo_url_or_path : str
        The URL or local path to the repository to generate embeddings for.
    embeddings_dir : str
        The directory to save the generated embeddings dataset to. This should be the kept the same when querying the dataset.
    model_type : str, optional
        The type of model to use for generating the embeddings. Options: instructor, openai. Default: instructor
    model_name : str, optional
        The name of the model to use for generating the embeddings. Options available depend on the model type.
    """
    if dataset_exists(dataset_name, embeddings_dir):
        # To help the user, give the full disk path to the embeddings directory
        embeddings_dir_expanded = os.path.abspath(embeddings_dir)

        logging.error(f'Dataset named {dataset_name} already exists in embeddings directory ({embeddings_dir_expanded}), delete it first if you want to regenerate it.')
        return
    
    logging.info("Loading model...")
    embedding_model = create_embedding_model(model_type, model_name)
    
    # Check if given repository is a URL or a local path
    if repo_url_or_path.startswith('http'):
        # Remote repositories
        if repo_url_or_path.endswith('.zip'):
            generate_embeddings_for_remote_zip_archive(dataset_name, repo_url_or_path, embeddings_dir, embedding_model)
        elif is_supported_remote_repository(repo_url_or_path):
            generate_embeddings_for_remote_repository_archive(dataset_name, repo_url_or_path, embeddings_dir, embedding_model)
        else:
            logging.error(f'Unsupported remote repository: {repo_url_or_path}')
            return
    else:
        # Local repositories
        if repo_url_or_path.startswith('~'):
            repo_url_or_path = os.path.expanduser(repo_url_or_path)
        
        if os.path.isdir(repo_url_or_path):
            generate_embeddings_for_local_repository(dataset_name, repo_url_or_path, embeddings_dir, embedding_model)
        elif repo_url_or_path.endswith('.zip'):
            generate_embeddings_for_local_zip_archive(dataset_name, repo_url_or_path, embeddings_dir, embedding_model)
        else:
            logging.error(f'Unsupported local repository: {repo_url_or_path}')
            return



def query_embeddings(
        dataset_name: str,
        query: str,
        embeddings_dir: str) -> Tuple[List[float], List[str], List[str]]:
    """
    Query a dataset using natural language.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to query.
    query : str
        The query to search for.
    embeddings_dir : str
        The directory containing embedding datasets.

    Returns
    -------
    Tuple[List[float], List[str], List[str]]
        A tuple of (similarities, estimated_location, file_path)
        similarities - a float between 0.0 and 1.0 representing the similarity of the query to the file. 1.0 is a perfect match.
        estimated_location - a string representing the estimated location of the query in the file. For example, '50%' means the query is estimated to be halfway through the file.
        file_path - the path to the file in the dataset.
    """
    if not dataset_exists(dataset_name, embeddings_dir):
        # To help the user, give the full disk path to the embeddings directory
        embeddings_dir_expanded = os.path.abspath(embeddings_dir)

        logging.error(f'Dataset named {dataset_name} does not exist in embeddings directory ({embeddings_dir_expanded}), generate it first.')
        raise FileNotFoundError(f'Could not find dataset {dataset_name}.')
    
    # Load the dataset from disk.
    logging.info("Loading dataset...")
    dataset = datasets.load_from_disk(os.path.join(embeddings_dir, dataset_name))

    # Read the metadata file.
    metadata_file_path = os.path.join(embeddings_dir, dataset_name, 'metadata.json')
    if os.path.exists(metadata_file_path):
        with open(metadata_file_path, 'r') as metadata_file:
            metadata = json.load(metadata_file)
    else:
        logging.warn("Could not find metadata file, assuming default values (instructor - hkunlp/instructor-large).")
        metadata = {
            'model_type': 'instructor',
            'model_name': 'hkunlp/instructor-large',
        }
        logging.info(f"You can create a metadata file at {metadata_file_path} to specify the model type and model name to use.")
        logging.info(f"Example metadata file contents: {json.dumps(metadata, indent=4)}")


    # Load the model
    logging.info("Loading model...")
    embedding_model = create_embedding_model(metadata['model_type'], metadata['model_name'])
    
    query_embedding = embedding_model.generate_embedding_for_query(query)

    # Load the index from disk.
    # TODO

    logging.info('Querying embeddings...')
    similarities = []
    estimated_location = []

    bar = tqdm.tqdm(dataset)
    for row in bar:
        bar.set_postfix(best_match = f'{round(max(similarities, default=0.0) * 100.0, 0)}%')
        all_embeddings_for_document = row['embeddings']
        if len(all_embeddings_for_document) == 0:
            similarities.append(0.0)
            estimated_location.append('0%')
            continue

        # Each file may have more than one embedding
        # Check them all to see which one is most similar to the query
        best_similarity = 0.0
        best_similarity_index = -1

        for index, embedding in enumerate(all_embeddings_for_document):
            similarity = cosine_similarity(query_embedding, embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_similarity_index = index

        similarities.append(best_similarity)
        
        # Calculate the estimated location of the query in the file
        percentage_through_file = best_similarity_index / len(all_embeddings_for_document)
        min_location = percentage_through_file
        max_location = percentage_through_file + (1 / len(all_embeddings_for_document))

        min_location = round(min_location * 100.0, 0)
        max_location = round(max_location * 100.0, 0)

        if min_location == max_location:
            estimated_location.append(f'{min_location:.0f}%')
        else:
            estimated_location.append(f'{min_location:.0f}%-{max_location:.0f}%')

    # Return the results
    return similarities, estimated_location, dataset['file_path']


# Internal functions
def create_embedding_model(model_type: str, model_name: Optional[str]) -> ModelType:
    known_types = [OpenAIModel, InstructorModel]

    for known_type in known_types:
        if model_type == known_type.get_model_type():
            return known_type(model_name)
    else:
        raise ValueError(f'Unsupported model type: {model_type}')

def dataset_exists(dataset_name: str, embeddings_dir: str) -> bool:
    # Check if folder named dataset_name exists in embeddings_dir.
    return os.path.exists(os.path.join(embeddings_dir, dataset_name))

def is_supported_remote_repository(repo_url: str) -> bool:
    # Check if the given repo_url is supported by this script.
    for supported_url in supported_remote_repositories:
        if repo_url.startswith(supported_url):
            return True
    return False


## Generator functions
def generate_embeddings_for_remote_repository_archive(
        dataset_name: str,
        repo_url: str,
        embeddings_dir: str,
        embedding_model: ModelType) -> None:
    assert is_supported_remote_repository(repo_url)

    for supported_url in supported_remote_repositories:
        if repo_url.startswith(supported_url):
            download_url = repo_url + supported_remote_repositories[supported_url]
            break
    else:
        logging.error(f'Unsupported remote repository: {repo_url}')
        return

    logging.debug(f'Detected {supported_url} repository.')

    generate_embeddings_for_remote_zip_archive(
        dataset_name,
        download_url,
        embeddings_dir,
        embedding_model
    )


def generate_embeddings_for_remote_zip_archive(
        dataset_name: str,
        zip_url: str,
        embeddings_dir: str,
        embedding_model: ModelType) -> None:
    with BytesIO() as zip_buffer:
        # Download the zip file into a memory buffer, then use zipfile to retrieve the contents.
        logging.info(f'Downloading {zip_url}...')

        with urllib.request.urlopen(zip_url) as response:
            zip_buffer.write(response.read())
            zip_buffer.seek(0)

        with zipfile.ZipFile(zip_buffer) as zip_ref:
            generate_embeddings_for_zipfile(
                dataset_name,
                zip_ref,
                embeddings_dir,
                embedding_model
            )

def generate_embeddings_for_local_zip_archive(
        dataset_name: str,
        zip_path: str,
        embeddings_dir: str,
        embedding_model: ModelType) -> None:
    logging.info(f'Loading {zip_path}...')
    
    # Use zipfile to browse the contents of the zip file without extracting it.
    with zipfile.ZipFile(zip_path) as zip_ref:
        generate_embeddings_for_zipfile(
            dataset_name,
            zip_ref,
            embeddings_dir,
            embedding_model
        )

def generate_embeddings_for_zipfile(
        dataset_name: str,
        zipfile: zipfile.ZipFile,
        embeddings_dir: str,
        embedding_model: ModelType) -> None:
    logging.info(f'Generating embeddings from zipfile for {dataset_name}...')
    
    file_list = zipfile.namelist()

    # Progress bar is based on file size (as a stand-in for number of tokens encoded), not number of files.
    # Calculate total size of all files in the zip file based on their decompressed size.
    # file_size - decompressed size | compress_size - compressed size
    total_size = sum(zipinfo.file_size for zipinfo in zipfile.infolist())

    # For each file in the zip file, generate embeddings for it.
    all_embeddings = []
    bar = tqdm.tqdm(total=total_size)
    for file_path in file_list:
        bar.set_description(file_path)
        try:
            with zipfile.open(file_path, 'r') as file:
                file_contents = file.read()
                file_contents = file_contents.decode('utf-8')
                all_embeddings.append(generate_embeddings_for_contents(file_contents, embedding_model))
        except UnicodeDecodeError as e:
            logging.debug(f'Could not read as text file: {file_path}')
            all_embeddings.append([])
        #except:
        #    if verbose:
        #        logging.debug(f'Issue generating embeddings for: {file_path}')
        #    all_embeddings.append([])
        finally:
            bar.update(zipfile.getinfo(file_path).file_size)
    
    # Generate a dataset from the embeddings.
    dataset = datasets.Dataset.from_dict({
        'file_path': file_list,
        'embeddings': all_embeddings
    })

    # Save the dataset to disk.
    dataset_path = os.path.join(embeddings_dir, dataset_name)
    dataset.save_to_disk(dataset_path)

    # Create dataset metadata file.
    create_metadata_file(dataset_path, embedding_model)

    # Generate index using FAISS.
    generate_faiss_index_for_dataset(dataset, dataset_name, embeddings_dir)


def generate_embeddings_for_local_repository(
        dataset_name: str,
        repo_path: str,
        embeddings_dir: str,
        embedding_model: ModelType) -> None:
    logging.info(f'Generating embeddings from local directory {repo_path} for {dataset_name}...')

    file_paths = []
    embeddings = []

    shared_root_length = len(repo_path)
    if not repo_path.endswith('/'):
        shared_root_length += 1

    # Populate file_paths with all files in repo_path and its subdirectories.
    # While doing so, calculate the total size of all files in repo_path.
    total_size = 0
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            file_path = os.path.join(root, file)

            # Check if the file can be read as a text file
            try:
                with open(file_path, 'rt') as file:
                    _ = file.read() # TODO: Is this needed? Does the exception get generated by open or read?
            except UnicodeDecodeError as e:
                logging.debug(f'Could not read as text file: {file_path}')
                continue
            
            # When storing file_path, remove shared repo_path prefix.
            relative_file_path = file_path[shared_root_length:]
            file_paths.append(relative_file_path)

            file_size = os.path.getsize(file_path)
            total_size += file_size
    
    # Generate embeddings for each file in file_paths.
    # Base progress bar on total size of all files (as a stand-in for number of tokens encoded)
    bar = tqdm.tqdm(total=total_size)

    for file_path in file_paths:
        full_file_path = os.path.join(repo_path, file_path)
        bar.set_description(file_path)

        try:
            with open(full_file_path, 'rt') as file:
                file_contents = file.read()
                embedding = generate_embeddings_for_contents(file_contents, embedding_model)
                embeddings.append(embedding)
        #except:
        #    if verbose:
        #        print(f'WARNING: Issue generating embeddings for: {full_file_path}')
        #    embeddings.append([])
        finally:
            bar.update(os.path.getsize(full_file_path))
    
    # Generate a dataset from the embeddings.
    dataset = datasets.Dataset.from_dict({
        'file_path': file_paths,
        'embeddings': embeddings
    })

    # Save the dataset to disk.
    dataset_dir = os.path.join(embeddings_dir, dataset_name)
    dataset.save_to_disk(dataset_dir)

    # Create dataset metadata file.
    create_metadata_file(dataset_dir, embedding_model)

    # Generate index using FAISS.
    generate_faiss_index_for_dataset(dataset, dataset_name, embeddings_dir)

def generate_embeddings_for_contents(
        file_contents: str,
        embedding_model: ModelType) -> List[List[float]]:
    # Tokenize the file contents in chunks based on the model's max chunk length
    tokens = embedding_model.tokenize(file_contents)
    max_chunk_length = embedding_model.get_max_document_chunk_length()

    # Split tokens into chunks
    all_embeddings: List[List[float]] = []
    for chunk_number, i in enumerate(range(0, len(tokens), max_chunk_length)):

        chunk = tokens[i:i + max_chunk_length]
        chunk = embedding_model.detokenize(chunk)
        
        logging.debug(f'Chunk {chunk_number} token length: {min(max_chunk_length, len(tokens) - i)} | Chunk string length: {len(chunk)} | Max chunk length: {max_chunk_length}')
        all_embeddings.append(embedding_model.generate_embedding_for_document(chunk))
    
    return all_embeddings

def generate_faiss_index_for_dataset(
        dataset: datasets.Dataset,
        dataset_name: str,
        embeddings_dir: str) -> None:
    pass

def create_metadata_file(
        dataset_dir: str,
        embedding_model: ModelType) -> None:
    metadata_file_path = os.path.join(dataset_dir, 'metadata.json')
    metadata = {
        'model_type': embedding_model.get_model_type(),
        'model_name': embedding_model.get_model_name(),
    }

    with open(metadata_file_path, 'w') as metadata_file:
        json.dump(metadata, metadata_file)

def cosine_similarity(a: List[float], b: List[float]) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
"""

This module contains the ConversationScanner class which is used to scan a conversation export from ChatGPT.
It also incorporates methods to create a local index of all documents and upsert it to Pinecone.

"""
import json
import logging
import re
import uuid
import itertools
import os
from datetime import datetime

import pinecone
import openai
import tiktoken
import nltk

class MajorTypeError(BaseException):
    """
    A custom exception class for raising errors when the basic type of a value 
    is not as expected. This is typically used for validating nested data structures 
    when the data type at the first layer does not meet expectations.

    Usage
    -----
    >>> data = {'key': 'value'}
    >>> expected_type = list
    >>> if not isinstance(data, expected_type):
    >>>     raise MajorTypeError(f"Expected type {expected_type}, but got {type(data)}")

    Attributes
    ----------
    message : str
        A human-readable string describing the exception.

    Methods
    -------
    __init__(self, message: str)
        Initializes a new instance of MajorTypeError.

    Raises
    ------
    MajorTypeError
        If the basic type of a value does not meet expectations during data validation.
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class ConversationScanner:
    """
    A class used primarily to scan a conversation export from ChatGPT.

    This class is used to scan a conversation export from ChatGPT and create a local index of all documents. It also
    incorporates methods to upsert the global index to Pinecone. 

    Usage
    -----
    There are two ways to use this class. The first is to import a precomputed local index and use it to upsert its data to the Pinecone globla index.
    The second is to use the class to create a local index from an OpenAI ChatGPT conversations export and then upsert it to the Pinecone global index.

    The first method is used as follows:
    >>> from memgpt import ConversationScanner
    >>> scanner = ConversationScanner(user_id, openai_api_key, pinecone_api_key, pinecone_index_name, file_path, debug)
    >>> scanner.load_local_index(file_path)
    >>> scanner.upsert_index()

    The second method is used as follows:
    >>> from memgpt import ConversationScanner
    >>> scanner = ConversationScanner(user_id, openai_api_key, pinecone_api_key, pinecone_index_name, file_path, debug)
    >>> scanner.load_data(file_path)
    >>> scanner.compute_local_index()
    >>> scanner.upsert_index()

    [!] It is recommended to frequently check the status of the Scanner using `CovnersationScanner.status()` to ensure that the Scanner is producing expected results.
    
    [!] If there are any errors, it is recommended to rerun with `debug=True` to get more information about the error in the debug.log file.

    [!] To save on API costs when running often with the same index, it is recommended to use `ConversationScanner.save_local_index()` to save the local index to a file instead of recomputing it every time.

    Parameters
    ----------

    user_id : str
        The ID of the user. Used to identify the namespace where the documents should be stored.

    openai_api_key : str
        The API key for OpenAI.

    pinecone_api_key : str
        The API key for Pinecone.

    pinecone_index_name : str
        The name of the Pinecone index.

    pinecone_environment : str, optional
        The Pinecone environment to use. If not provided, defaults to 'asia-southeast1-gcp-free'.

    file_path : str, optional
        The file path of the data to be associated with the current instance. If not provided, defaults to an empty string.

    debug : bool, optional
        A boolean value indicating whether the application is in debug mode. If not provided, defaults to False.
        In debug mode, the application logs various processes to a file.

    Attributes
    ----------

    data: list
        A list of all the conversation data. Needs to be loaded using the 'load_data' method.

    index : Pinecone Index
        The Pinecone index associated with the instance.

    local_index : dict
        A local index of all documents. Needs to be created using the 'compute_local_index' method.

    Methods
    -------

    set_file_path(file_path)
        Sets the file path for the current instance.

    load_data(file_path="")
        Loads data from a file into the 'data' attribute of the class instance. If no file path is specified, it attempts to use the
        'file_path' attribute of the instance.

    compute_local_index()
        Computes a local index of all documents and stores it in the 'local_index' attribute of the instance.

    upsert_index()
        Upserts the local index to the Pinecone index.

    save_local_index(file_path="")
        Saves the local index to a file. If no file path is specified, it generates a new file path of format "local_index_{user_id}_{time.time()}.json".

    load_local_index(file_path)
        Loads the local index from a file.

    status()
        Prints the status of the current instance to the console.

    validate_precomputed_embedding()
        Validates the `local_index` attribute of the current instance.

    validate_openai_conversation_export()
        Validates the `data` attribute of the current instance to be a filtered(!) openai conversation export. 
        It will only return true after a `compute_local_index()` call since it is only place where `__filter_openai_conversation_export()` is called.

    Data Structures
    ---------------

    openai_conversation_export : list
        A list of dictionaries, each representing a conversation. Each dictionary needs to have at least the following structure:
        {\n
            "title": str,\n
            "mapping": {\n
                "id": {\n
                    "messages": {\n
                        "id": str,\n
                        "author": {\n
                            "role": str,\n
                            ...\n
                        }\n
                        "content": {\n
                            "parts": list<str>,\n
                            ...\n
                        }\n
                        ...\n
                    }\n
                    ...\n
                }\n
                ...\n
            }\n
            ...\n
        }

    precomputed_embedding: dict
        A dictionary containing the precomputed embeddings of all documents. The keys are the documents content and the values are the relevant data. Each follow this structure:
        {\n
            "vectors": {\n
                "id": str,\n
                "metadata": {\n
                    "role": str,\n
                    "content": str,\n
                },\n
                values: list<float>\n
            }\n
            "namespace": str\n
        }


    """

    def __init__(self, user_id, openai_api_key, pinecone_api_key, pinecone_index_name, data_file_path="", pinecone_environment="asia-southeast1-gcp-free", debug=False):
        """
        Initializes the class instance with given parameters.

        This constructor sets up the initial attributes for an instance of the class, including various keys for OpenAI 
        and Pinecone, a file path, and debug mode.

        Parameters
        ----------
        user_id : str
            The ID of the user. 

        openai_api_key : str
            The API key for OpenAI. 

        pinecone_api_key : str
            The API key for Pinecone.

        pinecone_index_name : str
            The name of the Pinecone index.

        data_file_path : str, optional
            The file path to the data which is to be associated with the current instance. If not provided, defaults to an empty string.

        pinecone_environment : str, optional
            The Pinecone environment to use. If not provided, defaults to 'asia-southeast1-gcp-free'.

        debug : bool, optional
            A boolean value indicating whether the application is in debug mode. If not provided, defaults to False.

        Attributes
        ----------
        data : list
            An empty list that can be used to store data related to the instance.

        index : Pinecone Index
            A Pinecone index associated with the instance.

        local_index : dict
            An empty dictionary to be used as a local index.

        Returns
        -------
        Class instance
        """
        self.user_id = user_id
        self.openai_api_key = openai_api_key
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_index_name = pinecone_index_name
        self.file_path = data_file_path
        self.pinecone_environment = pinecone_environment
        self.data = []
        self.debug = debug
        if self.debug:
            logging.basicConfig(
                filename=f'./debug/debugging_scanner_{datetime.now().strftime("%Y%m%d-%H%M%S")}.log', level=logging.DEBUG)

        openai.api_key = self.openai_api_key
        pinecone.init(api_key=self.pinecone_api_key,
                      environment=self.pinecone_environment)
        self.index = pinecone.Index(self.pinecone_index_name)
        self.local_index = {}

    def set_file_path(self, file_path):
        """
        Sets the file path for the current instance.

        This function is used to set the file path associated with the current instance of the class. The provided file path 
        is stored in the 'file_path' attribute of the instance.

        Parameters
        ----------
        file_path : str
            The file path to be associated with the current instance. This should be a valid path string.

        Returns
        -------
        None
        """
        self.file_path = file_path

    def load_data(self, file_path=""):
        """
        Loads data from a file into the 'data' attribute of the class instance.

        This function attempts to open a file at the specified path, reads all lines of the file, and loads the data into 
        the 'data' attribute of the instance. If the file path is not provided as a parameter, it attempts to use the 
        'file_path' attribute of the instance. If no file path is specified, it raises a ValueError. If the file is not found,
        it raises a FileNotFoundError, and if the file cannot be read, it raises a PermissionError.

        Parameters
        ----------
        file_path : str, optional
            The path of the file to load data from. If not provided, uses the 'file_path' attribute of the instance.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If no file path is specified.
        FileNotFoundError
            If the specified file is not found.
        PermissionError
            If the file cannot be read.
        """
        if file_path != "":
            self.file_path = file_path

        if self.file_path == "":
            raise ValueError("File path is empty! Set a file path first.")
        elif not os.access(self.file_path, os.R_OK):
            raise PermissionError("File is not readable!")
        elif not os.path.isfile(self.file_path):
            raise FileNotFoundError("File not found! Make sure the given filepath exists")
        else:
            with open(self.file_path, "r", encoding="utf-8") as f:
                try:
                    self.data = json.load(f)
                except json.JSONDecodeError as e:
                    print("Invalid JSON format.")
                    raise e

    def __filter_openai_conversation_export(self, data):
        """
        Filters the OpenAI conversation data based on required keys.

        This private method iterates through the provided data, validating each item and its nested structure based 
        on the required keys. The method ensures that the conversation data has the necessary structure and only retains 
        valid mappings in the data.

        Parameters
        ----------
        data : list
            A list of dictionaries representing the conversation data.

        Returns
        -------
        list
            The input data list after removing invalid mappings.

        Note
        ----
        This function also performs logging if the class instance's 'debug' attribute is True.
        """

        required_keys = set(['title', 'mapping', 'id', 'current_node'])
        required_mapping_keys = set(['id', 'message'])
        required_message_keys = set(['author', 'content'])
        required_author_keys = set(['role'])
        required_content_keys = set(['parts'])

        removed_count = 0

        def validate_mapping(mapping_item):
            nonlocal removed_count
            if not isinstance(mapping_item, dict):
                return False, "Mapping item is not a dictionary."

            if not required_mapping_keys.issubset(set(mapping_item.keys())):
                return False, "Mapping item does not contain all required keys."

            if not isinstance(mapping_item['message'], dict):
                return False, "Message in mapping item is not a dictionary."

            if not required_message_keys.issubset(set(mapping_item['message'].keys())):
                return False, "Message does not contain all required keys."

            author = mapping_item['message']['author']
            content = mapping_item['message']['content']

            if not isinstance(author, dict) or not required_author_keys.issubset(set(author.keys())):
                return False, "Author does not contain all required keys."

            if not isinstance(content, dict) or not required_content_keys.issubset(set(content.keys())):
                return False, "Content does not contain all required keys."

            if not isinstance(content['parts'], list):
                return False, "Content parts is not a list."

            if not all(isinstance(part, str) for part in content['parts']):
                return False, "Content parts is not a list of strings."

            return True, ""

        for item in data:
            if not isinstance(item, dict) or not required_keys.issubset(set(item.keys())):
                continue  # Skip the item if it doesn't meet the basic requirements

            if not isinstance(item['mapping'], dict):
                continue  # Skip the item if 'mapping' is not a dict

            # Make a copy for debugging
            original_mapping = dict(item['mapping'])
            valid_mappings = {}
            for k, v in item['mapping'].items():
                is_valid, reason = validate_mapping(v)
                if is_valid:
                    valid_mappings[k] = v
                elif self.debug:  # log invalid mapping and the reason
                    logging.debug("Removed mapping %s: %s, Reason: %s",
                                  k, original_mapping[k], reason)
                    removed_count += 1
            item['mapping'] = valid_mappings

        if self.debug:
            logging.debug("Total mappings removed: %s", removed_count)
        return data

    def __is_openai_conversation_export(self, data):
        """
        Validates whether the provided data conforms to the expected structure of an OpenAI conversation export.

        This private method iterates over the provided data, checking each item and its nested structure based on 
        the required keys. It raises ValueError if an item does not match the expected structure or misses required keys.
        If all items are validated successfully without raising an error, the function returns True, indicating the data 
        is a valid OpenAI conversation export.

        Parameters
        ----------
        data : list
            A list of dictionaries representing the conversation data.

        Returns
        -------
        bool
            True if the data is valid, False otherwise.

        Raises
        ------
        ValueError
            If an item in the data does not conform to the expected structure or lacks required keys.

        TypeError
            If the data contains inappropriate values.

        MajorTypeError
            If the first layer of the data is not a list.
        """

        if not isinstance(data, list):
            raise MajorTypeError(f"Data {data} is not a list")

        required_keys = set(['title', 'mapping', 'id', 'current_node'])
        required_mapping_keys = set(['id', 'message'])
        required_message_keys = set(['author', 'content'])
        required_author_keys = set(['role'])
        required_content_keys = set(['parts'])

        def is_openai_conversation(item):
            if not isinstance(item, dict):
                raise MajorTypeError(f"Item {item} is not a dict")

            if not required_keys.issubset(set(item.keys())):
                missing_keys = required_keys - set(item.keys())
                raise ValueError(f"Expected keys {missing_keys} but not found in item {item}")

            if not isinstance(item['mapping'], dict):
                raise TypeError(f"Mapping {item['mapping']} is not a dict")

            for mapping_item in item['mapping'].values():
                if not isinstance(mapping_item, dict):
                    raise TypeError(f"Mapping item {mapping_item} is not a dict")

                if not required_mapping_keys.issubset(set(mapping_item.keys())):
                    missing_keys = required_mapping_keys - set(mapping_item.keys())
                    raise ValueError(f"Expected keys {missing_keys} but not found in mapping item {mapping_item}")

                if not isinstance(mapping_item['message'], dict):
                    raise TypeError(f"Message {mapping_item['message']} in mapping item {mapping_item} is not a dict")

                if not required_message_keys.issubset(set(mapping_item['message'].keys())):
                    missing_keys = required_message_keys - set(mapping_item['message'].keys())
                    raise ValueError(f"Expected keys {missing_keys} but not found in message {mapping_item['message']}")

                author = mapping_item['message']['author']
                content = mapping_item['message']['content']

                if not isinstance(author, dict) or not required_author_keys.issubset(set(author.keys())):
                    missing_keys = required_author_keys - set(author.keys())
                    raise ValueError(f"Expected keys {missing_keys} but not found in author {author}")

                if not isinstance(content, dict) or not required_content_keys.issubset(set(content.keys())):
                    missing_keys = required_content_keys - set(content.keys())
                    raise TypeError(f"Expected keys {missing_keys} but not found in content {content}")

                if not isinstance(content['parts'], list):
                    raise TypeError(f"Content parts {content['parts']} is not a list")

                if not all(isinstance(part, str) for part in content['parts']):
                    raise TypeError(f"Content parts {content['parts']} is not a list of strings")

        if not any([is_openai_conversation(item) for item in data]):
            return False

        return True

    def __is_precomputed_embedding(self, data):
        """
        Validates whether the provided data matches the expected structure of a precomputed embedding index.

        This private method checks the structure and values of the data based on the required keys. 
        It raises a ValueError if the data does not match the expected structure, lacks required keys, 
        or contains inappropriate values. If the data is validated successfully without raising an error, 
        the function returns True, indicating that the data is a valid precomputed embedding index.

        Parameters
        ----------
        data : dict
            A dictionary representing the potential precomputed embedding data.

        Returns
        -------
        bool
            True if the data is a valid precomputed embedding, False otherwise.

        Raises
        ------
        ValueError
            If the data does not conform to the expected structure or lacks required keys.

        TypeError
            If the data contains inappropriate values.

        MajorTypeError
            If the data is not a dict.
        """

        def is_precomputed_embeddings_item(item):
            required_keys = ['id', 'metadata', 'values']
            metadata_keys = ['role', 'content']

            if not isinstance(item, dict):
                raise MajorTypeError(f"Data {str(item)[:20]} is not a dict")

            if item == {}:
                raise ValueError(f"Data {item} is empty")

            if not isinstance(item, dict):
                raise TypeError(f"Item {item} is not a dict")

            if set(item.keys()) != set(required_keys):
                raise ValueError(f"Expected keys {required_keys} but found {list(item.keys())} in item {item}")

            try:
                metadata = item['metadata']

                if set(metadata.keys()) != set(metadata_keys):
                    raise ValueError(f"Expected keys {metadata_keys} but found {list(metadata.keys())} in metadata {metadata}")

                elif not isinstance(metadata['role'], str) or \
                   not isinstance(metadata['content'], str):
                    raise TypeError(f"Expected metadata values to be strings but found {metadata}")

                values = item['values']
                if not isinstance(values, list):
                    raise TypeError(f"Expected values to be a list but found {values}")

                for val in values:
                    if not isinstance(val, (int, float)):  # considering embeddings could be float too
                        raise TypeError(f"Expected values to be a list of numbers but found {values}")

            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid structure or value in item {item}") from e

            return True

        required_keys = ['vectors', 'namespace']

        if not isinstance(data, dict):
            raise MajorTypeError(f"Data {str(data)[:10]} is not a dict")

        if data == {}:
            raise ValueError(f"Data {data} is empty")

        if set(data.keys()) != set(required_keys):
            raise ValueError(f"Expected keys {required_keys} but found {list(data.keys())} in data {data}")

        for item in data['vectors']:
            is_precomputed_embeddings_item(item)

        return True

    def __extract_documents_from_conversations(self, conversations):
        """
        Extracts documents from a list of conversations.

        This private method iterates through a list of conversations, each represented as a dictionary, 
        and extracts messages from the 'mapping' field of each conversation. Each message is then 
        tokenized into sentences, cleaned up, and combined back into a single document. The role of 
        the message author and the document are then added as a dictionary to a list of documents.

        Parameters
        ----------
        conversations : list
            A list of dictionaries where each dictionary represents a conversation and has a 'mapping' field
            containing another dictionary of messages.

        Returns
        -------
        list
            A list of dictionaries where each dictionary represents a document and contains the 'role' of 
            the message author and the 'content' of the document.

        Note
        ----
        Each document in the returned list is a dictionary with the following format:\n
        {\n
            'role': <str, role of the message author>,\n
            'content': <str, document text content>,\n
        }
        """
        documents = []

        for conversation in conversations:
            mappings = conversation.get('mapping', {})

            for message in mappings.values():
                content = message.get('message', {}).get('content', {})
                parts_content = content.get('parts', [''])

                if parts_content[0]:
                    parts = nltk.sent_tokenize(parts_content[0])
                    parts = [re.sub(r'\s+', ' ', part).strip() for part in parts]
                    document = ' '.join(parts)

                    documents.append({'role': message['message']['author']['role'], 'content': document})

        return documents

    def __split_documents(self, documents):
        """
        Splits documents into smaller chunks if they exceed the maximum token limit.

        This private method iterates over a list of documents, each represented as a dictionary, 
        and splits the content into smaller parts if the total number of tokens exceeds 8191, 
        which is the maximum number of tokens allowed for certain models.

        If the document does not exceed the maximum token limit, it is added to a new list as is. 
        If the document does exceed the limit, it is split into two parts and each part is recursively split until all parts 
        are below the maximum token limit. All parts are then added to the new list as separate documents.

        Parameters
        ----------
        documents : list
            A list of dictionaries where each dictionary represents a document and contains the 'role' of 
            the message author and the 'content' of the document.

        Returns
        -------
        list
            A list of dictionaries where each dictionary represents a document and contains the 'role' of 
            the message author and the 'content' of the document.

        Raises
        ------
        Exception
            If there is an error when processing a document.

        Note
        ----
        Each document in the input and returned list is a dictionary with the following format:
        {
            'role': <str, role of the message author>,
            'content': <str, document text content>,
        }
        """
        enc = tiktoken.encoding_for_model("text-embedding-ada-002")

        new_documents = []
        for idx, document in enumerate(documents):
            try:
                tokens = enc.encode(document['content'])
                if len(tokens) > 1000 and self.debug:
                    logging.debug('%s has %s tokens', idx, len(tokens))
                if len(tokens) > 8191:
                    content = document['content']
                    part_1 = []
                    part_2 = []
                    token_count = 0
                    
                    for word in content.split():
                        if token_count + len(enc.encode(word)) < 8191:
                            part_1.append(word)
                            token_count += len(enc.encode(word))
                        else:
                            part_2.append(word)

                    new_documents.extend(self.__split_documents([{'role': document['role'], 'content': ' '.join(part_1)}]))
                    new_documents.extend(self.__split_documents([{'role': document['role'], 'content': ' '.join(part_2)}]))
                else:
                    new_documents.append(document)
            except Exception as e:
                if self.debug:
                    logging.error('Error with document: %s, Exception: %s', document, e)
                raise e

        if self.debug:
            len_before = len(documents)
            len_after = len(new_documents)
            logging.debug("Split %s documents into %s documents", len_before, len_after)

            len_before = len(new_documents)
            new_documents = [dict(t) for t in {tuple(d.items()) for d in new_documents}]
            len_after = len(new_documents)

            logging.debug("Removed %s duplicate documents", len_before - len_after)
        return new_documents

    def compute_local_index(self):
        """
        Computes the local index for data loaded from the file at the file path.

        This method loads data from the given file path, checks if it's already a precomputed embedding. 
        If it is, the method simply sets the local index to this data and returns.

        If the data is not a precomputed embedding, the method checks if it's a valid OpenAI conversation export.
        It then filters this data, extracts documents from the conversations, and splits any documents 
        that exceed the maximum token limit.

        Each split document is then processed to compute its embedding and an ID is generated for it. 
        These are stored in a local index where the key is the content of the document and the value is a dictionary 
        with the ID, metadata, and the embedding of the document.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the file path is not set or if the data structure is invalid.

        Notes
        ------
        Each entry in the `self.local_index` dictionary has the format:
        {\n
        '<content of the document>': {\n
          'id': '<UUID generated for the document>',\n
          'metadata': {\n
              'role': '<role of the message author>',\n
              'content': '<content of the document>'\n
          },\n
          'values': '<computed embedding for the document>'\n
        },
        """

        if self.file_path is None:
            raise ValueError("File path is not set")

        if len(self.data) == 0:
            self.load_data(self.file_path)
            return

        # skip if already computed
        try:
            self.__is_precomputed_embedding(self.data)
            self.local_index = self.data
        except (ValueError, TypeError, MajorTypeError):
            pass

        # validate structure of data
        try:
            self.__is_openai_conversation_export(self.data)
        except (ValueError, MajorTypeError) as e:
            raise ValueError("Invalid data structure") from e
        except TypeError:
            pass

        # filter conversations
        data_temp = self.__filter_openai_conversation_export(self.data)
        self.data = data_temp

        # extract documents from conversations
        documents = self.__extract_documents_from_conversations(self.data)

        # split documents (circumvents ada-002's 8192 token limit and removes duplicates)
        documents = self.__split_documents(documents)

        # compute embeddings and create local inverted index
        embeddings_index = {'vectors': [], 'namespace': self.user_id}
        enc = tiktoken.encoding_for_model("text-embedding-ada-002")

        for idx, document in enumerate(documents):
            if self.debug:
                logging.info('Processing %s of %s (tokens: %s)', idx, len(documents), len(enc.encode(document["content"])))
            embedding = openai.Embedding.create(input=document['content'], model="text-embedding-ada-002")["data"][0]["embedding"] # type: ignore
            embeddings_index['vectors'].append({'id': str(uuid.uuid4()), 'metadata': document, 'values': embedding})

        self.local_index = embeddings_index

    def upsert_local_index(self):
        """
        Upserts the local index to the database.

        This method splits the local index into chunks, then upserts each chunk into the database.
        It raises an error if the local index hasn't been computed or if the database hasn't been set.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the local index hasn't been computed or if the database hasn't been set.
        """
        def chunks(iterable, batch_size=100):
            iterator = iter(enumerate(iterable))
            chunk = tuple(itertools.islice(iterator, batch_size))

            while chunk:
                first_index = chunk[0][0]  # First element's index
                actual_last_index = chunk[-1][0]  # Last element's index

                # We only want the elements, not their indices
                elements_only = [item[1] for item in chunk]

                yield elements_only, first_index, actual_last_index

                chunk = tuple(itertools.islice(iterator, batch_size))



        if not self.__is_precomputed_embedding(self.local_index):
            raise ValueError("Local index is not computed")

        if self.index is None:
            raise ValueError("Database is not set")

        for chunk in chunks(self.local_index['vectors']):
            if self.debug:
                logging.info('Upserting %s to %s | %s', chunk[1], chunk[2], chunk[0][0])
            self.index.upsert(chunk[0], namespace=self.local_index['namespace'])



    def save_local_index(self, index_file_path=""):
        """
        Saves the local index to a json file.

        This method saves the local index to a file at the given file path. If no file path is provided,
        the method generates a new file path using the current timestamp.

        Parameters
        ----------
        file_path : str, optional

        Returns
        -------
        None
        """
        # generate a new file path using the current timestamp if no file path is provided
        if len(index_file_path) == 0:
            index_file_path = f"local_index_{self.user_id}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"

        # save the local index to the provided file path
        with open(index_file_path, 'w', encoding="utf-8") as f:
            json.dump(self.local_index, f)

    def load_local_index(self, index_file_path):
        """
        Loads the local index from a json file.

        This method loads the local index from a file at the given file path.

        Parameters
        ----------
        file_path : str

        Returns
        -------
        None
        """
        with open(index_file_path, 'r', encoding="utf-8") as f:
            self.local_index = json.load(f)

        self.__is_precomputed_embedding(self.local_index)

    def status(self):
        """
        
        Show status of connectivity to services and data fields.

        Prints the values of the following fields:
        - `self.file_path`
        - `self.data` as a tuple of the form: (`self.__is_openai_conversation_export(self.data)`, `str(self.data)[:10]`)
        - `self.local_index` as a tuple of the form: (`self.__is_precomputed_embedding(self.local_index)`, `str(self.local_index)[:10]`)

        - `self.index`
        - `self.user_id`
        - `self.debug`

        Returns
        -------
        None
        """

        print(f"file_path: {self.file_path}")
        try:
            print(f"data: {self.__is_openai_conversation_export(self.data), str(self.data)[:10]}...")
        except (ValueError, TypeError, MajorTypeError):
            print(f"data: {(False, str(self.data)[:20])}...")

        try:
            print(f"local_index: {self.__is_precomputed_embedding(self.local_index), str(self.local_index)[:10]}")
        except (ValueError, TypeError, MajorTypeError):
            print(f"local_index: {(False, str(self.local_index)[:20])}...")
        print(f"index: {self.index}")
        print(f"user_id: {self.user_id}")
        print(f"debug: {self.debug}")

    def validate_precomputed_embedding(self):
        """
        Validates the local_index attribute as a precomputed embeddings index.

        This method checks if the loaded local index is a valid precomputed embedding.
            -> local_index needs to be set before calling this method. Use `load_local_index` to load data from a file or `compute_local_index` to compute a new index from `self.data`.

        Returns
        -------
        bool
            True if the local index is a valid precomputed embedding, False otherwise.

        Raises
        ------

        AssertionError
            If the local index is not a valid precomputed embedding.
        """
        try:
            self.__is_precomputed_embedding(self.local_index)
            return True
        except (ValueError, TypeError, MajorTypeError) as e:
            raise AssertionError("Invalid precomputed embedding") from e

    def validate_openai_conversation_export(self):
        """
        Validates the data attribute as an OpenAI conversation export.

        This method checks if the loaded data is a valid OpenAI conversation export.
            -> data needs to be set before calling this method. Use `load_data` to load data from a file.

        Returns
        -------
        bool
            True if the data is a valid OpenAI conversation export, False if the structure has invalid datatypes in required keys.

        Raises
        ------
        AssertionError
            If the data is not a valid OpenAI conversation export.
        """
        try:
            self.__is_openai_conversation_export(self.data)
            return True
        except (ValueError, MajorTypeError) as e:
            raise AssertionError("Invalid OpenAI conversation export") from e
        except TypeError:
            return False

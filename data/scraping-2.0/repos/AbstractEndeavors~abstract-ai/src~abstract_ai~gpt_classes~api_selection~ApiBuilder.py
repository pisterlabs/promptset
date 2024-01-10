import openai
from abstract_security.envy_it import get_env_value
class ApiManager:
    """
    This class handles the management of OpenAI API keys and headers.
    It is capable of retrieving the API key from environment variables, loading it for use in requests,
    and constructing authorization headers for the API calls.

    Attributes:
        content_type (str): The type of the content being sent in the request. Defaults to 'application/json'.
        api_env (str): The name of the environment variable containing the API key. Defaults to 'OPENAI_API_KEY'.
        api_key (str): The API key for OpenAI. If not provided, it is retrieved using the `get_openai_key` method.
        header (dict): The headers to be used in the requests. If not provided, they are generated using the 'get_header' method.

    Methods:
        get_openai_key(): Retrieves OpenAI API key from environment variables.
        load_openai_key(): Loads the OpenAI API key for use in requests.
        get_header(): Generates the request headers for the API calls.
    """
    def __init__(self,content_type=None,header=None,api_env:str=None,api_key:str=None)->None:
        self.content_type=content_type or 'application/json'
        self.api_env=api_env or 'OPENAI_API_KEY'
        self.api_key=api_key or self.get_openai_key()
        self.header=header or self.get_header()
        self.load_openai_key()
    def get_openai_key(self)->str:
        """
        Retrieves the OpenAI API key from the environment variables.

        Args:
            key (str): The name of the environment variable containing the API key. 
                Defaults to 'OPENAI_API_KEY'.

        Returns:
            str: The OpenAI API key.
        """
        return get_env_value(key=self.api_env)
    def load_openai_key(self)->None:
        """
        Loads the OpenAI API key for authentication.
        """
        openai.api_key = self.api_key
    def get_header(self)->dict:
        """
        Generates request headers for API call.
        
        Args:
            content_type (str): Type of the content being sent in the request. Default is 'application/json'.
            api_key (str): The API key for authorization. By default, it retrieves the OpenAI API key.
            
        Returns:
            dict: Dictionary containing the 'Content-Type' and 'Authorization' headers.
        """
        return {'Content-Type': self.content_type, 'Authorization': f'Bearer {self.api_key}'}
    

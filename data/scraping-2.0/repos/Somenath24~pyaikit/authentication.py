import openai
class Authenticator:
    def __init__(self):
        self.api_key = None

    def setup(self, api_key,org_id):
        """Set up the authentication with the provided API key."""
        self.api_key = api_key
        # open AI setup logic
        openai.organization = org_id
        openai.api_key = api_key
        return(openai)

    def authenticate_request(self, request):
        """Authenticate the request by adding the API key."""
        if self.api_key:
            request.headers['Authorization'] = f'Bearer {self.api_key}'
        else:
            raise ValueError('API key is not set. Please call setup(api_key) to set the API key.')

    # Additional authentication-related methods can be added here
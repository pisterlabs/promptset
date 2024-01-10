from OpenAIAuth import Auth0


def get_access_token(email, password):
    auth = Auth0(email_address=email, password=password)
    access_token = auth.get_access_token()
    print("access:", access_token)
    return access_token
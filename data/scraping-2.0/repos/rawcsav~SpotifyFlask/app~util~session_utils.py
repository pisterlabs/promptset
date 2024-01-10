import os
import secrets
import string
from datetime import timezone, timedelta, datetime
import openai
import requests
import sshtunnel
from cryptography.fernet import Fernet
from flask import abort, session, current_app


def verify_session(session):
    if "tokens" not in session:
        abort(400)
    return session["tokens"].get("access_token")


def fetch_user_data(access_token):
    headers = {"Authorization": f"Bearer {access_token}"}
    res = requests.get(current_app.config['ME_URL'], headers=headers)
    if res.status_code != 200:
        abort(res.status_code)

    return res.json()


def generate_state():
    return "".join(
        secrets.choice(string.ascii_uppercase + string.digits) for _ in range(16)
    )


def prepare_auth_payload(state, scope, show_dialog=False):
    payload = {
        "client_id": current_app.config['CLIENT_ID'],
        "response_type": "code",
        "redirect_uri": current_app.config['REDIRECT_URI'],
        "state": state,
        "scope": scope,
    }
    if show_dialog:
        payload["show_dialog"] = True
    return payload


def request_tokens(payload, client_id, client_secret):
    res = requests.post(current_app.config['TOKEN_URL'], auth=(client_id, client_secret), data=payload)
    res_data = res.json()
    if res_data.get("error") or res.status_code != 200:
        return None, res.status_code
    return res_data, None


def convert_utc_to_est(utc_time):
    return utc_time.replace(tzinfo=timezone.utc).astimezone(timezone(timedelta(hours=-4)))


def load_encryption_key():
    return os.environ['CRYPT_KEY'].encode()


def encrypt_data(api_key):
    cipher_suite = Fernet(load_encryption_key())
    encrypted_api_key = cipher_suite.encrypt(api_key.encode())
    return encrypted_api_key


def decrypt_data(encrypted_api_key):
    cipher_suite = Fernet(load_encryption_key())
    decrypted_api_key = cipher_suite.decrypt(
        encrypted_api_key)
    return decrypted_api_key.decode()


def is_api_key_valid(key):
    openai.api_key = key
    try:
        test = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ],
            max_tokens=10,
            temperature=0,
        )
        if test.choices[0].message.content:
            return True
    except openai.OpenAIError:
        return False


def refresh_tokens():
    if 'tokens' not in session:
        return False

    payload = {
        'grant_type': 'refresh_token',
        'refresh_token': session['tokens'].get('refresh_token')
    }

    res_data, error = request_tokens(payload, current_app.config['CLIENT_ID'], current_app.config['CLIENT_SECRET'])
    if error:
        return False

    new_access_token = res_data.get('access_token')
    new_refresh_token = res_data.get('refresh_token', session['tokens']['refresh_token'])
    expires_in = res_data.get('expires_in')
    new_expiry_time = datetime.now() + timedelta(seconds=expires_in)

    session['tokens'].update({
        'access_token': new_access_token,
        'refresh_token': new_refresh_token,
        'expiry_time': new_expiry_time.isoformat()
    })

    return True


def get_tunnel(SSH_HOST, SSH_USER, SSH_PASS, SQL_HOSTNAME, max_attempts=3):
    attempt_count = 0
    sshtunnel.SSH_TIMEOUT = 5.0
    sshtunnel.TUNNEL_TIMEOUT = 5.0
    while attempt_count < max_attempts:
        try:
            tunnel = sshtunnel.SSHTunnelForwarder(
                (SSH_HOST),
                ssh_username=SSH_USER,
                ssh_password=SSH_PASS,
                remote_bind_address=(SQL_HOSTNAME, 3306)
            )
            tunnel.start()
            return tunnel
        except sshtunnel.BaseSSHTunnelForwarderError:
            attempt_count += 1
            if attempt_count == max_attempts:
                raise

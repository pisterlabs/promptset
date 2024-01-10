import os
import sys
import csv
import openai


DOMAINS = {
    'default',
    'interactiveqa',
}


def read_api_keys(path):
    api_keys = dict()
    with open(path) as f:
        rows = csv.DictReader(f)
        for row in rows:
            domain = row['domain']
            host = row['host']
            if domain not in DOMAINS:
                print(f'Update frontend first to use specific API key for this domain: {domain}')
            api_keys[(domain, host)] = row['key']
    return api_keys


def get_openai_default_api_key(api_keys):
    if ('openai', 'default') not in api_keys:
        raise RuntimeError('No default key provided for OpenAI API')
    key = api_keys[('openai', 'default')]
    return key


def get_crfm_default_api_key(api_keys):
    if ('crfm', 'default') not in api_keys:
        raise RuntimeError('No default key provided for CRFM API')
    key = api_keys[('crfm', 'default')]
    return key


def set_openai_api_key(key):
    openai.api_key = key
    print('Using OpenAI key:', key)


def set_crfm_api_key(key):
    from benchmarking.src.common.authentication import Authentication
    auth = Authentication(api_key=key)
    print('Using CRFM key:', key)
    return auth


def setup_crfm(path, api_keys):
    if not os.path.exists(path):
        raise RuntimeError(f'Run source {path}/venv/bin/activate')

    sys.path.append(path)
    from benchmarking.src.common.authentication import Authentication
    from benchmarking.src.proxy.accounts import Account
    from benchmarking.src.proxy.remote_service import RemoteService

    key = api_keys[('crfm', 'default')]

    auth = Authentication(api_key=key)
    service = RemoteService("https://crfm-models.stanford.edu")
    account: Account = service.get_account(auth)
    print("Current token usage: " + str(account.usages['gpt3']['monthly']))
    return auth, service


def set_default_api_keys(api_keys):
    is_openai_set, is_crfm_set = False, False
    try:
        key = get_openai_default_api_key(api_keys)
        set_openai_api_key(key)
        is_openai_set = True
    except:
        pass
    try:
        key = get_crfm_default_api_key(api_keys)
        set_crfm_api_key(key)
        is_crfm_set = True
    except:
        pass
    if is_openai_set or is_crfm_set:
        pass
    else:
        raise RuntimeError('Neither OpenAI or CRFM API default key is set')


def set_openai_api_key_for_domain(api_keys, domain):
    if ('openai', domain) not in api_keys:
        print('No specific key for domain:', domain)
        key = get_openai_default_api_key(api_keys)
    else:
        key = api_keys[('openai', domain)]
    set_openai_api_key(key)


def set_crfm_api_key_for_domain(api_keys, domain):
    if ('crfm', domain) not in api_keys:
        print('No specific key for domain:', domain)
        key = get_crfm_default_api_key(api_keys)
    else:
        key = api_keys[('crfm', domain)]
    set_crfm_api_key(key)

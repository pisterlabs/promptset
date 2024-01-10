import json
import requests
import re
import time
from copy import copy
from os import environ
import base64
import openai
import uuid
import datetime
import re
from datetime import datetime as dt
from urllib.parse import urlparse, parse_qs
from datetime import datetime, timedelta

import requests
from certifi import where
from concurrent.futures import ThreadPoolExecutor
from functools import reduce



HOST = "https://ai.fakeopen.com"
SHARE_TOKEN_URI = "/token/register"
ACESS_TOKEN_URI = "/auth/login"
POOL_TOKEN_URI = "/pool/update"

def default_api_prefix():
    return 'https://ai-{}.fakeopen.com'.format((datetime.now() - timedelta(days=1)).strftime('%Y%m%d'))

class Auth0:
    def __init__(self, email: str, password: str, proxy: str = None, use_cache: bool = True, mfa: str = None):
        self.session_token = None
        self.email = email
        self.password = password
        self.use_cache = use_cache
        self.mfa = mfa
        self.session = requests.Session()
        self.req_kwargs = {
            'proxies': {
                'http': proxy,
                'https': proxy,
            } if proxy else None,
            'verify': where(),
            'timeout': 100,
        }
        self.access_token = None
        self.refresh_token = None
        self.expires = None
        self.user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) ' \
                          'Chrome/109.0.0.0 Safari/537.36'

    @staticmethod
    def __check_email(email: str):
        regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
        return re.fullmatch(regex, email)

    def auth(self, login_local=False) -> str:
        if self.use_cache and self.access_token and self.expires and self.expires > dt.now():
            return self.access_token

        if not self.__check_email(self.email) or not self.password:
            raise Exception('invalid email or password.')

        return self.__part_one() if login_local else self.get_access_token_proxy()

    def get_refresh_token(self):
        return self.refresh_token

    def __part_one(self) -> str:
        url = '{}/auth/preauth'.format(default_api_prefix())
        resp = self.session.get(url, allow_redirects=False, **self.req_kwargs)

        # print("xxxxxxxxxxxx")
        if resp.status_code == 200:
            json = resp.json()
            if 'preauth_cookie' not in json or not json['preauth_cookie']:
                raise Exception('Get preauth cookie failed.')
            
            print(json['preauth_cookie'])
            return self.__part_two(json['preauth_cookie'])
        else:
            raise Exception('Error request preauth.')

    # def getPreauthCookie(self):


    def __part_two(self, preauth: str) -> str:
        code_challenge = 'w6n3Ix420Xhhu-Q5-mOOEyuPZmAsJHUbBpO8Ub7xBCY'
        code_verifier = 'yGrXROHx_VazA0uovsxKfE263LMFcrSrdm4SlC-rob8'

        url = 'https://auth0.openai.com/authorize?client_id=pdlLIX2Y72MIl2rhLhTE9VV9bN905kBh&audience=https%3A%2F' \
              '%2Fapi.openai.com%2Fv1&redirect_uri=com.openai.chat%3A%2F%2Fauth0.openai.com%2Fios%2Fcom.openai.chat' \
              '%2Fcallback&scope=openid%20email%20profile%20offline_access%20model.request%20model.read' \
              '%20organization.read%20offline&response_type=code&code_challenge={}' \
              '&code_challenge_method=S256&prompt=login&preauth_cookie={}'.format(code_challenge, preauth)
        return self.__part_three(code_verifier, url)

    def __part_three(self, code_verifier, url: str) -> str:
        headers = {
            'User-Agent': self.user_agent,
            'Referer': 'https://ios.chat.openai.com/',
        }
        resp = self.session.get(url, headers=headers, allow_redirects=True, **self.req_kwargs)

        if resp.status_code == 200:
            try:
                url_params = parse_qs(urlparse(resp.url).query)
                state = url_params['state'][0]
                return self.__part_four(code_verifier, state)
            except IndexError as exc:
                raise Exception('Rate limit hit.') from exc
        else:
            raise Exception('Error request login url.')

    def __part_four(self, code_verifier: str, state: str) -> str:
        url = 'https://auth0.openai.com/u/login/identifier?state=' + state
        headers = {
            'User-Agent': self.user_agent,
            'Referer': url,
            'Origin': 'https://auth0.openai.com',
        }
        data = {
            'state': state,
            'username': self.email,
            'js-available': 'true',
            'webauthn-available': 'true',
            'is-brave': 'false',
            'webauthn-platform-available': 'false',
            'action': 'default',
        }
        resp = self.session.post(url, headers=headers, data=data, allow_redirects=False, **self.req_kwargs)

        if resp.status_code == 302:
            return self.__part_five(code_verifier, state)
        else:
            raise Exception('Error check email.')

    def __part_five(self, code_verifier: str, state: str) -> str:
        url = 'https://auth0.openai.com/u/login/password?state=' + state
        headers = {
            'User-Agent': self.user_agent,
            'Referer': url,
            'Origin': 'https://auth0.openai.com',
        }
        data = {
            'state': state,
            'username': self.email,
            'password': self.password,
            'action': 'default',
        }

        resp = self.session.post(url, headers=headers, data=data, allow_redirects=False, **self.req_kwargs)
        if resp.status_code == 302:
            location = resp.headers['Location']
            if not location.startswith('/authorize/resume?'):
                raise Exception('Login failed.')

            return self.__part_six(code_verifier, location, url)

        print(resp.status_code)
        print(resp.headers)
        if resp.status_code == 400:
            raise Exception('Wrong email or password.')
        else:
            raise Exception('Error login.')

    def __part_six(self, code_verifier: str, location: str, ref: str) -> str:
        url = 'https://auth0.openai.com' + location
        headers = {
            'User-Agent': self.user_agent,
            'Referer': ref,
        }

        resp = self.session.get(url, headers=headers, allow_redirects=False, **self.req_kwargs)
        if resp.status_code == 302:
            location = resp.headers['Location']
            if location.startswith('/u/mfa-otp-challenge?'):
                if not self.mfa:
                    raise Exception('MFA required.')
                return self.__part_seven(code_verifier, location)

            if not location.startswith('com.openai.chat://auth0.openai.com/ios/com.openai.chat/callback?'):
                raise Exception('Login callback failed.')

            return self.get_access_token(code_verifier, resp.headers['Location'])

        raise Exception('Error login.')

    def __part_seven(self, code_verifier: str, location: str) -> str:
        url = 'https://auth0.openai.com' + location
        data = {
            'state': parse_qs(urlparse(url).query)['state'][0],
            'code': self.mfa,
            'action': 'default',
        }
        headers = {
            'User-Agent': self.user_agent,
            'Referer': url,
            'Origin': 'https://auth0.openai.com',
        }

        resp = self.session.post(url, headers=headers, data=data, allow_redirects=False, **self.req_kwargs)
        if resp.status_code == 302:
            location = resp.headers['Location']
            if not location.startswith('/authorize/resume?'):
                raise Exception('MFA failed.')

            return self.__part_six(code_verifier, location, url)

        if resp.status_code == 400:
            raise Exception('Wrong MFA code.')
        else:
            raise Exception('Error login.')

    def __parse_access_token(self, resp):
        if resp.status_code == 200:
            json = resp.json()
            if 'access_token' not in json:
                raise Exception('Get access token failed, maybe you need a proxy.')

            if 'refresh_token' in json:
                self.refresh_token = json['refresh_token']

            self.access_token = json['access_token']
            self.expires = dt.utcnow() + datetime.timedelta(seconds=json['expires_in']) - datetime.timedelta(minutes=5)
            return self.access_token
        else:
            raise Exception(resp.text)

    def get_access_token(self, code_verifier: str, callback_url: str) -> str:
        url_params = parse_qs(urlparse(callback_url).query)

        if 'error' in url_params:
            error = url_params['error'][0]
            error_description = url_params['error_description'][0] if 'error_description' in url_params else ''
            raise Exception('{}: {}'.format(error, error_description))

        if 'code' not in url_params:
            raise Exception('Error get code from callback url.')

        url = 'https://auth0.openai.com/oauth/token'
        headers = {
            'User-Agent': self.user_agent,
        }
        data = {
            'redirect_uri': 'com.openai.chat://auth0.openai.com/ios/com.openai.chat/callback',
            'grant_type': 'authorization_code',
            'client_id': 'pdlLIX2Y72MIl2rhLhTE9VV9bN905kBh',
            'code': url_params['code'][0],
            'code_verifier': code_verifier,
        }
        resp = self.session.post(url, headers=headers, json=data, allow_redirects=False, **self.req_kwargs)

        return self.__parse_access_token(resp)

    def get_access_token_proxy(self) -> str:
        url = '{}/auth/login'.format(default_api_prefix())
        headers = {
            'User-Agent': self.user_agent,
        }
        data = {
            'username': self.email,
            'password': self.password,
            'mfa_code': self.mfa,
        }
        resp = self.session.post(url=url, headers=headers, data=data, allow_redirects=False, **self.req_kwargs)

        return self.__parse_access_token(resp)



class TokenHelper():
    def __init__(self) -> None:
        self.url = 'https://openchat.geekgpt.site/auth/token'
        self.arkoseToken = "409178bff6070c8a8.8970431604|r=ap-southeast-1|meta=3|metabgclr=transparent|metaiconclr=%23757575|guitextcolor=%23000000|pk=0A1D34FC-659D-4E23-B17B-694DCFCF6A6C|at=40|ag=101|cdn_url=https%3A%2F%2Fclient-api.arkoselabs.com%2Fcdn%2Ffc|lurl=https%3A%2F%2Faudio-ap-southeast-1.arkoselabs.com|surl=https%3A%2F%2Fclient-api.arkoselabs.com|smurl=https%3A%2F%2Fclient-api.arkoselabs.com%2Fcdn%2Ffc%2Fassets%2Fstyle-manager"
    
    def getAccessTokenV3(self, username:str, password:str) -> str:
        if self.arkoseToken == "":
            self.arkoseToken = self.getArkoseToken()
            # with open("arkoseToken.txt", "w") as f:
            #     f.write(self.arkoseToken)
            # print(self.arkoseToken)

        headers = {
            'authority': 'openchat.geekgpt.site',
            'accept': '*/*',
            'accept-language': 'zh-CN,zh;q=0.9',
            'content-type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'origin': 'https://openchat.geekgpt.site',
            'referer': 'https://openchat.geekgpt.site/auth',
            'sec-ch-ua': '"Google Chrome";v="117", "Not;A=Brand";v="8", "Chromium";v="117"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',
            'x-requested-with': 'XMLHttpRequest',
        }

        data = {
            'username': username,
            'password': password,
            'mfa_code': '',
            'option': 'apple',
            'arkose_token': self.arkoseToken
        }

        resp = requests.post(self.url, headers=headers, data=data)
        respjson = resp.json()
        print(respjson)
        if 'access_token' in  respjson:
            return respjson['access_token']
        else:
            return ""
    
    def getArkoseToken(self):
        url = 'https://openchat.geekgpt.site/fc/gt2/public_key/0A1D34FC-659D-4E23-B17B-694DCFCF6A6C'

        headers = {
            'authority': 'openchat.geekgpt.site',
            'accept': '*/*',
            'accept-language': 'zh-CN,zh;q=0.9',
            'cache-control': 'no-cache',
            'content-type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'origin': 'https://openchat.geekgpt.site',
            'pragma': 'no-cache',
            'referer': 'https://openchat.geekgpt.site/v2/1.5.5/enforcement.fbfc14b0d793c6ef8359e0e4b4a91f67.html',
            'sec-ch-ua': '"Google Chrome";v="117", "Not;A=Brand";v="8", "Chromium";v="117"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',
        }

        data = {
            'bda': 'eyJjdCI6Ik9weldRTU1NbkZXalBXaHRuOWVUT29MMzNVK1Q1d2lldlNCRVpaR2ZFREhjOXhpRkx5TkNXQXNzUXRJSjBvc1I2RlMzcHBmcDNZTjVVdVl4YzZwdkE4RjhreFM0MFRQU2lXQlRQaldYTTJCTkFwZW5CMHBBeE0reDB0ejFnNmR0UXlXRkRoYlIrTkNTenVrVUd3UEpabjZUSGlDWG9WNTdDbmYySitpU1hQYmp4bWtKS3krK0hXZU4zbWxsdEtjck5QZlhNS3hrZU5iblZxUkwwVHJVWDRTZ3BJbk4yWVkyWDFGZ3lkTDNlQytIa0VrSjgrMGlxd1VvcnRTcUtWRUowK3dtWmlkVnZ5NWFFWHNKbjRuMUVDa0NocUVzSStoRlhRWEpFSkgwTkRpbCs5YTJMV1lubC85SXIwVmJaaGdwZE5GdnVOMEJFM2M0TTRNQyt4S3A0STBUZ0QvUGpPeTNER3liRnR1REtBVEhqM3BnSkhzY0xENjBIbFdFSUt1LzVqaUJRdGVIOGo3aFFodVJIOHlXa2IxWG9pNGFpVzNzclBLVVVyZXZMcWJpcWxRWEp3QTZpeUJxLzVlT05SMHUwcDdvbFkyeDZoRjVVeHVtNG90NThjSW4ySUpmVEJtZHJCbjV1Z2M4VFdYRWlBc01MOGRmNmNTcFdoM1dnaktBUzVEQ1Q4STFMSE10bDlwZ3czZC9ncWJhSDNSNzlYM3VCd0hvMGpLZ2RZTkFTOEtqK3c1U1p5blZ2VHMybUs3UzRneUR1RlQ4TW93OUExN3U3R0RWZ0F4Vlc1L3BUejBJWCtoWU02QUsreFNoTjI4RVNZVi9DTkdtSUVTUEJCakFxTUhYemQ3T3Z6bGhyV3FZalpFdU5mMmtmQnp0ei9OMFNIZzV3M1lwdFhVaEUxSmtEYWhKak1oTzVSS3JGMzE3REZRK1J1S2lKa0ZKRHNMWDF3elJtL1hXTzloaUFkOHJzalRWb1JnMGt2d1FlbjVrMVNwK0x2T0lYUE8xWStsaHN4bUx0VlpSQ2lHbmdmQmFSUi9kTFA2VFMzWGNBUHlxU0hYSnBvd09DVVlLUWRabnYweU41T3FoWThNWTN4OGN5NE1ESVdFaXZINTVJTVphbU1GS3c0UHhSY2JaUk5CYWMvbkIrT0VNY0RrdW5ka3NiMHFSUEt3NVNGM3NrakpOM2txNm5YUzk5NG9OeXhEem8zQi9uK2I1aHNpVFptaDdkWld0eW12WloyVnNqWVFwUEpXcmFmbUFCQkZhaDJ2bis5S0RmSys5eUxjRWlJb0NaMGY0LzNDNjFac0dQL05oNUFiS3RMQk5EQjZHSm1ycFFjSzA4MUFIa1BXcWhtTWE5RmxDeGJrTVAyT0JSMS81cXRxZGJRQitsRDdncGZDWVNBYXc0bExxbitQaGtKcS9lWWpnd1pWWEU2OHYzaFo5UFJaaWFMbUxVcGxTODB3SkhOVlNQejU5QUFGNjlqK0RDWnBXN1RKK2U1OVA1eHhleGJ0MHN3WkJyaVZ1cWlLZ3ljOUxTZyt4SnRLUzZQTU1hdW9JVHlsRTdLcjhWZ3VLZ1dlend3OTMyR1MwV0daS3kySTJDY0prM0c5SFBJaUdQOXY1WEt6WExtbGVUb2NuSERxQ014VVMxUG0wbFBNcW1Gb0RNcmxiZ25WWDc3eXpYTHdaMWo5c3ByRnVEazBBY1ZSNFQvV2tWNTZWZHFia3hCUkpZLzNvZUQwSWIyTWs4eGpWMEV3ajU1ZFFNMmJJTUY5bmNnM0dNc1BMNVlYb1c1L0pkUUZhTEwxT0hxVHdPcHdHSUtzSHpPYnczdTc5MElrZGVHeE0wMFlMNTlXZ1Rnb1Q5VFVaTUxMNWI2OUZFMlRFVG5kSlAwZjBQcENTaUIrMTRRUGZhRWpiTlNZSEdxY1I0NzFlbEV4OER1aGhNbVM3VG9kMU5DWHlvOEdnZTlrKzJJb3F3a0RPNTdpZVRLOEFpZ29OTU9yeHR3RUtSY2N3MWxITGphK2FSZGkzZ2xxMUpXNDlCbGNKbXpnLzZSUXAybFU4ZFE3ZUhkY2tBcERBYi9nSnlCekFmTGhoZHM3SmZ5RzFsSkZSUWlDTTJtMnl5Y1Q1K1RBQTZKMGx2bmJDUFRIaVI4VjJONmFndjhFR0FSR3pzWXJabjZ3aXFPS3NoTTVDMyttL3ZyVmNxVm1Fd2ZvUEVkcEhac0EyY1N6YU40RjNXUjBSKzNpVzBxUnVVa2xCVHJDYWlSR2pCMzBlUnJJdWdxaS9OWVJMMlJ6bFF6SkdVa0pENWR2aTQ2UW5nSElmelFQTitobHdsT2x0ZHZWVW5Pc0ViY2F6UzkyU1JzeG1FdDFaaVBrajJVeUkzdGhGbkkrWWNnS1RZc2JpRHhPT1FQdmtsQ3FlcUdiTDlPNjBOL29XNG5VU1c3Wk9tTlBoMUxSSmF0aTQwc3lGQ0NvWkVCanFzZTEyRDhyaHZGL3dBMnd0SWdMTUhZNEx6ODNTYXFabmVGQ0dIY09DREpWc2UzZ29QNHF6YlhOa1dUdkp0UUlLMnFsU0NzaVVtVlI4ZUdjY2FTMmtBNjR4VkhYUkJKUXlpYzFNSWVxVFlIOFdlZjdmZkpOTHE2R3JjMkJUS1NJaGRScFNvZlZoRkQ4TERCS21XZGZOUzd5eklFOVA4MHRlQm5nOTlnb1lFR2VlSGs1dy9VMWRnNVk1N3FCYTVITjQvM3ZNUEtZbTFqS3ZVK1AyR2E3WkVka3V0dmpSbFl1NzBiRGhVQ2FGcUZOMksxL0xxZmdhaThnVERjUDRnL1FRMVhNNXRWSHg3TXFDWjhSWit2SG83STdlZHpGbVJnTERrZkxoYUt5VHBCZXFDRDQzTlBFanFNNUMxamR0QlVCSlNHQlRoWXJMdjVHbVdZV1ovWnlScTVUTkRNcUZxdjBJODc2R29iaFJjVVZhM2lkc1hCMHlDTlRNQnFGZEVpRkcyZVdYSGQ1UldaUDRjaWxMbzBjOWNPMTh5WUJoSFRVMlU5VjFiS0tvQTJxZGhhcVJUN2Q3MW5NM1FndUJZMW5pTm1ualptZGdnaHZoUHgrYlZhYkpXeGdnRVRZSXJ0b3pPUEk2dXpyUzBHUkE3MjE5emx3VFl3cVg4OTFrOGdTSnlwVEJPckgyazRZVXNXSDBhck01RURGcEJEdHJ5WituTENTZitEMWlOTGtaVndQV0xJaWh4MzZJMXBHdG9MN0NHenNKdzl3RFE2bDhYNTQxYUhrYVZ5K2tlcG1TNkNMNDgrWUlqSFBmUWNmZjdUT1RxODU3V3Jmcmc0bWhsblM0aE05dmt2clQza3ZJRmRlejZqUUZNb2VZYVFOc1Noc0kwK2E5clBqVWhZeVp6K0JuYTMreEQ5MDVISHc5bXhObWdTWFlsRUxDQ2pPc1NsRDl1UFB4TWxhWkF4Y1cvMEpNWnlsMTVKb1FLcjNHUjZ3QzRRTjVJWTJQMnRuS0Q3QjVXUy9BOC9teTIvWEEzWjdQUmMwd0pKdXl6aUtXUVhpY1FWejhsV3JUOFBIQlVlTVJUNW81aXJEVGVMM3d6eUxwUU83SjJaR0t2RUZXRTlUa0FrTDUyNjR1OHN1Uy9nUTVMR2ZEZWRBbHVjeXN2OVJndE5PUVdUOUVac1RDQ3BEaXVKOEZyNG5pMnBjSVJQRTlVNC9XZ1hvMXE2TE5sdVBhRmpjMVEzT1NVL3Jwcy9XTEdtVFNrTXZIc1lZUmNiQXN1MGtBeEVVUWNzamN0bHFNVmRkZjMyY2Jxekl0Ri9HcVBnSXNPdnJDcmNzN3drWE9ObFBQTExYRTNQQzcycnVqRjNsZUpTeFZNb2Y1aFd0RWphcnV2K0c3NEJ2Zk9zS3JPeGhnemZ5MnNwenNYSDh0Zk9venJkV2o3alR3MnF3S3lDcDdTMHYwemsvdndab08zLzladDZqNEFKS3gva1VLQUxMamtFaWVQdHU3anRhS0lkVHlHb2ZtZEhGeUt2QjBacnkwcW5EK3FPb0RLbEVmYUl5NFk2TkNWWW41c05MbFNWdHNRWENheExDM0Fhc3RyNktIeDRCd0M3czAvUU1qaW4xcWl5ckNvWlNGMFdlU1ozYTFWbzc4R0dPaVMvQVdRQWl2bjh3UEJwR2lXaHh3Mnp0dDk3MjJWU1JROEFYUytYSlVkY2kybnFsM1JNVERnUU9GY2RPV0I0a0d0RWlDeGQ5Vk9HSHJCUnYvazdOU2lJWWM0YjZHVmJpcmNVR1ZwSUQ4Unp3ZUtWN0ppQ0M3WEFjSTZQZDJURjNtK0JLSnZFWFhVVUNtdjNLOWR5MlF4ZnZSell4Y3hqN083Y1ltOWY5TUh1Y2xDNXNxdEovSXc5ZkFVL1BFWGhNVHhBS1B2QkdzT3NLZGpqRGJmU2tSUEtqZFFOd2xtbU9QbG1LSGw1eXVkaEZpMEpZYXEvczV1WUkyaTdKWHMxQ1FkTUpZNy9meDF3K2RsVjNjUk9vYSttN1p2VHo5ODkyTXRmcEtQODR1LzB1Sm1pRUVyUXpKOTBRS2tMQ04zMEpSYmJTd0g3am9PSjZtaFNPMmpJazVTSEtBNklGQm5nL1loWVlFNXp0T1Zud0pUT0xjdE1NUUppYnpFR25TdjZCM1pFRFNqU2xEbTRNcTNveFlMZFBGeG9VQWpaZitOQ3kxb0dsVm5tZUI5cXI4ODlZT2ozVnF3aFdZMGx1R2V0dnB3RVNyU1NGT3YxN1l0d0l3L2hOMUNNMWptbWdnNXViSnBaZld1WmxEK21wTTNwRDRHMzBwczFpWXd0b2R4WlN0eUxMY3BhSDlaQW9QellEVTV5SmgvbVo0QUhJUUtTUzB4T0pLYnRFQk10b0FHS0hCMVpTSG5hcURzcVpLOCtPZXNLSTRFQUdXSW56NWZ5WHp5SFoxSHFyTVE4eUlSSituUm9YcmdQRnJTVHVHVmZaZ1JaYVhodUpkdEtXRGZOdDRlU3o4TW5VRlFsSlBxQUQwYW12SWVWNm5pZWhrRElwNFVtbTRtK21VQ3hSMU8wZlQ2cEtwKzJNSk1CeEl4RHpJREZFZnFtMDU3aTlGSFU1RXVOUEt4dFNVWlpZZ3JZK21ZUWJGUzB5WFQ4RVdwc1B2OVFQb0hMOGg2ZnYwMU1jTUlxOEZJVk5XQk9COGxOaHpINzUxVXpBRHZVeGZRMGFnYlAvSVRlNnNWdTE3UjRud0dpeWZlNDJkdnIvUTFEQUxmbkVaSzFSYk9YWXpBV0laWXVndmRFKzJxTDlxZ2x4SFdPOVMxakNVTEFLS1JTOW1lTStTYy9NQXBsd2dFVlZIMEptMDZZMGlSN3Ztc3JLaEZ4T2hQcU1Mc0FzQXdFbU9QSmtnbjF5T0lRUitJalJ5VFB4Rko4OWFwTkxXWWNIbVFuR2RoOVVEL05ZeERoZ1JjNStUYVp4UDR4NUpnQytJdVI3cnl4dFVNdWxGcTc2Q0hwZy9nVmI2aXZJTU1XMVhoVC9jY3Mrbk1INlgyN0poVU5IWlNuQzNhS2U2aWh4K01RY3dPWmtnamlDaGt3VnZZekxsSEN1a2RQSCs5YzlnZDk3d3hjRG9VdTJrZFQxeXdkUFJYZlgrY2h4RW9uNWhVNG4zWjlJdFdRSlc5YWc2bUt0V3M4T3EvakVyNS9Mb0pKQkJVN1hSY01IT25mK2xqV1dRalNVa3dUc05xRWxzOHVndytzaXBMS1owSlhxZnIyc1hoL2tzM1NzOWk2K09ZTDJHaEFpVFgwSHpCQm1sUXZJTXBDNng4b1lMWUYyVXVDaTV6bVdGbzN0clRVby9JY1AyaU5TQmZuMTdCQzJFTkEvOHd2dElUN1FuOTF0Z0FTeDVRVmtBUU5EczZUM3lGSUlYajBmaURRbDJRb01uNXZDcUprdjExeWNIQ2hqME5VNFFUSHRYVnNHNC80ZzB2cTM3bGpFdXQ2RFN6MW5pclNUT0ljRFNFemJ6d3BKUFB5UWRZaVdpanIxdzJJNm5mV1I1ZTZQR1BoSWtieGNObENXSjZnZ2JqUlBLOFl1bUNTWkZrbU03RmhuWGh3R3N0anora1ZxdWlkeEk4c084RXY3cmNCZklCSDlYUFpGK3lLTEQ5aWNLWUh2SXVRZ2oxdW1NNmVxZmZLaTdHam1WM0ZqYmlZb1Mzampnd2dlZ0hZZDJvNEhGcUZuN3J5bG1Fa25qVWFxeHdPZUl2ZzQrdzdTOGRCb25mdTJ5UzZyekRKaC9SclJhakNGL3QxeGtFYkJXSzNWcW1sK20wNnV4Z3JSc0gwM1VVMllHR2ZsRm1ybVMrTkM0d0tXVWNZSXQ5S3F3MzRmT0FxZVg2OVVzYTJDbXBYOGtMVHRYRjJDV3haR2tnL1I0QTlPRCttWkZIb1dya2UvTlpjbUJKNExqVk52NHkyVDliR2NMSmtmVVpwazR2ZzFQTi9JZFRUS1h2RFF5Z0UvOU5iTDRBZjJySytPZjVrdVN2RzVveUFIQ0ZCbTllWnkzV2IreVJGTWtjdmR4UHVlaEhSSDhIcGltRUhvQVNZdVk1bzJpV0VxZkZBMHFtWDdaUnlZSDlGRitOam9NZUdFT3hnckNTMHAyYWQyaldJMFQ4N2Jwclk3YXpRcngzR3ZYYkVPLy9DeUtPL2pVdTNHRWVhRVRjK1lwMUlRMmY3L0szUEUxY0p0ak1BRUptWjl5QkdodytJRnozTkdrRnRWcStqZUxyblFBZ1Q2Y1ZRZWc4ZUJ4Q2ErR0lFWjVEQkt0UnFFck41M2JiMXU1VmRoV25hMDVVU1daekZ4aFJWVEZZcHdEd0thb0JJME12dFVKWGphUm5iUWJsMnc2VVRHMmNtZWMvcnVRUnJFR29DcTZVZHpUSFczcEJTMjdvUGNJZnQrNTdZd2JiVmpSTmEwVW9yejllKzlkdkxmQmRXU0FDbE9tRGlNSU9maFBQRUVUV2x5UXF1aDlsUDcwbnErZWVqUkxyTEhVMDB4ZmVxeElReFplQjdzQ2NEM2s3QStLd3p6K2xiQlN6M2dJRXlPUjREeDdJVC9oUEdxM1cyWGd4SmJxa0ZKdmtQU1E2M2ZKZzJVK3VNY3RPeFIySFJLSVBjMFNFVGFvQnlnRjh2LzQ5cWZFUnRHNzh1VFpIT1h2UnFaSmZSLzJKc3dDOTNZd21IUHpvSlk3T29NcmFjWFpqWHRQdUxlbnc2QWJiYWNXdDRFZWpvSnptd2hiYjA5M08vYk55b2JMU2ZGc0JVN1pDQVpyRnRqWTV6K3EyaW05VndydHJSQjR2L3hnOVFYUlF0am0vR3RrMkZsR2wrWGdVZjRrOWhNalhNb0NUR0t5SW5zcDlsOGQ0dFhXdnUzMWVxSTVoVWh0MUV2M3czSGk3YnJnWldnVkpiTUFzNm85VzRRcFJ1T04vQVpYOG45N1U4UHpwN0U2cjhQOTZKdWxDbFpIRG9XNjQ2TUk1MjYydW5WZXYvT3JRQmpOTDRCYkdTVjBIVGVhdWxxNUZFcWNuN25uV2hVUjRXQ2hNRDdNd0FLU0RFQXQ0SjF3eGF0M29aNitkek15bGZMT3JtNDFVdFlKTCtaQ1kzMXdUNjVsWXJWSmVTYU53dGwwQjQ2OVpjdFFaN3RxSUROTUg0ZXRkbUtHbW9jWkUzd2IyZ1RQRlBmcitOTHNJUU42ejZ6YVN6cmticGs3emdxcXhRQ2gxVm9pRS9LYU1lMWF1eWkvK2ZFUCtuVjFsaGhsdG5VcEV2SFRXTEFHVXNWOVk0RW9WbGVJRHV3VzJmTjk4aS9zUkRJdnc2WVlZS0J3cHRpM2x0bG9TVzg0azdnSzN6d0lzczlkOUJoL1pHa2xCNWl6ditnaUYrQSsyckpUS1FNL3NCeE9IRitkSlM2Y3dhdU5jdkZiTHpBMjdBUHVFT2pMcksrajV5eE16ZUYrZ0owaGdQamdHY2ZyVFMwSmRyWkRrTDF0SWg5RUIrOW9XWVR2YUZHZ0ZZWlIvNEdEK2xZSXoxbXBnUmxvVDF6Z3pZc1VvZUllVG9HQS8xMXpoazQzOEh3NW5FcmwwcjdRb1J3SEwrOWVyRzQ5YmVrQ2tuZzNVN24rR0cwYU56VkFneW81eWNXZ0RscHdnd0RRMnhHRmJ5WGtEZ1V1cWVCUDNwdnEyZWx5ZlJ2aXN3WmFuK2JxMnRqbll3dnFZdVh0VFczSWQzUHppMUJGU1EyMWtFcUc2MmRkZTZuZGZKVXF1MktWdHF2RmFhSkNKNTU4TE8vRHBzRmFpR3RITEJFeTFIdHpJdjllZ2VrelNHNU1tMlhFcE5kdEpQdmFWUER6Q0hBQWVkY3JxSXhiNWhhQWhEWkRYU2Z1K1A3Q0w3NmZxOXhiaDJvY3YzWEFzem1zRVEya1FSdnBWeGJ0UWRLUnlIMU9wajRJY2VUK2FndDRweGt0L3Jnb1NlS3RuNmFYVGRNUUUzNS9VbnZyQ1U5Z1VTSEVwd2JhVWxlVDZZWk9rYnhDeStYeUJWM3RtdHhJQlF0d2ZZRWpEYlQrMEI2ZUNWR1ZIejZJY2ZJWjN0a3drNG5WNlh6TjdFNDh2V0YzRWgzVjV2bDlNOEU1RlZ4VTVCS3RKWkJ5NnBmYnBpeCtmemVrSTZRMWx2NGRieEVidkM0KzltUkRiY3dHa1pYWS91ZWRPZDhTd2l4Um9jeWtrNDFjbEhNMzY2WUZtTXpRZnVzaitsTW9jUWhQTUJZdmlXY2JaUzV1RTZlaGlaOG53SlNXZjI3VjU0ckdyWnpjUlcvdlA3UXBkWlROY3IxakFYQ3IvODd0M0gwNzZDQXVRdnZVc1o1dmJ0NGQyNi9lTmZvSUw1UVRscGZzL3V1Tmp4NGZwRWxtUUNRRG8yaytlZFB1QXlRSTNUUFdlWVFLM0JsVktzUXFxK1EvQzJKWU91Mkp1MElPL1JTVHRaZ3JadHZZN2NSTmNWS01XeDA3MjFsTXpKaWk3dUpCMFFEQlV4TERQYUVoTHpwbTdSdTByemVMa3JWUFl2cSt6Z2RYODJhdXdlYjVLTVV0aXoyb2hiVktmdmtYamYwSDhDQjdDZFdQUWgveTQrMjVtdGdRUEtRQVFGVzE0WlBFZkRrT1Zjc0o5ZEgyc3hiRkdFV3pVRHA2bldJckJvMERoSm42MnFnTnNDZWs2d0k1MlVrT2RJcEtHRzQ0b085c252M0ZIYnJROFpxTkNrTDNjOUROdWIrQWdET2RYRUprWTdacDVObnAzeFBHQ3A1dGp2NDZEOHFJQTVaOEZ1T3dNWm5zck1FU3NJZUlaYTVSTWNwanUrRjBOaDd1NzM3cWhrZGc2SXVOaU5RRm1uTDYwRllMSUNjTlgyMWo4cEQrVzhwQWRDUzNHMjNmTytkLzZMNExnMDlBUHREelhudG9wcUY0ZHl4N2hNTGZ5SE1aUUZ0QzNUcVZYQzh0QlpHdzN5V242Z2M4bWVwbEhpaS9EU2w3UnpraERxNGNqWlplQ1lheFNpdXJnOG10V29CcTVQM2JmUWtJTERBQ3RTTVVONUE9IiwiaXYiOiI0ZmRmOGFjMzUwN2Q2MjEyNzNlMTY1MTI4MmFhYzhmYyIsInMiOiJhM2ZmYTViMjZlMDRjODM4In0=',
            'public_key': '0A1D34FC-659D-4E23-B17B-694DCFCF6A6C',
            'site': 'https://openchat.geekgpt.site',
            'userbrowser': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',
            'capi_version': '1.5.5',
            'capi_mode': 'lightbox',
            'style_theme': 'default',
            'rnd': '0.50878010360099',
        }

        resp = requests.post(url, headers=headers, data=data)
        respjson = resp.json()
        if "token" in respjson:
            return respjson["token"]
        else:
            return ""

    def getAccessTokenByAnotherProxy(self, uname, password, proxies=None):
        
        url = "https://openchat.geekgpt.site/auth/token"
        # url = 'https://openchat.geekgpt.site/auth/login'
    
        headers = {
            'authority': 'openchat.geekgpt.site',
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'accept-language': 'zh-CN,zh;q=0.9',
            'cache-control': 'max-age=0',
            'content-type': 'application/x-www-form-urlencoded',
            'origin': 'https://openchat.geekgpt.site',
            'referer': 'https://openchat.geekgpt.site/auth/login',
            'sec-ch-ua': '"Google Chrome";v="117", "Not;A=Brand";v="8", "Chromium";v="117"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'document',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-site': 'same-origin',
            'sec-fetch-user': '?1',
            'upgrade-insecure-requests': '1',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36'
        }
    
        data = {
            'username': uname,
            'password': password,
            'mfa_code': '',
            'action': 'default'
        }
    
        response = requests.post(url, headers=headers, data=data,allow_redirects=False, proxies=proxies)
    
        # 检查响应是否为303重定向
        # 获取响应中的Set-Cookie头部
        return response.json()

        set_cookie_header = response.headers.get('Set-Cookie')
    
        cookies_dict = {}
    
        if set_cookie_header:
            # 将Set-Cookie头拆分为Cookie键值对
            cookies = set_cookie_header.split(',')
            for cookie in cookies:
                cookie_parts = cookie.strip().split(';')
                for part in cookie_parts:
                    key_value = part.strip().split('=')
                    if len(key_value) == 2:
                        key, value = key_value
                        cookies_dict[key] = value
    
        # 查看是否存在键为"ninja_session"的Cookie，并获取其值
        ninja_session_cookie = cookies_dict.get('ninja_session')
    
        if ninja_session_cookie:
            print(f"ninja_session Cookie Value: {ninja_session_cookie}")
        else:
            print("No ninja_session Cookie Found")
    
    
        url = 'https://openchat.geekgpt.site/auth/session'
    
        headers = {
            'authority': 'openchat.geekgpt.site',
            'accept': '*/*',
            'accept-language': 'zh-CN,zh;q=0.9',
            'cookie': 'ninja_session='+ninja_session_cookie,
            'referer': 'https://openchat.geekgpt.site/',
            'sec-ch-ua': '"Google Chrome";v="117", "Not;A=Brand";v="8", "Chromium";v="117"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36'
        }
    
        response = requests.get(url, headers=headers)
    
        # 打印响应内容
        print(response.text)

    # def MakeShareTokenGetter(self,proxyURL:str) -> callable[[dict], dict]:
    #     def mapper(account:dict) -> dict:
    #         self.getAccessToken()



class FakeOpenai():
    # 定义接口的URL

    def __init__(self) :
        pass

    def makeAccessTokenGetter(self, proxyURL:str) :
        def mapper(account:dict) -> dict:
            account["access_token"], account["refresh_token"] = self.getAccessToken(
                account["username"], account["password"], 
                proxies={"http": proxyURL, "https": proxyURL}
            )
            return account
        return mapper

    def getAccessToken(self, username, password, proxies=None): 
        # 定义请求头
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        # 定义请求参数
        payload = {
            "username": username,
            "password":password,
        }
        response = requests.post(HOST+ACESS_TOKEN_URI, headers=headers, data=payload, proxies=proxies)
        # 检查响应是否成功
        if response.status_code == 200:
            print("登录成功！")
            # 解析响应内容，获取 Access Token 和 Refresh Token 等信息
            responseData = response.json()
            access_token = responseData.get("access_token")
            refresh_token = responseData.get("refresh_token")
            # 打印获取的信息
            # print("Access Token:", access_token)
            # print("Refresh Token:", refresh_token)
            return access_token, refresh_token
        else:
            print("登录失败，状态码:", response.status_code)
            return "", ""  
        
    def makeShareTokenGetter(self, proxyURL:str):
        def mapper(account:dict) -> dict:
            account["share_token"] = self.getShareToken(
                account["access_token"], 
                proxies={"http": proxyURL, "https": proxyURL}
            )
            return account
        return mapper

    def getShareToken(self, access_token, proxies=None):
        """_summary_
        接口描述： 注册或更新 Share Token 。
        unique_name：一个唯一的名字，这里要注意相同 unique_name 和 access_token 始终生成相同的 Share Token 。
        access_token：ChatGPT 账号的 Access Token 。
        site_limit：限制 Share Token 使用的站点，格式为：https://xxx.yyy.com，可留空不作限制。
        expires_in：Share Token 的有效期，单位为：秒，为 0 时表示与 Access Token 同效，为 -1 时吊销 Share Token 。
        show_conversations：是否进行会话隔离，true 或 false ，默认为 false 。
        show_userinfo：是否隐藏 邮箱 等账号信息，true 或 false ，默认为 false 。
        返回字段： 返回 Share Token 等信息。 {'expire_at': 1696002304, 'show_conversations': True, 'show_userinfo': False, 'site_limit': '', 'token_key': '', 'unique_name': ''}
        频率控制： 无。
        """
        url = HOST + SHARE_TOKEN_URI
        headers = { "Content-Type": "application/x-www-form-urlencoded" }
        payload = {
            "unique_name": uuid.uuid4(),  
            "access_token": access_token, 
            "expires_in": 0,
            "show_conversations": "false", # 是否进行会话隔离，true 或 false ，默认为 false 。
            "show_userinfo": "false", # 是否隐藏 邮箱 等账号信息，true 或 false ，默认为 false 。
        }
        response = requests.post(url, headers=headers, data=payload, proxies=proxies)
        # 检查响应是否成功
        if response.status_code == 200:
            print("share token generated")
            # 解析响应内容，获取 Access Token 和 Refresh Token 等信息
            responseData = response.json()
            tokenKey = responseData.get("token_key")
            return tokenKey
        else:
            print("登录失败，状态码:", response.status_code)
            return ""


    def getPoolToken(self, shareTokens):
        url = HOST + POOL_TOKEN_URI
        headers = { "Content-Type": "application/x-www-form-urlencoded" }
        shareToknsParam = ""
        for t in shareTokens:
            shareToknsParam += t + "\n"
        payload = {
            "share_tokens": shareToknsParam,
        }
        response = requests.post(url, headers=headers, data=payload, proxies=None)
        if response.status_code == 200:
            responseData = response.json()
            return responseData.get("pool_token"), responseData.get("count")
        else:
            print("pool token generate failed")

    def loadAccounts(filename: str)-> list[dict]:
        accounts = []
        with open("./accounts.json", "r") as f:
            accounts= json.loads(f.read()) 
            return accounts

# u, p = "ponhinson@gmail.com", "Mcx@hin6.626"
    proxyURL = ""
    proxyURL = None
    with ThreadPoolExecutor(max_workers=20) as executor:
        accounts = list(executor.map(f.makeAccessTokenGetter(proxyURL), accounts))
    with ThreadPoolExecutor(max_workers=20) as executor:
        accounts = list(executor.map(f.makeShareTokenGetter(proxyURL), accounts))



    print(accounts)
# print(json.dumps(accounts, indent=4))
    shareTokens = []
    for acc in accounts:
        if acc["share_token"] != "":
            shareTokens.append(acc["share_token"])
    poolToken, count = f.getPoolToken(shareTokens=shareTokens)
    if poolToken != "":
        print(f"pool token generated: {poolToken} containg {count} share accounts, please save it")
# auth = Auth0(u,p)
# print(auth.get_access_token_proxy())

test()

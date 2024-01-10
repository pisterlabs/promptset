from dotenv import load_dotenv
import json
import logging
import openai
import os
import random
import requests
import streamlit as st
import streamlit_authenticator as stauth
import string
import subprocess
from supabase import create_client
import time
from typing import Any, Dict, List, Optional, Union, Callable

import httpx
from httpx import Response

from src.utils import setup_logging, load_yaml
from src.solution import Solution

logger = logging.getLogger(__name__)

ADMIN_USER = "ADMIN_USER"
ADMIN_PSWD = "ADMIN_PSWD"


@st.cache_resource
def setup_app(config_file):
    # should first set all the env variables:
    # openai setup, wandb setup, 
    load_dotenv()
    setup_logging()
    supabase_client = setup_supabase()

    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    _config = load_yaml(config_file)

    _supabase = Supabase(client=supabase_client, **_config["supabase"])
    _solution = Solution(**_config["openai"])
    print(_config["torchserve"])
    _ts = LocalTS(
        model_store=_config["torchserve"]["model_store"],
        log_location=_config["torchserve"]["log_location"], 
        metrics_location=_config["torchserve"]["metrics_location"])
    _api = ManagementAPI(address=_config["torchserve"]["management_address"])
    _inference = InferenceAPI(
        address=_config["torchserve"]["inference_address"])
    with open(_config["labels_to_ints"], "r") as f:
        _labels_dict = json.load(f)
        _labels_to_ints = {
            v: int(k) for k,v in _labels_dict.items()
        }

    if 'supabase' not in st.session_state:
        st.session_state.supabase = _supabase
    if 'solution' not in st.session_state:
        st.session_state.solution = _solution
    if "ts" not in st.session_state:
        st.session_state.ts = _ts
    if 'api' not in st.session_state:
        st.session_state.api = _api
    if 'inference' not in st.session_state:
        st.session_state.inference = _inference
    if 'labels_to_ints' not in st.session_state:
        st.session_state.labels_to_ints = _labels_to_ints

    logger.info("Initialized")


class Supabase:
    def __init__(self, table, bucket, client):
        self.table = table
        self.bucket = bucket
        self.client = client

    def fetch_table(self, conf):
        label_subjects = []
        (_, rows), _ = self.client.table(self.table).select("*").execute()
        rows = rows
        for row in rows:
            if float(row["confidence"]) <= conf:
                label_subjects.append(row)
        return label_subjects
    
    def update_table(self, id, label, conf=1.0):
        data, count = self.client.table(self.table).update(
            {"label": label, 'confidence': conf}
            ).eq('id', id).execute()
        return data, count
    
    def store_prediction(self, path, cls, conf, id):
        self.client.table(self.table).insert({
            "id": id, "name": path, "label": cls, "confidence": conf}).execute()

    def store_image(self, path, image):
        return self.client.storage.from_(self.bucket).upload(
            path, image, { "content-type": "image/jpeg"})

    def fetch_bucket(self, name):
        try:
            res = self.client.storage.from_(self.bucket).download(name)
            return res
        except Exception as e:
            logger.warning(e)


@st.cache_resource
def setup_supabase():
    supabase_client = init_connection()
    sign_in(supabase_client)
    return supabase_client


def init_connection():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    return create_client(url, key)


def sign_in(client):
    mail = os.getenv("SUPABASE_MAIL")
    password = os.getenv("SUPABASE_PSWD")
    return client.auth.sign_in_with_password(
        {"email": mail, "password": password})


def sign_out(client):
    client.auth.sign_out()


def generate_random_string(length):
    characters = string.ascii_letters + string.digits  # You can include other characters if needed
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string


def auth(expiry_days=7):
    _name = "admin"
    _user = os.getenv(ADMIN_USER)
    _password = os.getenv(ADMIN_PSWD)
    _hashed_password = stauth.Hasher([_password]).generate()

    _credentials = {
        "usernames":
            {
                _user: {
                    "email": None,
                    "name": _name,
                    "password": _hashed_password[0],
                    }
            },
    }

    authenticator = stauth.Authenticate(
        credentials=_credentials,
        cookie_name=generate_random_string(10),
        key="admin", 
        cookie_expiry_days=expiry_days)
    
    return authenticator.login("ADMIN LOGIN", "main")
        

def handle_status(status):
    if status:
        return True
    elif status == False:
        st.error('Username/password is incorrect')
        return False
    elif status == None:
        st.warning('Please provide your credentials')
        return False


def page_config(title, icon):
    st.set_page_config(
        page_title = title,
        page_icon = icon)


def show_image(img, name):
    st.image(img, caption=name, use_column_width="always")


class InferenceAPI:
    def __init__(self, address: str, error_callback: Callable = None) -> None:
        self.address = address
        if not error_callback:
            error_callback=self.default_error_callback
        self.client = httpx.Client(timeout=1000,
                                   event_hooks={"response": [error_callback]})
    @staticmethod
    def default_error_callback(response: Response) -> None:
        if response.status_code != 200:
            logger.warning(f"status code: {response.status_code},{response}")
    
    def ping_serve(self):
        try:
            res = self.client.post(self.address + "/ping")
            status = res.json()["status"]
            return status
        except httpx.HTTPError:
            return None

    def predict(self, url, file):
        try:
            res = requests.post(self.address + f"/predictions/{url}", data=file)
            return res.json()
        except httpx.HTTPError:
            return None


class ManagementAPI:
    def __init__(self, address: str, error_callback: Callable = None) -> None:
        self.address = address
        if not error_callback:
            error_callback=self.default_error_callback
        self.client = httpx.Client(timeout=1000,
                                   event_hooks={"response": [error_callback]})
    @staticmethod
    def default_error_callback(response: Response) -> None:
        if response.status_code != 200:
            logger.warning(f"status code: {response.status_code},{response}")

    def get_loaded_models(self) -> Optional[Dict[str, Any]]:
        try:
            res = self.client.get(self.address + "/models")
            return res.json()
        except httpx.HTTPError:
            return None

    def get_model(self,
                  model_name: str,
                  version: Optional[str] = None,
                  list_all: bool = False) -> List[Dict[str, Any]]:
        req_url = self.address + "/models/" + model_name
        if version:
            req_url += "/" + version
        elif list_all:
            req_url += "/all"


        res = self.client.get(req_url)
        return res.json()

    def register_model(
        self,
        mar_path: str,
        model_name: Optional[str] = None,
        handler: Optional[str] = None,
        runtime: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_batch_delay: Optional[int] = None,
        initial_workers: Optional[int] = None,
        response_timeout: Optional[int] = None,
    ) -> Dict[str, str]:

        req_url = self.address + "/models?url=" + mar_path + "&synchronous=false"
        if model_name:
            req_url += "&model_name=" + model_name
        if handler:
            req_url += "&handler=" + handler
        if runtime:
            req_url += "&runtime=" + runtime
        if batch_size:
            req_url += "&batch_size=" + str(batch_size)
        if max_batch_delay:
            req_url += "&max_batch_delay=" + str(max_batch_delay)
        if initial_workers:
            req_url += "&initial_workers=" + str(initial_workers)
        if response_timeout:
            req_url += "&response_timeout=" + str(response_timeout)

        res = self.client.post(req_url)
        return res.json()

    def delete_model(self,
                     model_name: str,
                     version: Optional[str] = None) -> Dict[str, str]:
        req_url = self.address + "/models/" + model_name
        if version:
            req_url += "/" + version
        res = self.client.delete(req_url)
        return res.json()

    def change_model_default(self,
                             model_name: str,
                             version: Optional[str] = None):
        req_url = self.address + "/models/" + model_name
        if version:
            req_url += "/" + version
        req_url += "/set-default"
        res = self.client.put(req_url)
        return res.json()

    def change_model_workers(
            self,
            model_name: str,
            version: Optional[str] = None,
            min_worker: Optional[int] = None,
            max_worker: Optional[int] = None,
            number_gpu: Optional[int] = None) -> Dict[str, str]:
        req_url = self.address + "/models/" + model_name
        if version:
            req_url += "/" + version
        req_url += "?synchronous=false"
        if min_worker:
            req_url += "&min_worker=" + str(min_worker)
        if max_worker:
            req_url += "&max_worker=" + str(max_worker)
        if number_gpu:
            req_url += "&number_gpu=" + str(number_gpu)
        res = self.client.put(req_url)
        return res.json()
    

class LocalTS:
    def __init__(self,
                 model_store: str,
                 log_location: Optional[str] = None,
                 metrics_location: Optional[str] = None) -> None:
        if log_location:
            if not os.path.isdir(log_location):
                os.makedirs(log_location, exist_ok=True)
        if metrics_location:
            if not os.path.isdir(metrics_location):
                os.makedirs(metrics_location, exist_ok=True)

        self.model_store = model_store
        self.log_location = log_location
        self.metrics_location = metrics_location

    def start_torchserve(self) -> str:

        if not os.path.exists(self.model_store):
            return "Can't find model store path"
        log_path = os.path.join(
            self.log_location, "ts-app.log"
        ) if self.log_location is not None else None
        torchserve_cmd = f"torchserve --start --ncs --model-store {self.model_store}"
        p = subprocess.Popen(
            torchserve_cmd.split(" "),
            stdout=subprocess.DEVNULL,
            stderr=open(log_path, "a+")
            if log_path else subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True
        )
        p.communicate()
        if p.returncode == 0:
            return f"Torchserve is starting (PID: {p.pid})..please refresh page"
        else:
            return f"Torchserve is already started. Check {log_path} for errors"

    def stop_torchserve(self) -> Union[str, Exception]:
        try:
            p = subprocess.run(["torchserve", "--stop"],
                               check=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True)
            return p.stdout
        except (subprocess.CalledProcessError, OSError) as e:
            return e

    def get_model_store(self) -> List[str]:
        return os.listdir(self.model_store)
    

def error_callback(response:Response):
    if response.status_code != 200:
        st.write("There was an error!")
        st.write(response)
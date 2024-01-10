import time
from sqlalchemy import create_engine
from proxy_app.Database.models.base import Base
from proxy_app.Database.models.openaiRequestResponse import OpenAIRequestResponse
from proxy_app.Database.models.apiKeyToQuota import APIKeyToQuota
from sqlalchemy.orm import Session
import psycopg2

class ProxyAPIDatabase:
    def __init__(self, db_option, db_type, db_module, username, password, host, name):
        self.db_type = db_type
        self.db_module = db_module
        self.username = username
        self.password = password
        self.host = host
        self.name = name
        self.db_option = db_option
        if db_option == "SQLite":
            self.url = "sqlite:///proxy_api.db"
        else:
            self.url = f"{db_type}+{db_module}://{username}:{password}@{host}/{name}"
        self.engine = create_engine(self.url)

    def init_db(self):
        Base.metadata.create_all(self.engine)

    def create_api_key_with_quota(self, api_key, rem_quota, req_count):
        with Session(self.engine) as session:
            api_key_to_quota = APIKeyToQuota(
                api_key=api_key, rem_quota=rem_quota, req_count=req_count
            )
            session.add(api_key_to_quota)
            session.commit()
            print(f"Created new API key: {api_key} with initial quota: {rem_quota}")
        return api_key

    def insert_data(self, req_id, api_key, req_data, response):
        with Session(self.engine) as session:
            requestResponse = OpenAIRequestResponse(
                req_id=req_id, api_key=api_key, req_data=req_data, response=response
            )
            session.add(requestResponse)
            session.commit()
            print(f"Inserted Request Data: {req_data}")

    def update_remQuota(self, api_key, rem_quota):
        with Session(self.engine) as session:
            result = (
                session.query(APIKeyToQuota)
                .filter(APIKeyToQuota.api_key == api_key)
                .first()
            )
            if result:
                result.rem_quota = rem_quota
                result.req_count += 1
                session.commit()
                print(f"Updated rem_quota for API_Key: {api_key} to {rem_quota}")

    def add_quota(self, api_key, quota):
        with Session(self.engine) as session:
            result = (
                session.query(APIKeyToQuota)
                .filter(APIKeyToQuota.api_key == api_key)
                .first()
            )
            if result:
                result.rem_quota += quota
                session.commit()
                print(f"Added quota for API_Key: {api_key} to {quota}")

    def validate_api_key(self, api_key):
        with Session(self.engine) as session:
            api_key_to_quota = (
                session.query(APIKeyToQuota)
                .filter(APIKeyToQuota.api_key == api_key)
                .first()
            )
            return True if api_key_to_quota else False

    def validate_api_key_request(self, api_key):
        with Session(self.engine) as session:
            api_key_to_quota = (
                session.query(APIKeyToQuota)
                .filter(APIKeyToQuota.api_key == api_key)
                .first()
            )
            if api_key_to_quota:
                return True, api_key_to_quota.rem_quota, f"{api_key}__{time.time_ns()}"
            else:
                return False, api_key_to_quota.rem_quota, None

    def get_AllRequestData(self, api_key):
        with Session(self.engine) as session:
            query_list = (
                session.query(OpenAIRequestResponse)
                .filter(OpenAIRequestResponse.api_key == api_key)
                .all()
            )
            return query_list if query_list else None

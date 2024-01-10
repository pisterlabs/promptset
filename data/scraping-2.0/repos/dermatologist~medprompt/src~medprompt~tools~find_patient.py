"""
 Copyright 2023 Bell Eapen

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from typing import Any, Optional, Type
from kink import di
from langchain.callbacks.manager import (AsyncCallbackManagerForToolRun,
                                         CallbackManagerForToolRun)
from langchain.tools import StructuredTool
from langchain.pydantic_v1 import BaseModel, Field
class SearchInput(BaseModel):
    given: Optional[str] = Field()
    family: Optional[str] = Field()
    birth_date: Optional[str] = Field()
    patient_id: Optional[str] = Field()
# Usage: tools =[FhirPatientSearchTool()]
class FhirPatientSearchTool(StructuredTool):
    name = "patient search"
    description = """
    Searches FHIR server for a patient with available data.
    given: Given name of the patient.
    family: Family name of the patient.
    birth_date: Date of birth of the patient.
    patient_id: ID of the patient.
    All parameters are optional.
    Returns a list of all matching patient names if multiple patients are found.
    If no patient is found, returns the string "No patient found".
    If only one patient is found, returns the patient ID as a string.
    """
    args_schema: Type[BaseModel] = SearchInput



    def _run(
            self,
            given: str = None,
            family: str = None,
            birth_date: str = None,
            patient_id: str = None,
            run_manager: Optional[CallbackManagerForToolRun] = None
            ) -> Any:
        try:
            fhir_server = di["fhir_server"]
        except:
            return "Sorry, I cannot find a FHIR server to search."
        params = {}
        if patient_id:
            params["_id"] = patient_id
        else:
            if given:
                params["given"] = given
            if family:
                params["family"] = family
            if birth_date:
                params["birthdate"] = birth_date
        _url = "/Patient"
        try:
            _response = fhir_server.call_fhir_server(_url, params)
        except:
            return "Sorry I cannot find the answer as the FHIR server is not responding"
        return self.process_response(_response)

    async def _arun(
            self,
            given: str = None,
            family: str = None,
            birth_date: str = None,
            patient_id: str = None,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None
            ) -> Any:
        try:
            fhir_server = di["fhir_server"]
        except:
            return "Sorry, I cannot find a FHIR server to search."
        params = {}
        if patient_id:
            params["_id"] = patient_id
        else:
            if given:
                params["given"] = given
            if family:
                params["family"] = family
            if birth_date:
                params["birthdate"] = birth_date
        _url = "/Patient"
        try:
            _response = await fhir_server.async_call_fhir_server(_url, params)
        except:
            return "Sorry I cannot find the answer as the FHIR server is not responding or an implementation in not provided."
        return self.process_response(_response)


    def process_response(self, _response):
        _count = 0
        try:
            _count = _response["total"]
        except KeyError:
            _count = len(_response["entry"])
        except:
            #raise ValueError("FHIR server not responding")
            return "Sorry I cannot find the answer as the FHIR server is not responding."
        if _count == 0:
            return "No patient found"
        elif _count == 1:
            return "The patient id is {}".format(_response["entry"][0]["resource"]["id"])
        else:
            return "Sorry, I cannot find the answer as there are {} patients with this demographic.".format(_count)
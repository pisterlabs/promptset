import random
from typing import Optional, Type

from langchain.tools import BaseTool
import json

from pydantic import BaseModel, Field


class OpenAccountCheckInput(BaseModel):
    # Check the input for open an account
    type_of_account: str = Field(..., description="The type of the account the user wants to be open: Current or Savings.")
    name_of_client: str = Field(..., description="The name of the client")
    address_of_client: str = Field(..., description="The address of the client (street, city, postcode, country)")
    jobtitle_of_client: str = Field(..., description=" The job title of the client.")


class OpenAccountTool(BaseTool):
    name = "open_account"
    description = ("Use this tool when you need you need to create a new account."
                   "The user need to provide a account type, a name, a address and a job title")

    args_schema: Optional[Type[BaseModel]] = OpenAccountCheckInput

    def _run(self, type_of_account: str, name_of_client: str, address_of_client: str, jobtitle_of_client: str):
        block_card = self.open_account(type_of_account, name_of_client, address_of_client, jobtitle_of_client)
        return block_card

    def _arun(self, location: str, unit: str):
        raise NotImplementedError("Doesn't support async yet")

    def open_account(self, type_of_account, name_of_client, address_of_client, jobname_of_client):
        # Prepare the data as a JSON dictionary

        ### TODO: it is still necessary to create some validation function to verify each item
        account_details = {
            "type_of_account": type_of_account,
            "name_of_client": name_of_client,
            "address_of_client": address_of_client,
            "jobname_of_client": jobname_of_client
        }

        # Convert the dictionary to JSON format
        account_details_json = json.dumps(account_details)

        # Make the API call
        api_url = "https://openaccount.fake/"
        # response = requests.post(api_url, data=card_data_json)
        #
        # # Check the API response
        # if response.status_code == 200:
        #     response_data = response.json()
        #     if response_data.get("status") == "success":
        #         return "Card blocked successfully"
        #     else:
        #         return "Failed to block card"
        # else:
        #     return "API call failed"

        ## As I don't have an endpoint, randomly giving 50% to success and failure
        if random.randint(1, 10) % 2 == 0:
            return "success"
        else:
            return "failure"

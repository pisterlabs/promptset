""" Util that calls SendGrid. """
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Extra, root_validator

from prompts import (
    SEND_GRID_SEND_EMAIL,
)
from langchain.utils import get_from_dict_or_env


class SendGridApiWrapper(BaseModel):
    """ Wrapper for SendGrid API. """

    send_grid: Any  #: :meta private:

    operations: List[Dict] = [
        {
            "mode": "email_send",
            "name": "Send an email",
            "description": SEND_GRID_SEND_EMAIL,
        }
    ]

    class Config:
        """ Configuration for this pydantic object. """
        extra = Extra.forbid

    def list(self) -> List[Dict]:
        return self.operations

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """ Validate that api key and python package exists in environment. """
        api_key = get_from_dict_or_env(values, "api_key", "SEND_GRID_API_KEY")

        try:
            from sendgrid import SendGridAPIClient
        except ImportError:
            raise ImportError(
                "sendgrid is not installed. "
                "Please install it with `pip install sendgrid`"
            )

        send_grid = SendGridAPIClient(api_key)
        values["send_grid"] = send_grid

        return values


    def email_send(self, query: str) -> str:
        """ 
        Take a json object of {"channel": "channel_name", "message": "message"}
        and write it to the channel
        """
        try:
            import json
            from sendgrid.helpers.mail import Mail

            # Parse the json object
            params = json.loads(query)
            fields = dict(params)

            message = Mail(
                from_email=fields["from_email"],
                to_emails=fields["to_emails"],
                subject=fields["subject"],
                html_content=fields["body"])

            # Call the chat.postMessage method using the sendGrid WebClient
            resp = self.send_grid.send(message)
            if (resp.status_code != 200):
                raise Exception("Failed to send email")
                                
            return "Successfully sent email"
        except ImportError:
            raise ImportError(
                "json is not installed. " "Please install it with `pip install json`"
            )
        except Exception as e:
            print("Error: {}".format(e))
            raise Exception("Failed to send email")

    def run(self, mode: str, query: Optional[str]) -> str:
        """ Based on the mode from the caller, run the appropriate function. """
        if mode == "email_send":
            return self.email_send(query)
        else:
            raise ValueError(f"Got unexpected mode {mode}")

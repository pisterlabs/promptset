from typing import Optional
from langchain.tools import BaseTool
import logging


# Set up logging configuration
logging.basicConfig(
    filename="chat_service.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Define Tools
class UnemploymentRateTool(BaseTool):
    name = "Unemployment Rate"
    description = (
        "use this tool when you need to return current unemployment rate in US "
        "optionally given a time window. "
        "To use the tool you don't need to provide any of the parameters "
        "You can optionally provide two parameters: "
        "['starting', 'ending'] "
        "where 'starting' and 'ending' are must be in ISO 8601 date format strings always and without exceptions."
    )

    def _run(self, starting: Optional[str] = None, ending: Optional[str] = None):
        logging.info(f"Calling _run of Unemployment Rate tool.")
        return "Unemployment Rate Tool Result"

    def _arun(self, starting: Optional[str] = None, ending: Optional[str] = None):
        logging.info(f"Calling _arun of Unemployment Rate tool.")
        raise NotImplementedError("This tool does not support async")


class PayrollsTool(BaseTool):
    name = "Payrolls"
    description = (
        "use this tool when you need to return current nonfarm payrolls in US "
        "optionally given a time window. "
        "To use the tool you don't need to provide any of the parameters "
        "You can optionally provide two parameters: "
        "['starting', 'ending'] "
        "where 'starting' and 'ending' are must be in ISO 8601 date format strings always and without exceptions."
    )

    def _run(self, starting: Optional[str] = None, ending: Optional[str] = None):
        logging.info(f"Calling _run of Payrolls tool.")
        return "Payrolls Tool Result"

    def _arun(self, starting: Optional[str] = None, ending: Optional[str] = None):
        logging.info(f"Calling _arun of Payrolls tool.")
        raise NotImplementedError("This tool does not support async")


class CorporateTool(BaseTool):
    name = "Corporate"
    description = (
        "use this tool when you need to return some corporate metric of some US company "
        "To use the tool you must provide two parameters: "
        "['symbol', 'metric'] "
        "where 'symbol' is symbol of stock of given company on some us exchange and 'metric' is metric which we want to get."
    )

    def _run(self, symbol: Optional[str] = None, metric: Optional[str] = None):
        logging.info(f"Calling _run of Corporate tool.")
        return "Corporate Tool Result"

    def _arun(self, symbol: Optional[str] = None, metric: Optional[str] = None):
        logging.info(f"Calling _arun of Corporate tool.")
        raise NotImplementedError("This tool does not support async")

# /Tools/DownloadFile.py

import os
import requests
from instructor import OpenAISchema
from pydantic import Field
from Utilities.Config import WORKING_DIRECTORY
from Utilities.Log import Log, Debug, type
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Agency.Agency import Agency

class DownloadFile(OpenAISchema):
    """Used to download a file from the internet given a url"""

    url: str = Field(
        ...,
        description="The url to download the file from",
    )
    working_dir: str = Field(
        default=WORKING_DIRECTORY,
        description="The path to the directory to be write files to."
    )
    filename: str = Field(
        default=None,
        description="Specify a custom name to save the downloaded file as",
    )
    # todo remove and replace with timestamp prefix config
    overwrite: bool = Field(
        default=False,
        description="If true, will overwrite the file if it already exists."
    )

    def run(self, agency: 'Agency'):
        
        Log(type.ACTION, f"Downloading file from: {self.url}")
            
        # Set file name if agent did not supply one
        if self.filename is None:
            self.filename = self.url.split("/")[-1]

        # If file already exists, return message
        if os.path.exists(self.working_dir + self.filename):
            if self.overwrite:
                Log(type.ACTION, f"Overwriting file: {self.working_dir + self.filename}")
            else:
                result = f"File {self.working_dir + self.filename} already exists.\n"
                result += "Specify to overwrite if you this is intended, or\n"
                result += "increment the file version for a unique file name"
                Log(type.ERROR, result)
                return result
            
        try:
            with requests.get(self.url, stream=True) as r:
                r.raise_for_status()
                with open(self.working_dir + self.filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): 
                        f.write(chunk)
        except requests.exceptions.RequestException as e:
            result = f"Error downloading file: {e}"
            Log(type.ERROR, result)
            return result
        
        result = f"{self.url} has been downloaded to '{self.working_dir + self.filename}'"
        Log(type.ACTION, result)
        return result


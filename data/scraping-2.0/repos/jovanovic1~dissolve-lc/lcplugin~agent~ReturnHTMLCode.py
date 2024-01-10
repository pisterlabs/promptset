import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from pydantic import BaseModel, validator, Field
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools import BaseTool
from typing import Optional, Type

# load dotenv
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# You can provide a custom args schema to add descriptions or custom validation
llm = ChatOpenAI(openai_api_key=openai_api_key,model_name='gpt-3.5-turbo-16k-0613',temperature=0)

# html_code
html = """
<div class="filter-wrapper-platform">
	   <div class="plp-page-filters-v2 js-plp-filter-group pangea-inited" data-filter-id="platform">
	      <h2>Platform</h2>
	      <button class="toggle-visibility js-toggle-visibility" aria-label="Platform" aria-expanded="true"></button>
	      <div class="inner js-inner" style="display: block;">
	         <ul>
	            <li class="option">
                  <input name="platform-checkbox" type="checkbox" id="platform-checkbox-0" data-filter-group="platform" data-filter-val="windows" data-filter-title="Windows">
                  <label class="h6" for="platform-checkbox-0"><span class="filter-title">Windows</span>
                  </label>
	            </li>
	         
	            <li class="option">
                  <input name="platform-checkbox" type="checkbox" id="platform-checkbox-1" data-filter-group="platform" data-filter-val="mac" data-filter-title="Mac">
                  <label class="h6" for="platform-checkbox-1"><span class="filter-title">Mac</span>
                  </label>
	            </li>
	         
	            <li class="option">
                  <input name="platform-checkbox" type="checkbox" id="platform-checkbox-2" data-filter-group="platform" data-filter-val="linux" data-filter-title="Linux">
                  <label class="h6" for="platform-checkbox-2"><span class="filter-title">Linux</span>
                  </label>
	            </li>
	         
	            <li class="option">
                  <input name="platform-checkbox" type="checkbox" id="platform-checkbox-3" data-filter-group="platform" data-filter-val="chrome" data-filter-title="Chrome">
                  <label class="h6" for="platform-checkbox-3"><span class="filter-title">Chrome</span>
                  </label>
	            </li>
	         
	            <li class="option">
                  <input name="platform-checkbox" type="checkbox" id="platform-checkbox-4" data-filter-group="platform" data-filter-val="surface" data-filter-title="Surface">
                  <label class="h6" for="platform-checkbox-4"><span class="filter-title">Surface</span>
                  </label>
	            </li>
	         
	            <li class="option">
                  <input name="platform-checkbox" type="checkbox" id="platform-checkbox-5" data-filter-group="platform" data-filter-val="android" data-filter-title="Android">
                  <label class="h6" for="platform-checkbox-5"><span class="filter-title">Android</span>
                  </label>
	            </li>
	         
	            <li class="option">
                  <input name="platform-checkbox" type="checkbox" id="platform-checkbox-6" data-filter-group="platform" data-filter-val="ios" data-filter-title="iOS">
                  <label class="h6" for="platform-checkbox-6"><span class="filter-title">iOS</span>
                  </label>
	            </li>
	         
	            <li class="option">
                  <input name="platform-checkbox" type="checkbox" id="platform-checkbox-7" data-filter-group="platform" data-filter-val="ipados" data-filter-title="iPadOS">
                  <label class="h6" for="platform-checkbox-7"><span class="filter-title">iPadOS</span>
                  </label>
	            </li>
	         </ul>
	      </div>
"""

class ReturnHTMLCode(BaseTool):
    name = "fetch_html_code"
    description = "this will help fetch the html_code of the page user is currenly viewing"

    def _run(
        self, 
        query: str
    ) -> str:
        """Use the tool."""
        return html

    async def _arun(
        self, 
        query: str
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


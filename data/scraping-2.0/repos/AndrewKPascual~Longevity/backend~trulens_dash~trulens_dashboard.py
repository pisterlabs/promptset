import sys
sys.path.append('/mnt/d/old stuff/quest-copilot/backend')
from trulens_eval import Tru, TruCustomApp, Feedback, Select
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.openai import OpenAI as fOpenAI
import numpy as np
from trulens_eval.tru_custom_app import instrument
import os
from dotenv import load_dotenv
import importlib
importlib.invalidate_caches()

load_dotenv()

tru = Tru()

tru.run_dashboard()
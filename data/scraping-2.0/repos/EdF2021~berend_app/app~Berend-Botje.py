# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import streamlit as st
from PIL import Image

import openai
openai_api_key = os.getenv("OPENAI_API_KEY")

image = Image.open('images/producttoer.jpeg')

from streamlit.logger import get_logger
# import tiktoken
# import tiktoken_ext
image = Image.open('images/producttoer.jpeg')


from streamlit.logger import get_logger
# import tiktoken
# import tiktoken_ext
# import openai
# >>>>>>> c73e3ec134906f799647debf55cab1a1a04ad895

LOGGER = get_logger(__name__)



ENCODINGS = 'cl100k_base'

def run():
    st.set_page_config(
        page_title="Berend-Botje Skills",
        page_icon="👋",
        layout="wide",
        initial_sidebar_state="collapsed"
        
        )
    st.write("### Welkom bij Berend-Botje Skills 👋")
    st.image(image, caption=None, width=240, use_column_width=None, clamp=True, channels="RGB", output_format="png")
    
    st.sidebar.success("Kies één van Berend's skills")

    st.markdown(
        """
        ###### Berend-Botje is een slimme AI assistent die je kan helpen bij het uitvoeren van diverse werkzaamheden. 
        Afhankelijk van de werkzaamheden gebruikt Berend hiervoor 1 of meer  skills.  Deze skills maken gebruik van AI modellen van **openai** zoals ChatGPT.  Het verschil met ChatGPT is dat alle informatie binnen de omgeving van de gebruiker blijft!  
        ###### 👈 Voorbeelden.  Selecteer in de zijbalk een voorbeeld skill. van Berend-Botje!
        1. [De Lesplanner](Lesplan_Demo)
        2. [De Notulist](Mapping_Demo)
        2. [De Dataanalist](DataFrame_Demo)
        3. [De Datavormgever](Plotting_Demo)
        #### Meer weten?
        - Ga naar ....
        - 
        #### Het is werk under conder construction 
        - Lees bijvoorbeeld 
    """
    )


if __name__ == "__main__":
    run()

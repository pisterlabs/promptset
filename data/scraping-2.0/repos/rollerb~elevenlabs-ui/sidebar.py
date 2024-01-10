import os, el_audio, utils, datetime
import streamlit as st
from elevenlabs import Voice, User, set_api_key
from dataclasses import dataclass
from utils import log
from openai import OpenAI
from streamlit_js_eval import streamlit_js_eval

@dataclass
class SidebarData:
  el_key: str
  model_id: str
  voices: list[Voice]
  voice_names: list[str]
  enable_audio_editing: bool
  enable_normalization: bool
  stability: float
  simarlity_boost: float
  style: float
  join_gap: int  
  openai_api_key: str
  openai_model: str
  openai_temp: float
  openai_max_tokens: int

def get_usage_percent() -> dict:
  """Get the character usage percent from the Eleven Labs API."""
  user_info = User.from_api()
  percent = user_info.subscription.character_count / user_info.subscription.character_limit * 100
  resets = user_info.subscription.next_character_count_reset_unix
  resets = datetime.datetime.fromtimestamp(resets).strftime("%m/%d")
  return {
    "usage": percent,
    "reset": resets,
    "count": user_info.subscription.character_count,
    "limit": user_info.subscription.character_limit
  }

def get_voice_by_name(name: str, voices: list[Voice]) -> Voice:
  """Get a voice by the voice name."""
  name = utils.extract_name(name)
  return next((v for v in voices if v.name == name), None)

def get_cloned_voices(voices: list[Voice]) -> list[Voice]:
  """Get a list of cloned voices."""
  return [f"{v.name} (cloned)" for v in voices if v.category == "cloned"]

def voice_names_with_filter(
  voices: list[Voice], 
  gender: str, 
  age: str, 
  accent: str, 
  cloned: bool
) -> list[str]:
  """Get a list of voice names filtered by gender, age, accent, and cloned."""
  voice_names = []
  for v in voices:
    v_labels = v.labels
    v_gender = v_labels["gender"] if "gender" in v_labels else None
    v_age = v_labels["age"] if "age" in v_labels else None
    v_acccent = v_labels["accent"] if "accent" in v_labels else None
    match = True
    if gender and v_gender != gender:
      match = False
    if age and v_age != age:
      match = False
    if accent and v_acccent != accent:
      match = False
    if cloned:
      if v.category != "cloned":
        match = False
    if match:
      voice_names.append(f"{v.name} ({v.category})" if v.category == "cloned" else v.name)
  return voice_names

@st.cache_data
def get_models(openai_api_key: str) -> list[str]:
  """Get a list of OpenAI models."""
  client = OpenAI(api_key=openai_api_key)
  response = client.models.list()
  model_ids = [m.id for m in response.data]
  return sorted(model_ids)

def create_sidebar() -> SidebarData:
  """Create the streamlit sidebar."""
  with st.sidebar:    
    el_key = st.text_input("ElevenLabs API Key", os.getenv("ELEVENLABS_API_KEY"), type="password", key="el_key")        
    
    if el_key:
      set_api_key(el_key)          
          
      with st.expander("OpenAI Options"):
        openai_api_key = st.text_input("API Key _(optional)_", os.getenv("OPENAI_API_KEY"), type="password")
        if openai_api_key:
          openai_models = get_models(openai_api_key)
          try:
            gpt4_index = openai_models.index("gpt-3.5-turbo-16k")
          except:
            gpt4_index = 0
          openai_model = st.selectbox("Model", openai_models, index=gpt4_index)
          openai_temp = st.slider("Temperature", 0.0, 1.5, 1.3, 0.1,  help="The higher the temperature, the more creative the text.")
          openai_max_tokens = st.slider("Max Tokens", 1024, 10000, 3072, 1024, help="Check the official documentation on maximum token size for the selected model.")
        else:
          openai_model = None
          openai_temp = None
          openai_max_tokens = None
      
      with st.expander("Dialogue Options"):  
        models = el_audio.get_models()
        model_ids = [m.model_id for m in models]
        model_names = [m.name for m in models]
        try:
          turbo_model_index = model_names.index("Eleven Turbo v2")
        except:
          turbo_model_index = 0
        model_name = st.selectbox("Speech Model", model_names, index=turbo_model_index)
        if model_name:
          model_index = model_names.index(model_name)
          model_id = model_ids[model_index]   
        
        edit_audio = st.toggle(
          "Enable Audio Editing", 
          value=False,
          help="Enable audio editing for each dialogue line. This is disabled by default to increase performance."
        )
        normalize_audiobook = st.toggle(
          "Enable Normalization", 
          value=False, 
          help="Adjusts the final dialogue to meet the audiobook standards. The standard states that the audio should have a celing of -3dB and a range of -18dB to -23dB."
        )
                                  
        stability = st.slider(
          "Stability", 
          0.0, 
          1.0, 
          value=0.35, 
          help="Increasing stability will make the voice more consistent between re-generations, but it can also make it sounds a bit monotone. On longer text fragments we recommend lowering this value."
        )
        simarlity_boost = st.slider(
          "Clarity + Simalarity Enhancement",
          0.0,
          1.0,
          value=0.80,
          help="High enhancement boosts overall voice clarity and target speaker similarity. Very high values can cause artifacts, so adjusting this setting to find the optimal value is encouraged."
        )
        style = st.slider(
          "Style Exaggeration",
          0.0,
          1.0,
          value=0.0,
          help="High values are recommended if the style of the speech should be exaggerated compared to the uploaded audio. Higher values can lead to more instability in the generated speech. Setting this to 0.0 will greatly increase generation speed and is the default setting."
        )
        join_gap = st.slider(
          "Gap Between Dialogue",
          0,
          1000,
          step=10,
          value=200,
          help="The gap between spoken lines in milliseconds."
        )        
      
      with st.expander("Voice Explorer"):
        el_voices = el_audio.get_voices()
        el_voice_names = [f"{voice.name}{' (cloned)' if voice.category == 'cloned' else ''}" for voice in el_voices]
        el_voice_accents = set([voice.labels["accent"] for voice in el_voices if "accent" in voice.labels])
        el_voice_ages = set([voice.labels["age"] for voice in el_voices if "age" in voice.labels])
        el_voice_genders = set([voice.labels["gender"] for voice in el_voices if "gender" in voice.labels])
        el_voice_accents = sorted(el_voice_accents, key=lambda x: x.lower())
        
        el_voice_cloned = st.toggle("Cloned Voices Only", value=False)
        el_voice_gender = st.selectbox("Gender Filter", el_voice_genders, index=None)
        el_voice_age = st.selectbox("Age Filter", el_voice_ages, index=None)
        el_voice_accent = st.selectbox("Accent Filter", el_voice_accents, index=None)
        
        speaker_voice_names = voice_names_with_filter(
          el_voices, 
          el_voice_gender, 
          el_voice_age, 
          el_voice_accent,
          el_voice_cloned
        )
        el_voice = st.selectbox("Speaker", speaker_voice_names)
        if el_voice:
          el_voice_details = get_voice_by_name(el_voice, el_voices)
          el_voice_id = el_voice_details.voice_id
          
          # voice sample        
          if el_voice_details.preview_url:
            st.audio(el_voice_details.preview_url, format="audio/mp3")
        
          st.markdown(f"_Voice ID: {el_voice_id}_") 
      
      with st.expander("Usage"):
        usage = get_usage_percent()
        st.markdown(f"**Character Percent:** {usage['usage']:.1f}%")
        st.markdown(f"**Character Count:** {usage['count']:,}")
        st.markdown(f"**Character Limit:** {usage['limit']:,}")
        st.markdown(f"**Reset:** {usage['reset']}")
      
      clear_dialogue = st.button("Clear Dialogue", help=":warning: Clear everything and start over. :warning:", use_container_width=True)
      if clear_dialogue:
        streamlit_js_eval(js_expressions="parent.window.location.reload()")
            
      return SidebarData(
        el_key=el_key,
        model_id=model_id,
        voices=el_voices,
        voice_names=el_voice_names,
        enable_audio_editing=edit_audio,
        enable_normalization=normalize_audiobook,
        stability=stability,
        simarlity_boost=simarlity_boost,
        style=style,
        join_gap=join_gap,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        openai_temp=openai_temp,
        openai_max_tokens=openai_max_tokens
      )
    else:
      return SidebarData(
        el_key="",
        model_id="",
        voices=[],
        voice_names=[],
        enable_audio_editing=False,
        enable_normalization=False,
        stability=0.35,
        simarlity_boost=0.80,
        style=0.0,
        join_gap=200,
        openai_api_key="",
        openai_model="",
        openai_temp=1.5,
        openai_max_tokens=4096
      )   

  
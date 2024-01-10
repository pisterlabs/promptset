import os
import openai
import streamlit as st
from dotenv import load_dotenv
import time
load_dotenv()

prompt = "Tu es un rédacteur de QCM pour la première année de médecine expert. Tu rédiges des QCM depuis de nombreuses années et tu sais parfaitement reformuler des annales de QCM pour formuler de nouveaux QCM. Pour rédiger de nouveaux QCM il existe plusieurs méthodes pour créer de fausses propositions, dont notamment :\n  - Les fausses négations\n - Les inversions de terme comme mitose/meiose, altérer/modifier\n - Les paronymes\n - Les mauvaises données chiffrées\n - Les propositions incohérentes\n - Les propositions fantaisistes\n - Les illogismes\n - Les anachronismes\n Ta tâche est maintenant de rédiger de 5 nouveaux QCMs à partir des annales données. Ne fais pas de hors sujets. N’invente pas de notion, n’utilise pas de notions non données dans les annales. Sois précis. Utilise le ton de rédaction utilisé dans les annales données. Ne te répète pas entre les différentes propositions. Donne une correction pour chaque QCM à chaque item faux. Chaque QCM doit avoir entre 1 et 5 réponses justes. Structure ta réponse au format markdown. Ne donne pas de numérotation de QCM (type (Q6, 21-22) ou QCM X)."
def writer(annales):
    for attempt in range(st.session_state["max_retries"]):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                temperature=st.session_state.get("TEMPERATURE"),
                max_tokens=st.session_state.get("MAX_TOKENS"),
                top_p=1,
                frequency_penalty=st.session_state.get("FREQUENCY_PENALTY"),
                presence_penalty=st.session_state.get("PRESENCE_PENALTY"),
                messages=[{"role": "system", "content": prompt},
                                {"role": "user", "content": "[Annales :]\n" + annales }]
            )
            st.session_state["total_tokens"] = st.session_state["total_tokens"] + response["usage"]["total_tokens"]
            st.session_state["completion_tokens"] = st.session_state["completion_tokens"] + response["usage"]['completion_tokens']
            st.session_state["prompt_tokens"] = st.session_state["prompt_tokens"] + response["usage"]['prompt_tokens']
            return response["choices"][0]["message"]["content"]
            break

        except openai.error.Timeout as e:
            if attempt < st.session_state["max_retries"] - 1:  # ne pas attendre après la dernière tentative 
                st.write(f"OpenAI API request timed out: {e}, retrying...")
                time.sleep(st.session_state["wait_time"])
            else:
                st.write("Max retries reached. Aborting.")
                st.session_state["error"] = 1

        except openai.error.APIError as e:
            if attempt < st.session_state["max_retries"] - 1:  # ne pas attendre après la dernière tentative 
                st.write(f"OpenAI API returned an API Error: {e}, retrying...")
                time.sleep(st.session_state["wait_time"])
            else:
                st.write("Max retries reached. Aborting.")
                st.session_state["error"] = 1
        
        except openai.error.APIConnectionError as e:
            if attempt < st.session_state["max_retries"] - 1:  # ne pas attendre après la dernière tentative 
                st.write(f"OpenAI API request failed to connect: {e}, retrying...")
                time.sleep(st.session_state["wait_time"])
            else:
                st.write("Max retries reached. Aborting.")
                st.session_state["error"] = 1
        
        except openai.error.InvalidRequestError as e:
            if attempt < st.session_state["max_retries"] - 1:  # ne pas attendre après la dernière tentative 
                st.write(f"OpenAI API request was invalid: {e}, retrying...")
                time.sleep(st.session_state["wait_time"])
            else:
                st.write("Max retries reached. Aborting.")
                st.session_state["error"] = 1

        except openai.error.AuthenticationError as e:
                st.write(f"OpenAI API request was not authorized: {e}, retrying...")
                st.write("Please change your OpenAI key.")
                st.session_state["error"] = 1
                pass
        
        except openai.error.PermissionError as e:
            if attempt < st.session_state["max_retries"] - 1:  # ne pas attendre après la dernière tentative 
                st.write(f"OpenAI API request was not permitted: {e}, retrying...")
                time.sleep(st.session_state["wait_time"])
            else:
                st.write("Max retries reached. Aborting.")
                st.session_state["error"] = 1
        
        except openai.error.RateLimitError as e:
            if attempt < st.session_state["max_retries"] - 1:  # ne pas attendre après la dernière tentative 
                st.write(f"OpenAI API request exceeded rate limit: {e}, retrying...")
                time.sleep(st.session_state["wait_time"])
            else:
                st.write("Max retries reached. Aborting.")
                st.session_state["error"] = 1
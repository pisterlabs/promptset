import openai
import streamlit as st
from utils import Features

class Writing:

    features = Features.Features()

    def __init__(self):
        openai.api_key = st.secrets["OPENAI_KEY"]

    def write(self, dyn_prompt, model, temp=0.73, top_p=1.0, tokens=500, freq_pen=1.73, pres_pen=0.43, stop=["END"]):
        if (dyn_prompt ==''):
            st.error('Error: No prompt provided')
            return ''
        else:
            with st.spinner('Querying OpenAI using ' + model + '...'):
                # fine-tuned models requires model parameter, whereas other models require engine parameter
                model_param = (
                    {"model": model}
                    if ":" in model
                       and model.split(":")[1].startswith("ft")
                    else {"engine": model}
                )

                try:
                    response = openai.Completion.create(
                        prompt=dyn_prompt,
                        temperature=temp,
                        max_tokens=tokens,
                        top_p=top_p,
                        frequency_penalty=freq_pen,
                        presence_penalty=pres_pen,
                        stop=stop,
                        **model_param)
                    response = response['choices'][0]['text']
                    return response
                except Exception as oops:
                    st.error("Completion Error: " + str(oops))
                    # return "Completion Error: " + str(oops)

    # TODO: make this work later. Force the model for now.
    def edit(self, instruction, dyn_prompt, model="text-davinci-edit-002", temp=0.73, top_p=1.0, tokens=500, freq_pen=1.73, pres_pen=0.43, stop=["END", "Scene:", "[Scene"]):
        if (dyn_prompt =='' or instruction == ''):
            st.error('Error: No prompt or instruction provided')
            return ''
        else:
            with st.spinner('Querying OpenAI using ' + model + '...'):
                # fine-tuned models requires model parameter, whereas other models require engine parameter
                model_param = (
                    {"model": model}
                    if ":" in model
                       and model.split(":")[1].startswith("ft")
                    else {"engine": model}
                )

                try:
                    response = openai.Completion.edit(
                        input=dyn_prompt,
                        instruction=instruction,
                        temperature=temp,
                        max_tokens=tokens,
                        top_p=top_p,
                        frequency_penalty=freq_pen,
                        presence_penalty=pres_pen,
                        stop=stop,
                        **model_param)
                    response = response['choices'][0]['text']
                    return response
                except Exception as oops:
                    st.error("Completion Error: " + str(oops))
                    return "Completion Error: " + str(oops)

    def get_tuned_content(self, prompt, model):
        try:
            p = self.features.get_prompt(st.session_state.feat)
            p = p.format(prompt)
            return self.write(p, model)
        except Exception as oops:
            st.error('ERROR in get_tuned_content function: ' + str(oops))

    def get_generic_content(self, prompt):
        try:
            p = self.features.get_prompt(st.session_state.feat)
            p = p.format(prompt)
            return self.write(p, "text-davinci-001")
        except Exception as oops:
            st.error('ERROR in get_generic_content function: ' + str(oops))

    def completeDavinci(self, prompt, temp=0.73):
        try:
            return self.write(prompt, "text-davinci-001", temp=temp)
        except Exception as oops:
            st.error('ERROR in completeDavinci function: ' + str(oops))

    def completeModel(self, prompt, model, temp=0.73):
        try:
            return self.write(prompt, model, temp=temp)
        except Exception as oops:
            st.error('ERROR in get_generic function: ' + str(oops))


    def getModels(self):
        models = []
        try:
            model_list = openai.Model.list()
            for row in model_list.data:
                if (row["owned_by"] != "openai" and row["owned_by"] != "system"):
                    models.append(row.id)


            models.append("text-davinci-001")
            models.append("text-curie-001")
            # inject insert capability
            models.append("text-davinci-002")
            # edit capability
            models.append("text-davinci-edit-001")

            return models
        except Exception as oops:
            st.error('ERROR in getModels function: ' + str(oops))

    def generate_campaign(self, concept, model):
        try:
            idea_base = self.features.get_prompt('Campaign')
            p = idea_base.format(concept)
            return self.write(p, model, temp=0.4)
        except Exception as oops:
            st.error('ERROR in generate_concept function: ' + str(oops))

    def generate_toc(self, campaign, model):
        try:
            toc_base = self.features.get_prompt('Table of Contents')
            p = toc_base.format(campaign)
            return self.write(p, model, temp=0.13)
        except Exception as oops:
            st.error('ERROR in generate_toc function: ' + str(oops))

    def generate_campaign_titles(self, campaign, model):
        try:
            title_base = self.features.get_prompt('Campaign Names')
            p = title_base.format(campaign)
            titles = self.write(p, model, temp=0.5)
            results = []
            for title in titles.split('\n'):
                if (title != '') and len(title) > 2:
                    results.append(title.strip())

            return results
        except Exception as oops:
            st.error('ERROR in generate_toc function: ' + str(oops))

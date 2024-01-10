import gradio as gr
import transformers

import os
import sys
import openai
from openai import Completion as complete
openai.api_key = "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"

with gr.Blocks() as demo:

    rubric = {
        "Row0": ["",
                    "Très bien",
                    "Bien",
                    "À améliorer",
                    ],
        "Row1": ["Compréhension de la tâche à réaliser",
                    "Élabore un plan de travail structuré et détaillé, comprenant les étapes divisées en tâches, l’échéancier et le nom des responsables.",
                    "Élaborer un plan de travail en n’organisant que les éléments essentiels (les tâches et le nom des responsables).",
                    "Élabore un plan de travail incomplet et imprécis.",
                    ],
        "Row2": ["Exécution de la tâche",
                    "Collecte toutes les données : montant qu’il a amassé, celui des autres élèves de son groupe ainsi que celui des autres groupes de son cycle. Collecte toutes les données de façon cohérente.",
                    "Collecte le montant qu’il a amassé et celui des autres élèves de son groupe. Organise les données de façon claire.",
                    "Collecte seulement le montant qu’il a amassé. Organise les données de façon confuse.",
                    ],
        "Row3": ["Analyse du déroulement de la démarche",
                    "Complète sa réflexion soigneusement et propose une autre méthode de travail pertinente.",
                    "Complète brièvement sa réflexion et propose une autre méthode de travail plus ou moins pertinente.",
                    "Complète sa réflexion à la hâte et sans reconnaître d’autres méthodes de travail.",
                    ],
    }

    def generate_feedback(rubric_extraction):
        """
        Generate the feedback for the rubric by calling davinci-003.
        """
        prompt = "Voici le texte extrait de la grille de correction : " + \
                str(rubric_extraction) + \
                "Voici la rétroaction générée par le modèle qui s'adresse à l'étudiant : "
        
        try:
            completion = complete.create(model="text-davinci-003", prompt=prompt, max_tokens=500)
            return str(completion.choices[0].text)
        except Exception as e:
            # return str(e)
            # return python version
            return str(sys.version)


    rubric_inputs = {}

    # 'x => window.getSelection().toString()'
    # rewrite the above to append the selected text to the input
    # and then return the result
    # 'x => x + window.getSelection().toString()'

    def append_feedback(selected_text, feedback):
        return selected_text, selected_text + feedback

    # def return_total_feedback(total_feedback=total_feedback):
    #     return total_feedback

    def update_value(value):
        return gr.Textbox.update(lines=2, visible=True, value="Short story: ")

    def reset_val():
        return ""

    def format_cumul_feedback(crit1, crit2, crit3, rubric=rubric):
        
        # get the criteria
        criteria = [rubric[row][0] for row in rubric if row != "Row0"]

        # return a string with each criteria and its feedback
        first_criteria_feedback = criteria[0] + ": " + crit1 + "\n"
        second_criteria_feedback = criteria[1] + ": " + crit2 + "\n"
        third_criteria_feedback = criteria[2] + ": " + crit3 + "\n"

        return first_criteria_feedback + second_criteria_feedback + third_criteria_feedback



    gr.Markdown("# AI tools for course production!")

    with gr.Tab("Text Tools"):
        with gr.Tab("Quiz Generator"):
            inp = gr.Textbox(placeholder="Put text here.")
            out = gr.Textbox()

        with gr.Tab("Rubric Generator 1"):
            with gr.Row():
                for criteria in rubric["Row0"]:
                    gr.Markdown(criteria)

            for keys in rubric:
                if keys != "Row0":
                    with gr.Row():
                        for criteria in rubric[keys]:
                            rubric_inputs[criteria] = gr.Markdown(criteria, interactive=False)
            with gr.Row():
                # with gr.Column(scale=1):
                #     sel1 = gr.Textbox(placeholder="Current selection.", label = "Selection 1")
                with gr.Column(scale=20):
                    crit1 = gr.Textbox(placeholder="Sélection pour le premier critère.", label = "Compréhension de la tâche à réaliser")
                with gr.Column(scale=0.5):
                    crit1_button_sel = gr.Button("Extraire le texte.")
                    crit1_button_reset = gr.Button("Réinitialiser la sélection.")
            with gr.Row(equal_height=True):
                # with gr.Column(scale=1):
                #     sel2 = gr.Textbox(placeholder="Current selection.")
                with gr.Column(scale=20):
                    crit2 = gr.Textbox(placeholder="Sélection pour le deuxième critère.", label = "Exécution de la tâche")
                with gr.Column(scale=0.5):
                    crit2_button_sel = gr.Button("Extraire le texte.")
                    crit2_button_reset = gr.Button("Réinitialiser la sélection.")
            with gr.Row(equal_height=True):
                # with gr.Column(scale=1):
                #     sel3 = gr.Textbox(placeholder="Current selection.")
                with gr.Column(scale=20):
                    crit3 = gr.Textbox(placeholder="Sélection pour le troisième critère.", label = "Analyse du déroulement de la démarche")
                with gr.Column(scale=0.5):
                    crit3_button_sel = gr.Button("Extraire le texte.")
                    crit3_button_reset = gr.Button("Réinitialiser la sélection.")
            
            with gr.Row(equal_height=True):
                cumul_btn = gr.Button("Cumuler le texte extrait.", variant="primary")
                generate_btn = gr.Button("Générer la rétroaction.", variant="primary")        
            with gr.Row(equal_height=True):
                cumul_out = gr.Textbox(placeholder="Sélection totale.", label = "Sélection des critères")
                with gr.Column():
                    generated_feedback = gr.Textbox(placeholder="Rétroaction générée.", label = "Rétroaction générée")
                    hattie_question_1 = gr.Textbox(placeholder="Première question de Hattie.", label = "Rétroaction générée")


            crit1_button_sel.click(fn= None,
                                inputs=crit1, 
                                outputs=crit1, 
                                _js='sel => sel + window.getSelection().toString() + ","')

            crit1_button_reset.click(fn= reset_val, inputs=None, outputs=crit1)


            crit2_button_sel.click(fn= None,
                                inputs=crit2, 
                                outputs=crit2,
                                _js='sel => sel + window.getSelection().toString() + ","')
            
            # crit2_button_sel.click(fn= None,
            #                        inputs=None, 
            #                        outputs=sel2, 
            #                        _js='sel => window.getSelection().toString()')

            crit2_button_reset.click(fn= reset_val, inputs=None, outputs=crit2)


            crit3_button_sel.click(fn= None,
                                inputs=crit3, 
                                outputs=crit3,
                                _js='sel => sel + window.getSelection().toString() + ","')
            
            # crit3_button_sel.click(fn= None,
            #                        inputs=None, 
            #                        outputs=sel3, 
            #                        _js='sel => window.getSelection().toString()')

            crit3_button_reset.click(fn= reset_val, inputs=None, outputs=crit3)
            
            cumul_btn.click(fn= format_cumul_feedback,
                            inputs=[crit1, crit2, crit3],
                            outputs=cumul_out)

            generate_btn.click(fn=generate_feedback,
                                inputs=cumul_out,
                                outputs=generated_feedback)
            # for inpts in rubric_inputs:
            #     # rubric_inputs[inpts].click(fn=None, inputs=rubric_inputs[inpts], outputs=sel1, _js='x => window.getSelection().toString()')
            #     # change sel1 according to the selected text dynamically

            #     # crit1_button.click(fn = join_strings, inputs=[sel1,crit1], outputs=crit1)

            #     crit2_button.click(fn=None, inputs=rubric_inputs[inpts], outputs=sel2, _js='x => window.getSelection().toString()')
            #     crit3_button.click(fn=None, inputs=rubric_inputs[inpts], outputs=sel3, _js='x => window.getSelection().toString()')
            #     # btn.click(fn = lambda x, y: x.join(y), inputs=[cumul_out, out], outputs=cumul_out)

        with gr.Tab("Summarizer"):
            inp = gr.Textbox(placeholder="Put text here.")
            out = gr.Textbox()
        with gr.Tab("Translator"):
            inp = gr.Textbox(placeholder="Put text here.")
            out = gr.Textbox()
        with gr.Tab("Text-to-Speech"):
            inp = gr.Textbox(placeholder="Put text here.")
            out = gr.Textbox()
        with gr.Tab("chatbot"):
            inp = gr.Textbox(placeholder="Put text here.")
            out = gr.Textbox()

        inp.change(fn=lambda x: x,
                inputs=inp, 
                outputs=out)

    with gr.Tab("Video Tools"):
        inp = gr.Video(placeholder="Put video here.")
        out = gr.Textbox()

    with gr.Tab("Audio Tools"):
        inp = gr.Text(placeholder="Put audio here.")
        out = gr.Textbox()

        inp.change(fn=lambda x: x,
                inputs=inp, 
                outputs=out)

    with gr.Tab("Image Tools"):
        inp = gr.Image(placeholder="Put image here.")
        out = gr.Textbox()

        inp.change(fn=lambda x: x,
                inputs=inp, 
                outputs=out)    

if __name__ == "__main__":
    demo.launch(debug=True)    

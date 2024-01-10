import ast

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

from optim.llm_loader import main as llm_loader

# LLM prompt template
# I have separated the prompt template into four parts to make it easier to read and reduce the number of characters.
SYMPTOMS1 = """
Your goal is to extract structured information from the user's input that matches the form described 
below. When extracting information please make sure it matches the type information exactly. Do not add any 
attributes that do not appear in the schema shown below.

itching: boolean with value True or False (Does the patient have the symptom of itching?)
skin_rash: boolean with value True or False (Does the patient have the symptom of skin rash?)
nodal_skin_eruptions: boolean with value True or False (Does the patient have the symptom of nodal skin eruptions?)
continuous_sneezing: boolean with value True or False (Does the patient have the symptom of continuous sneezing?)
shivering: boolean with value True or False (Does the patient have the symptom of shivering?)
chills: boolean with value True or False (Does the patient have the symptom of chills?)
joint_pain: boolean with value True or False (Does the patient have the symptom of joint pain?)
stomach_pain: boolean with value True or False (Does the patient have the symptom of stomach pain?)
acidity: boolean with value True or False (Does the patient have the symptom of acidity?)
ulcers_on_tongue: boolean with value True or False (Does the patient have the symptom of ulcers on tongue?)
muscle_wasting: boolean with value True or False (Does the patient have the symptom of muscle wasting?)
vomiting: boolean with value True or False (Does the patient have the symptom of vomiting?)
burning_micturition: boolean with value True or False (Does the patient have the symptom of burning micturition?)
spotting_urination: boolean with value True or False (Does the patient have the symptom of spotting urination?)
fatigue: boolean with value True or False (Does the patient have the symptom of fatigue?)
weight_gain: boolean with value True or False (Does the patient have the symptom of weight gain?)
anxiety: boolean with value True or False (Does the patient have the symptom of anxiety?)
cold_hands_and_feets: boolean with value True or False (Does the patient have the symptom of cold hands and feet?)
mood_swings: boolean with value True or False (Does the patient have the symptom of mood swings?)
weight_loss: boolean with value True or False (Does the patient have the symptom of weight loss?)
restlessness: boolean with value True or False (Does the patient have the symptom of restlessness?)
lethargy: boolean with value True or False (Does the patient have the symptom of lethargy?)
patches_in_throat: boolean with value True or False (Does the patient have the symptom of patches in throat?)

Please output the extracted information in DICT format. Do not output anything except for the extracted information. 
Do not add any clarifying information. Do not add any fields that are not in the schema. If the text contains 
attributes that do not appear in the schema, please ignore them. All output must be in JSON format and follow the 
schema specified above.



Input: Please extract the patient's different symptoms from this conversation.

 Please confirm the 
        presence of these symptoms with a Boolean (True or False).

 Do NOT include any additional information. The 
        output MUST follow the above scheme. Do NOT add any additional columns that are not included in the scheme.
{conversation}
Output: 
"""

SYMPTOMS2 = """
Your goal is to extract structured information from the user's input that matches the form described 
below. When extracting information please make sure it matches the type information exactly. Do not add any 
attributes that do not appear in the schema shown below.

irregular_sugar_level: boolean with value True or False (Does the patient have the symptom of irregular sugar level?) 
cough: boolean with value True or False (Does the patient have the symptom of cough?) high_fever: boolean with value 
True or False (Does the patient have the symptom of high fever?) sunken_eyes: boolean with value True or False (Does 
the patient have the symptom of sunken eyes?) breathlessness: boolean with value True or False (Does the patient have 
the symptom of breathlessness?) sweating: boolean with value True or False (Does the patient have the symptom of 
sweating?) dehydration: boolean with value True or False (Does the patient have the symptom of dehydration?) 
indigestion: boolean with value True or False (Does the patient have the symptom of indigestion?) headache: boolean 
with value True or False (Does the patient have the symptom of headache?) yellowish_skin: boolean with value True or 
False (Does the patient have the symptom of yellowish skin?) dark_urine: boolean with value True or False (Does the 
patient have the symptom of dark urine?) nausea: boolean with value True or False (Does the patient have the symptom 
of nausea?) loss_of_appetite: boolean with value True or False (Does the patient have the symptom of loss of 
appetite?) pain_behind_the_eyes: boolean with value True or False (Does the patient have the symptom of pain behind 
the eyes?) back_pain: boolean with value True or False (Does the patient have the symptom of back pain?) 
constipation: boolean with value True or False (Does the patient have the symptom of constipation?) abdominal_pain: 
boolean with value True or False (Does the patient have the symptom of abdominal pain?) diarrhoea: boolean with value 
True or False (Does the patient have the symptom of diarrhoea?) mild_fever: boolean with value True or False (Does 
the patient have the symptom of mild fever?) yellow_urine: boolean with value True or False (Does the patient have 
the symptom of yellow urine?) yellowing_of_eyes: boolean with value True or False (Does the patient have the symptom 
of yellowing of eyes?) acute_liver_failure: boolean with value True or False (Does the patient have the symptom of 
acute liver failure?) fluid_overload: boolean with value True or False (Does the patient have the symptom of fluid 
overload?) swelling_of_stomach: boolean with value True or False (Does the patient have the symptom of swelling of 
stomach?) swelled_lymph_nodes: boolean with value True or False (Does the patient have the symptom of swelled lymph 
nodes?) malaise: boolean with value True or False (Does the patient have the symptom of malaise?) 
blurred_and_distorted_vision: boolean with value True or False (Does the patient have the symptom of blurred and 
distorted vision?) phlegm: boolean with value True or False (Does the patient have the symptom of phlegm?) 
throat_irritation: boolean with value True or False (Does the patient have the symptom of throat irritation?) 
redness_of_eyes: boolean with value True or False (Does the patient have the symptom of redness of eyes?) 
sinus_pressure: boolean with value True or False (Does the patient have the symptom of sinus pressure?) runny_nose: 
boolean with value True or False (Does the patient have the symptom of runny nose?) congestion: boolean with value 
True or False (Does the patient have the symptom of congestion?) chest_pain: boolean with value True or False (Does 
the patient have the symptom of chest pain?) weakness_in_limbs: boolean with value True or False (Does the patient 
have the symptom of weakness in limbs?) fast_heart_rate: boolean with value True or False (Does the patient have the 
symptom of fast heart rate?)

Please output the extracted information in DICT format. Do not output anything except for the extracted information. 
Do not add any clarifying information. Do not add any fields that are not in the schema. If the text contains 
attributes that do not appear in the schema, please ignore them. All output must be in JSON format and follow the 
schema specified above.



Input: Please extract the patient's different symptoms from this conversation.

 Please confirm the 
        presence of these symptoms with a Boolean (True or False).

 Do NOT include any additional information. The 
        output MUST follow the above scheme. Do NOT add any additional columns that are not included in the scheme.
{conversation}
Output: 
"""

SYMPTOMS3 = """
Your goal is to extract structured information from the user's input that matches the form described 
below. When extracting information please make sure it matches the type information exactly. Do not add any 
attributes that do not appear in the schema shown below.

pain_during_bowel_movements: boolean with value True or False (Does the patient have the symptom of pain during bowel 
movements?) pain_in_anal_region: boolean with value True or False (Does the patient have the symptom of pain in anal 
region?) bloody_stool: boolean with value True or False (Does the patient have the symptom of bloody stool?) 
irritation_in_anus: boolean with value True or False (Does the patient have the symptom of irritation in anus?) 
neck_pain: boolean with value True or False (Does the patient have the symptom of neck pain?) dizziness: boolean with 
value True or False (Does the patient have the symptom of dizziness?) cramps: boolean with value True or False (Does 
the patient have the symptom of cramps?) bruising: boolean with value True or False (Does the patient have the 
symptom of bruising?) obesity: boolean with value True or False (Does the patient have the symptom of obesity?) 
swollen_legs: boolean with value True or False (Does the patient have the symptom of swollen legs?) 
swollen_blood_vessels: boolean with value True or False (Does the patient have the symptom of swollen blood vessels?) 
puffy_face_and_eyes: boolean with value True or False (Does the patient have the symptom of puffy face and eyes?) 
enlarged_thyroid: boolean with value True or False (Does the patient have the symptom of enlarged thyroid?) 
brittle_nails: boolean with value True or False (Does the patient have the symptom of brittle nails?) 
swollen_extremeties: boolean with value True or False (Does the patient have the symptom of swollen extremeties?) 
excessive_hunger: boolean with value True or False (Does the patient have the symptom of excessive hunger?) 
extra_marital_contacts: boolean with value True or False (Does the patient have the symptom of extra marital 
contacts?) drying_and_tingling_lips: boolean with value True or False (Does the patient have the symptom of drying 
and tingling lips?) slurred_speech: boolean with value True or False (Does the patient have the symptom of slurred 
speech?) knee_pain: boolean with value True or False (Does the patient have the symptom of knee pain?) 
hip_joint_pain: boolean with value True or False (Does the patient have the symptom of hip joint pain?) 
muscle_weakness: boolean with value True or False (Does the patient have the symptom of muscle weakness?) stiff_neck: 
boolean with value True or False (Does the patient have the symptom of stiff neck?) swelling_joints: boolean with 
value True or False (Does the patient have the symptom of swelling joints?) movement_stiffness: boolean with value 
True or False (Does the patient have the symptom of movement stiffness?) spinning_movements: boolean with value True 
or False (Does the patient have the symptom of spinning movements?) loss_of_balance: boolean with value True or False 
(Does the patient have the symptom of loss of balance?) unsteadiness: boolean with value True or False (Does the 
patient have the symptom of unsteadiness?) weakness_of_one_body_side: boolean with value True or False (Does the 
patient have the symptom of weakness of one body side?) loss_of_smell: boolean with value True or False (Does the 
patient have the symptom of loss of smell?) bladder_discomfort: boolean with value True or False (Does the patient 
have the symptom of bladder discomfort?) foul_smell_ofurine: boolean with value True or False (Does the patient have 
the symptom of foul smell of urine?) continuous_feel_of_urine: boolean with value True or False (Does the patient 
have the symptom of continuous feel of urine?) passage_of_gases: boolean with value True or False (Does the patient 
have the symptom of passage of gases?) internal_itching: boolean with value True or False (Does the patient have the 
symptom of internal itching?) toxic_look_typhos: boolean with value True or False (Does the patient have the symptom 
of toxic look typhos?) depression: boolean with value True or False (Does the patient have the symptom of 
depression?) irritability: boolean with value True or False (Does the patient have the symptom of irritability?) 
muscle_pain: boolean with value True or False (Does the patient have the symptom of muscle pain?) altered_sensorium: 
boolean with value True or False (Does the patient have the symptom of altered sensorium?) red_spots_over_body: 
boolean with value True or False (Does the patient have the symptom of red spots over body?) belly_pain: boolean with 
value True or False (Does the patient have the symptom of belly pain?) abnormal_menbooluation: boolean with value 
True or False (Does the patient have the symptom of abnormal menbooluation?) dischromic_patches: boolean with value 
True or False (Does the patient have the symptom of dischromic patches?) watering_from_eyes: boolean with value True 
or False (Does the patient have the symptom of watering from eyes?) increased_appetite: boolean with value True or 
False (Does the patient have the symptom of increased appetite?) polyuria: boolean with value True or False (Does the 
patient have the symptom of polyuria?) family_history: boolean with value True or False (Does the patient have the 
symptom of family history?) mucoid_sputum: boolean with value True or False (Does the patient have the symptom of 
mucoid sputum?) rusty_sputum: boolean with value True or False (Does the patient have the symptom of rusty sputum?) 
lack_of_concentration: boolean with value True or False (Does the patient have the symptom of lack of concentration?) 
visual_disturbances: boolean with value True or False (Does the patient have the symptom of visual disturbances?) 
receiving_blood_transfusion: boolean with value True or False (Does the patient have the symptom of receiving blood 
transfusion?

Please output the extracted information in DICT format. Do not output anything except for the extracted information. 
Do not add any clarifying information. Do not add any fields that are not in the schema. If the text contains 
attributes that do not appear in the schema, please ignore them. All output must be in JSON format and follow the 
schema specified above.



Input: Please extract the patient's different symptoms from this conversation.

 Please confirm the 
        presence of these symptoms with a Boolean (True or False).

 Do NOT include any additional information. The 
        output MUST follow the above scheme. Do NOT add any additional columns that are not included in the scheme.
{conversation}
Output: 
"""

SYMPTOMS4 = """
Your goal is to extract structured information from the user's input that matches the form described 
below. When extracting information please make sure it matches the type information exactly. Do not add any 
attributes that do not appear in the schema shown below.

coma: boolean with value True or False (Does the patient have the symptom of coma?) stomach_bleeding: boolean with 
value True or False (Does the patient have the symptom of stomach bleeding?) distention_of_abdomen: boolean with 
value True or False (Does the patient have the symptom of distention of abdomen?) history_of_alcohol_consumption: 
boolean with value True or False (Does the patient have the symptom of history of alcohol consumption?) 
receiving_unsterile_injections: boolean with value True or False (Does the patient have the symptom of receiving 
unsterile injections?) blood_in_sputum: boolean with value True or False (Does the patient have the symptom of blood 
in sputum?) prominent_veins_on_calf: boolean with value True or False (Does the patient have the symptom of prominent 
veins on calf?) palpitations: boolean with value True or False (Does the patient have the symptom of palpitations?) 
painful_walking: boolean with value True or False (Does the patient have the symptom of painful walking?) 
pus_filled_pimples: boolean with value True or False (Does the patient have the symptom of pus filled pimples?) 
blackheads: boolean with value True or False (Does the patient have the symptom of blackheads?) scurring: boolean 
with value True or False (Does the patient have the symptom of scurring?) red_sore_around_nose: boolean with value 
True or False (Does the patient have the symptom of red sore around nose?) skin_peeling: boolean with value True or 
False (Does the patient have the symptom of skin peeling?) silver_like_dusting: boolean with value True or False (
Does the patient have the symptom of silver like dusting?) small_dents_in_nails: boolean with value True or False (
Does the patient have the symptom of small dents in nails?) inflammatory_nails: boolean with value True or False (
Does the patient have the symptom of inflammatory nails?) blister: boolean with value True or False (Does the patient 
have the symptom of blister?) yellow_crust_ooze: boolean with value True or False (Does the patient have the symptom 
of yellow crust ooze?) prognosis: boolean with value True or False (Does the patient have the symptom of prognosis?)

Please output the extracted information in DICT format. Do not output anything except for the extracted information. 
Do not add any clarifying information. Do not add any fields that are not in the schema. If the text contains 
attributes that do not appear in the schema, please ignore them. All output must be in JSON format and follow the 
schema specified above.



Input: Please extract the patient's different symptoms from this conversation.

 Please confirm the 
        presence of these symptoms with a Boolean (True or False).

 Do NOT include any additional information. The 
        output MUST follow the above scheme. Do NOT add any additional columns that are not included in the scheme.
{conversation}
Output: 
"""


def main(conversation: str) -> dict:
    """
    Runs the main extraction process for a given conversation.

    Loads the LLM model. Iterates through different sets of symptoms templates and prompts. Loads the conversation as a
    document. Defines an LLMChain and a StuffDocumentsChain to process the conversation and extract information. Runs the
    extraction chain and converts the extracted string to a dictionary. Updates the extraction dictionary with the partial
    extraction results from each set of symptoms. Returns the final extraction dictionary.

    :param:
        conversation (str): The conversation to extract information from.

    :return:
        dict: The extracted information dictionary.
    """

    # Load LLM
    llm = llm_loader()
    extraction = {}  # The extraction dictionary
    # Loop through the symptoms
    for symptom in [SYMPTOMS1, SYMPTOMS2, SYMPTOMS3, SYMPTOMS4]:
        # Prompt template for messages
        prompt = PromptTemplate.from_template(template=symptom)
        # Conversation loader
        conversation_load = Document(page_content=conversation)
        # Define LLM chain
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        # Define StuffDocumentsChain
        stuff_chain = StuffDocumentsChain(
            llm_chain=llm_chain, document_variable_name="conversation", verbose=True
        )
        # Run the chain and replace the underscore with a space
        partial_extracted = stuff_chain.run([conversation_load]).replace("\_", "_").replace("```", "")
        partial_extracted = partial_extracted.replace("json", "").replace("\n", "")
        # Check if partial_extracted starts with { and ends with }
        if not partial_extracted.startswith("{"):
            partial_extracted = "{" + partial_extracted
        if not partial_extracted.endswith("}"):
            partial_extracted = partial_extracted + "}"

        # Convert the string to a dictionary
        partial_extracted = ast.literal_eval(partial_extracted)
        # Update the extraction dictionary
        extraction |= partial_extracted
    # Return the extraction dictionary
    return extraction

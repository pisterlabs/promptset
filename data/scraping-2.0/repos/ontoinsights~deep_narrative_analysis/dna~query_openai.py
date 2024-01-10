# Query for details using OpenAI

import json
import logging
import openai
import os
# from tenacity import *

from dna.utilities_and_language_specific import add_to_dictionary_values

openai.api_key = os.environ.get('OPENAI_API_KEY')
model_engine = "gpt-4-1106-preview"   # gpt-4 or gpt-4-1106-preview

any_boolean = 'true/false'
interpretation_views = 'conservative, liberal or neutral'
modal_text = '"can", "could", "have to", "may", "might", "must", "ought to", "shall", "should", "will" and "would"'
person = 'using the numbers 1, 2, 3'
semantic_labels = '"agent", "dative", "experiencer", "force", "instrument", "location", ' \
                  '"measure", "patient" or "theme"'
sentiment = 'positive, negative or neutral'
tense = 'past, present, future'
voice = 'active or passive'

possible_content1 = '\n\n{'
possible_content2 = 'json\n{'

# JSON formats
consistency_format = '{"consistent": "bool"}'

coref_format = '{"references_resolved": "bool", ' \
               '"updated_text": "string"}'

name_check_format = '{"same": "boolean",' \
                    '"simplified_name: "string"}'

narr_format = '{"goal_numbers": ["int"], ' \
               '"summary": "string", ' \
               '"rhetorical_devices": [{"device_number": "int", "evidence": "string"}], ' \
               '"interpreted_text": [{"perspective": "string", "interpretation": "string"}], ' \
               '"ranking_by_perspective": [{"perspective": "string", "ranking": "int"}]}'

quote_format1 = '{"sentiment": "string", ' \
                '"grade_level": "int", ' \
                '"summary": "string"}'

sent_format1 = '{"person": "int", ' \
                '"sentiment": "string", ' \
                '"voice": "string", ' \
                '"tense": "string", ' \
                '"modal": {"text": "string", "negated": "boolean"}, ' \
                '"errors": "boolean", ' \
                '"grade_level": "int", ' \
                '"summary": "string"}'

sent_format2 = '{"rhetorical_devices": [{"device_number": "int", "evidence": "string"}]}'

quote_format3 = '{"semantics": [{' \
                  '"trigger_text": "string", ' \
                  '"category_number": "int", ' \
                  '"negated": "boolean"}'

sent_format3 = '{"semantics": [{' \
                '"trigger_text": "string", ' \
                '"category_number": "int", ' \
                '"negated": "boolean", ' \
                '"nouns": [{"text": "string", "singular": "boolean", "semantic_role": "string", ' \
                '"noun_type": "int", "semantic_category": "int"}] }]}'

speaker_format = '{"speaker": "string"}'

# Ontology details
categories = [':Acquisition', ':AggressiveCriminalOrHostileAct', ':CaptureAndSeizure', ':Agreement',
              ':DisagreementAndDispute', ':ViolationOfAgreement', ':AgricultureApicultureAndAquacultureEvent',
              ':ArtAndEntertainmentEvent', ':Attempt', ':Avoidance', ':BodilyAct', ':Change', ':Cognition',
              ':EmotionalResponse', ':SensoryPerception', ':CommunicationAndSpeechAct', ':EnvironmentAndCondition',
              ':Continuation', ':DelayAndWait', ':DistributionAndSupply', ':StartAndBeginning', ':End',
              ':WinAndLoss', ':EnvironmentalOrEcologicalEvent', ':EconomicEnvironment', ':FinancialEvent',
              ':HealthAndDiseaseRelated', ':ImpactAndContact', ':InclusionAttachmentAndUnification',
              ':Separation', ':InformationHandling', ':IssuingAndPublishing', ':LegalEvent', ':LifeEvent',
              ':Measurement', ':MeetingAndEncounter', ':MovementTravelAndTransportation', ':PoliticalEvent',
              ':ProductionManufactureAndCreation', ':Punishment', ':RewardAndCompensation',
              ':RemovalAndRestriction', ':ReturnRecoveryAndRelease', ':RiskTaking', ':Searching', ':SpaceEvent',
              ':Storage', ':Substitution', ':UtilizationAndConsumption', ':Possession', ':Process', ':Affiliation',
              ':EventAndState']

categories_text = 'The semantic categories are: ' \
    '1. acquisition such as by purchase or sale, gifting, donation, finding something, seizure and theft ' \
    '2. aggressive, hostile or criminal act such as attack, assault, coercion, intimidation, bribery, threatening, ' \
    'dishonesty, betrayal, resistance, homicide and invasion' \
    '3. taking captive or capturing a person or animal ' \
    '4. agreement such as consensus and signing a contract ' \
    '5. disagreement or dispute ' \
    '6. violation of agreement ' \
    '7. any type of agricultural, apiculture, viniculture and aquacultural act or event (do NOT use for agribusiness ' \
    'occupations)' \
    '8. any type of art and entertainment act or event including attending a movie or sporting event, ' \
    'visiting a museum and playing a game (do NOT use for art, entertainment, sporting occupations)' \
    '9. any attempt to do or achieve something ' \
    '10. avoidance such as bans, boycotts, escape, evasion, ignoring, overlooking, prevention and concealment ' \
    '11. bodily act such as movement, eating, drinking, grooming and sleeping ' \
    '12. change such as increase, decrease or physical change such as melting, bending and vaporization ' \
    '13. any type of cognition such as thinking, focusing, reading, assessing, characterizing, comparing, ' \
    'making a decision, remembering, forgetting, learning and having knowledge or skills, with the exception of ' \
    'emotions, feelings and sensory perception ' \
    '14. any type of emotion or feeling ' \
    '15. any type of sensory perception including pain, hunger and exhaustion ' \
    '16. any type of communication and speech act such as recommending, acknowledging, denying, ' \
    'promising, requesting, boasting, asking a question, granting permission, refusing, deriding and surrendering ' \
    '17. description of a characteristic or an attribute of a person, place, event, or thing such as its physical ' \
    'appearance, weight, population, job, etc. ' \
    '18. continuation ' \
    '19. delay or wait' \
    '20. distribution or supply of something ' \
    '21. start of something ' \
    '22. end of something ' \
    '23. win or loss (when doing further analysis, use the perspective of the winner) ' \
    '24. any type of environmental or ecological event such as a disaster, weather event, or natural phenomenon ' \
    '25. any type of economic condition such as a recession, inflation, and increased/decreased taxes, income or ' \
    'debt ' \
    '26. any type of financial event such as depositing or withdrawing money, releasing an annual report or ' \
    'paying taxes (do NOT use for finance-related occupations)' \
    '27. health-related event or act such as contracting a disease, addiction, a physical injury, frailty, ' \
    'having an allergic reaction, getting a vaccine, malnutrition, pandemic and sterility (do NOT use for health-' \
    'related occupations)' \
    '28. impact and collision ' \
    '29. inclusion, unification and attachment such as adding to a list and assembling something ' \
    '30. separation ' \
    '31. information and data handling including IT operations (do NOT use for IT-related occupations) ' \
    '32. issuing and publishing such as a newspaper, magazine or press release ' \
    '33. any legal or judicial event such as a trial, verdict, jury selection and judicial ruling (do NOT use for ' \
    'legal or judicial occupations)' \
    '34. life event such as birth, death, marriage and divorce ' \
    '35. measurement, measuring and reported assessment, count, percentages, etc. ' \
    '36. meeting and encounter such as party, chance encounter or ceremony ' \
    '37. any type of movement, travel or transportation such as entering/leaving a port or train station, ' \
    'loading a truck or rail car, or making incremental changes such as pouring liquid into a container (do NOT ' \
    'use for transportation-related occupations)' \
    '38. any political occurrence such as an election, coup or transfer of power, and political campaign (do NOT ' \
    'use for political occupations)' \
    '39. any type of production, manufacture and creation event such as designing, building or producing ' \
    'products in a factory ' \
    '40. punishment ' \
    '41. reward and compensation ' \
    '42. removal or restriction of something, including blockage of movement, access, flow or freedom ' \
    '43. return, recovery and release ' \
    '44. risk taking including gambling ' \
    '45. search ' \
    '46. any type of space event such as a meteor shower, sun spots, eclipse, or rocket or satellite launch (do NOT ' \
    'use for space-related occupations)' \
    '47. any type of storage event such as moving cargo into or from storage, stockpiling or hoarding ' \
    '48. substitution or imitation of something or someone ' \
    '49. utilization and consumption ' \
    '50. possession ' \
    '51. natural or goal-directed process which is a set of interrelated/interdependent events or conditions ' \
    'that progress along a timeline such as a plan, script, the aging process or climate change' \
    '52. affiliation of a person to another person, group, location, etc. or membership of a person in a group ' \
    'or organization ' \
    '53. other'

narrative_goals = ['advocate', 'analyze', 'describe-set', 'describe-current', 'entertain',
                   'establish-authority', 'inspire', 'life-story']

narrative_goals_text = 'The narrative goal categories are: ' \
    '1. Promote or advocate for a cause, person or thing ' \
    '2. Analyze an issue, event or trend in-depth, by breaking it down ' \
    '3. Describe/chronicle how a set of events, trends and conditions evolve over time ' \
    '4. Describe/report a current event, trend or condition ' \
    '5. Entertain using humor and engaging content ' \
    '6. Establish authority by citing facts, statistics and quotes from experts ' \
    '7. Inspire and motivate through uplifting stories, news, etc. ' \
    '8. Relate a personal narrative or life story ' \

noun_categories = [':Person', ':Person, :Collection', ':GovernmentalEntity', ':OrganizationalEntity', ':EthnicGroup',
                   ':PoliticalGroup', ':ReligiousGroup', ':Animal', ':Plant', ':Location', ':LineOfBusiness',
                   ':EventAndState', ':SubstanceAndRawMaterial', ':EnvironmentAndCondition', ':LawAndPolicy',
                   ':MusicalInstrument', ':PharmaceuticalAndMedicinal', ':HealthAndDiseaseRelated',
                   ':ElectricityAndPower', ':MachineAndTool', ':WeaponAndAmmunition', ':WasteAndResidue',
                   ':Measurement', 'owl:Thing']

noun_categories_text = 'The noun types are: ' \
    '1. Person ' \
    '2. Group of people such as a family or ' \
    'people at a party (NOT governmental, business or organizational entities)  ' \
    '3. Government or government-funded entity ' \
    '4. Organization, sub-organization, club or society ' \
    '5. Ethnic group ' \
    '6. Political group ' \
    '7. Religious group ' \
    '8. Animal ' \
    '9. Plant ' \
    '10. Place or location ' \
    '11. Occupation ' \
    '12. Event that can be further classified using the semantic categories defined for the sentence ' \
    '13. Chemical substance or raw material ' \
    '14. Environmental, economic or ecological condition ' \
    '15. Law, policy or legal decision ' \
    '16. Musical instrument ' \
    '17. Pharmaceutical or medicinal entity ' \
    '18. Disease, illness or symptom ' \
    '19. Electricity and power-related ' \
    '20. Machine or tool ' \
    '21. Weapon or ammunition ' \
    '22. Waste or residue ' \
    '23. Quantity or measurement ' \
    '24. Other'

rhetorical_devices = ['ad hominem', 'allusion', 'antanagoge', 'aphorism', 'ethos', 'expletive',
                      'hyperbole', 'imagery', 'invective', 'irony', 'kairos', 'litote', 'logos',
                      'metaphor', 'nostalgia', 'pathos', 'pleonasm', 'repetition', 'loaded language',
                      'repeated statements', 'rhetorical question', 'juxtaposition']

rhetorical_devices_text = 'The rhetorical device categories are: ' \
    '1. Use of wording that verbally demeans or attacks a person (ad hominem) ' \
    '2. Reference to an historical/literary person, place or thing that has symbolic meaning such as ' \
    'saying that "sleeping late is my Achilles heel" where "Achilles heel" is the reference (allusion) ' \
    '3. Balancing negative wording with positive (antanagoge) ' \
    '4. Expressing a truth or moral principle such as "a stitch in time saves nine" (aphorism) ' \
    '5. Reference to authority figures, things that are popular, and/or to people in occupations that should ' \
    'have knowledge (such as doctors or professors) in order to justify a statement (ethos) ' \
    '6. Use of emphasis words, such as "in fact", "of course", "clearly" or "certainly" (expletive) ' \
    '7. Use of exaggerated wording (hyperbole) ' \
    '8. Use of imagery and descriptive phrases that paint a vivid picture that emotionally engages a reader ' \
    '9. Use of ridicule, or angry or insulting language (invective) ' \
    '10. Use of irony or satire ' \
    '11. Reference invoking feelings/remembrances of specific day, time, event or season such as ' \
    'discussing the Civil War in order to engage a reader (kairos) ' \
    '12. Use of double negative (litote) ' \
    '13. Use of logical reasoning terms, statistics and numbers (logos) ' \
    '14. Use of analogy, metaphor and simile to compare one thing with another of a different kind ' \
    '15. Use of wording that invokes nostalgia ' \
    '16. Wording that appeals to emotion such as fear or empathy (pathos) ' \
    '17. Use of superfluous or redundant language such as referring to a "burning fire" (pleonasm) ' \
    '18. Repeating words for emphasis ' \
    '19. Use of "loaded language" such as words like "double-dealing", with strong connotations which invoke ' \
    'emotions and judgments'

# Additional rhetorical devices for narratives
additional_devices_text = \
    '20. Repeating statements such that they are remembered ' \
    '21. Asking rhetorical questions ' \
    '22. Placing contrasting ideas or situations side by side (juxtaposition)'

# Co-reference related prompting - Future
coref_prompt = 'You are a linguist and NLP expert, analyzing text. Here are a series of sentences (ending with the ' \
    'string "**" which should be ignored): {sentences} ** Resolve co-references in the following sentences, ' \
    'updating the co-references with their more specific details. Return the sentences using the JSON format, ' + \
    f'{coref_format}. If there were no references to resolve, indicate this using the "references_resolved" boolean ' \
    f'- setting it to false. In this case, the "updated_text" element would be set to the original inputted text.'

# Validation prompting
name_check_prompt = 'You are a student researching whether two individuals could be the same, given only their ' \
    'names. The name that you are researching is {noun_text}. You think that the person with this name might be ' \
    'the same as someone who has the following names and nicknames: {labels}. Evaluate if the two people could ' \
    'be the same. If they are not, evaluate if the name can be simplified by removing adjectives, formal ' \
    'titles, etc. When simplifying, do not add or change the remaining text. Return the response as a JSON ' + \
    f'object with keys and values as defined by {name_check_format}.'

# Narrative-level prompting
narr_prompt = f'You are a political observer analyzing news articles. ' \
    'Here is the text of a narrative or article (ending with the string "**" which should be ignored): ' \
    '{narr_text} **' + \
    f'Here is a numbered list of the possible goals of the narrative. {narrative_goals_text} ' \
    f'Here is a numbered list of the possible rhetorical devices that may be used in the narrative. ' \
    f'{rhetorical_devices_text} {additional_devices_text} ' + \
    'Indicate the numbers of the 2 most likely narrative goals. Indicate the numbers of the rhetorical' \
    'devices used, and explain why those devices were returned. Also, summarize the narrative, and explain ' + \
    f'how it would be interpreted from each of the following perspectives: {interpretation_views}. Rank the ' \
    f'text from 1-5 for each perspective. Return the response as a JSON object with keys and values as ' \
    f'defined by {narr_format}.'

# Sentence-level prompting
nothing_else = 'Do NOT return any free-form string text in the response.'

quote_prompt1 = \
    'You are a linguist and NLP expert, analyzing quotations from news articles. ' \
    'Here is the text of a quotation from an article (ending with the string "**" which should be ignored): ' \
    '{quote_text} ** ' + \
    f'For the text, indicate its sentiment ({sentiment}) and create a summary in 8 words or less. ' \
    f'Indicate the grade level that is expected of a reader to understand its semantics. ' + \
    f'Return the response as a JSON object with keys and values as defined by {quote_format1}. {nothing_else}'

sent_prompt1 = \
    'You are a linguist and NLP expert, analyzing the text from narratives and news articles. ' \
    'Here is the text of a sentence from an article that should be analyzed. The text ends with the string ' \
    '"**" which should be ignored. {sent_text} ** ' + \
    f'For the sentence, indicate if it is in the first, second or third person ({person}), its sentiment ' \
    f'({sentiment}), whether it is in the {voice} voice, and its tense ({tense}). Create a summary of the ' \
    f'sentence in 8 words or less. Indicate the grade level that is expected of a reader to understand the ' \
    f'semantics of the sentence, and whether any grammatical or spelling errors are present ({any_boolean}. ' \
    f'Lastly, ONLY if there is a "modal" auxiliary verb in the sentence that is one of: {modal_text}, return its ' \
    f'text and indicate whether it is negated ({any_boolean}). If one of the listed modal auxiliary verbs is ' \
    f'not found, indicate the modal text as "none" and its negation flag as false. Return the response as a JSON ' \
    f'object with keys and values as defined by {sent_format1}. {nothing_else}'

common_prompt2_text = \
    f'ONLY return a rhetorical device result if a justification/evidence can be provided for it. Explain the ' \
    f'evidence in the results. Note that there may be no devices used in the text. If so, do NOT report this in ' \
    f'the results. Return the results ONLY as a JSON object with keys and values defined by ' \
    f'{sent_format2} {nothing_else} IF there are no rhetorical devices used in the text, return an empty ' \
    f'array for the value of the "rhetorical_devices" JSON key.'

quote_prompt2 = \
    'You are a linguist and NLP expert, analyzing quotations from news articles. ' \
    'Here is a numbered list of the possible types of rhetorical devices that may be used in a quotation. ' + \
    f'{rhetorical_devices_text} ' + \
    'Here is the text of a quotation from an article (ending with the string "**" which should be ignored): ' \
    '{sent_text} ** ' + \
    f'Provide the numbers associated with any rhetorical devices found in the quotation. ' \
    f'{common_prompt2_text} {nothing_else}'

sent_prompt2 = 'You are a linguist and NLP expert, analyzing the text from narratives and news articles. ' \
    'Here is a numbered list of the possible types of rhetorical devices that may be used in the text of ' + \
    f'a sentence from the narrative or article. {rhetorical_devices_text} ' + \
    'Here is the text of a specific sentence to be analyzed. The text ends with the string "**" which should be ' \
    'ignored. {sent_text} ** ' + \
    f'Provide the numbers associated with any rhetorical devices found in the sentence. ' \
    f'{common_prompt2_text} {nothing_else}'

short_or_vague = 'For very short or vague texts (perhaps with a single noun/verb), infer the semantics, ' \
                 'and do NOT mention vagueness or inference in the response.'

statement_of_the_form = 'If the analyzed text is a statement of the form, "x was/is/will be y", then the ' \
    'verb should be reported using the category number 17 (a description). The text should be processed assuming ' \
    'that "is"/"was"/"will be" is the verb and nothing should be mentioned in the response.'

# TODO: Are 3 categories the correct #, or more? (Also applies to sent_prompt3)
quote_prompt3 = 'You are a linguist and NLP expert, analyzing quotations from news articles. ' \
    'Here is a numbered list of categories of the possible semantics that may be discussed in a quotation. ' + \
    f'{categories_text} Here is the text of a specific quotation from an article (ending with the string "**" ' + \
    'which should be ignored): {sent_text} ** For the quotation, indicate the numbers of the 1-3 most appropriate ' + \
    f'semantic categories. {statement_of_the_form} ONLY return the maximum of 3 semantics IF there is ' + \
    'justification/evidence for each one. If there is insufficient evidence to justify assigning any semantic ' \
    'category, return the number 53 ("other"). In the response, provide the text that triggered the semantic ' + \
    f'categorization. {short_or_vague} Also indicate whether each (discovered or inferred) semantic is ' \
    f'negated ({any_boolean}). Return the response as a JSON object with keys and values as defined ' \
    f'by {quote_format3}. {nothing_else}'

sent_prompt3 = 'You are a linguist and NLP expert, analyzing the text of narratives and news articles. ' \
    'Here is a numbered list of categories of the possible semantics that may be discussed in the text. ' + \
    f'{categories_text} ' + \
    'Here is the text of a sentence from a narrative or article to be analyzed. The text ends with ' \
    'the string "**" which should be ignored. {sent_text} ** For the sentence, indicate the numbers of the' \
    '2 most appropriate semantic categories. Provide the text that triggered the semantic categorization. ' + \
    f'{short_or_vague} For each categorization, indicate if it is negated ({any_boolean}), and determine the ' \
    f'nouns that are associated with it via one of these semantic role labels: {semantic_labels}. For an associated ' \
    f'noun, return its role label and text string, whether it is singular ({any_boolean}), and its noun type. ' \
    f'{noun_categories_text} Return the number of the noun type. If the type is either number 12 or 24, also ' \
    f'assign a semantic category to it. If the number is not 12 or 24, return 0 for the semantic category. ' \
    f'Return the response as a JSON object with keys and values as defined by {sent_format3}. {nothing_else}'

speaker_prompt = 'You are a linguist and NLP expert, analyzing quotations from news articles. ' \
    'Here is the text of a news article (ending with the string "**" which should be ignored): {narr_text} ** ' \
    'Here is a quotation contained in the article (ending with the string "**" which should be ignored): ' \
    '{quote_text} ** ' \
    'For the quotation, find it in the news article, and indicate the name of the person who spoke/' \
    'communicated it. If the speaker is identified using a pronoun, analyze the text to dereference it. ' + \
    f'Return the response as a JSON object with keys and values as defined by {speaker_format}.'

wikipedia_prompt = 'You are a student researching a proper noun and are given its text and information ' \
    'that the noun is a type of "{text_type}". You retrieved a possible definition for the ' \
    'noun from Wikipedia. That definition is given by the following text, ending with the string "**" ' \
    'which should be ignored. {wiki_def} **. Evaluate if the definition is ' \
    'consistent with the noun type, and return the response as a JSON object with keys and ' + \
    f'values as defined by {consistency_format}.'


# @retry(stop=stop_after_attempt(5), wait=(wait_fixed(3) + wait_random(0, 2)))
def access_api(content: str) -> dict:
    """
    Surrounding the calls to the OpenAI API with retry logic.

    :param content: String holding the content of the completion request
    :param attempts: Number of current retry attempt
    :return: The 'content' response from the API as a Python dictionary
    """
    try:
        response = openai.ChatCompletion.create(
            model=model_engine,
            messages=[
                {"role": "user", "content": content}
            ],
            temperature=0
        )
        if '"finish_reason": "stop"' not in str(response):
            logging.error(f'Non-stop finish reason for content, {content}')
            return dict()
    except Exception:
        logging.error(f'OpenAI exception for content, {content}: {str(e)}')
        return dict()
    try:
        resp_dict = json.loads(response['choices'][0]['message']['content'].replace('\n', ' '))
    except Exception:
        revised_content = ''
        if possible_content1 in response['choices'][0]['message']['content']:
            revised_content = ' { ' + response['choices'][0]['message']['content'].split(possible_content1)[1]
        elif possible_content2 in response['choices'][0]['message']['content']:
            revised_content = ' { ' + response['choices'][0]['message']['content'].split(possible_content2)[1]
            revised_content = revised_content.split('```')[0]
        if revised_content:
            try:
                resp_dict = json.loads(revised_content.replace('\n', ' '))
            except:
                logging.error(f'Tried revised JSON content: {revised_content}')
                return dict()
        else:
            logging.error(f'Response is not JSON formatted: {str(response)}')
            return dict()
    return resp_dict

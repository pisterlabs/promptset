import json
import logging
import traceback
from datetime import datetime
import requests
import openai
from utils import heconstants
from utils.s3_operation import S3SERVICE
from services.kafka.kafka_service import KafkaService
from config.logconfig import get_logger

s3 = S3SERVICE()
producer = KafkaService(group_id="aipreds")
openai.api_key = heconstants.OPENAI_APIKEY
logger = get_logger()
logger.setLevel(logging.INFO)


class aiPreds:
    def clean_pred(self,
                   clinical_information,
                   remove_strings=[
                       "not found",
                       "no information found"
                       "information not provided",
                       "unknown",
                       "n/a",
                       "null",
                       "undefined",
                       "no data",
                       "none",
                       "missing",
                       "not available",
                       "not applicable",
                       "unidentified",
                       "not specified",
                       "not mentioned",
                       "none mentioned",
                       "not detected",
                       "insufficient data",
                       "no mention of",
                       "absence of",
                       "not indicated",
                       "no information on",
                       "unable to determine",
                   ],
                   ):
        try:
            for key in list(clinical_information.keys()):
                if isinstance(clinical_information[key], str) and any(
                        rs.lower() in clinical_information[key].lower() for rs in remove_strings
                ):
                    del clinical_information[key]

            details = clinical_information.get("details", {})
            if details:
                keys_to_remove = [k for k, v in details.items() if
                                  isinstance(v, str) and any(rs.lower() in v.lower() for rs in remove_strings)]

                for k in keys_to_remove:
                    del details[k]

                clinical_information["details"] = details

            return clinical_information
        except Exception as exc:
            msg = "Failed to get clean_pred :: {}".format(exc)
            trace = traceback.format_exc()
            logger.error(msg, trace)

    def clean_null_entries(self, entities):
        # List of keys to delete from the main dictionary
        keys_to_delete = []

        for key, value in entities.items():
            # Check if the value is a dictionary and has a 'text' key
            if isinstance(value, dict) and "text" in value:
                if value["text"] is None:
                    keys_to_delete.append(key)
            # Check if the value is an empty list in the 'entities' sub-dictionary
            elif key == "entities":
                empty_keys = [k for k, v in value.items() if v == []]
                for empty_key in empty_keys:
                    del value[empty_key]

        # Delete the keys from the main dictionary
        for key in keys_to_delete:
            del entities[key]

        return entities

    def execute_function(self, message, start_time):
        try:
            conversation_id = message.get("care_req_id")
            file_path = message.get("file_path")
            chunk_no = message.get("chunk_no")
            retry_count = message.get("retry_count")
            merged_segments = []

            conversation_datas = s3.get_files_matching_pattern(
                pattern=f"{conversation_id}/{conversation_id}_*json")
            if conversation_datas:
                for conversation_data in conversation_datas:
                    merged_segments += conversation_data["segments"]

            entities = {
                "age": {"text": None, "value": None, "unit": None},
                "gender": {"text": None, "value": None, "unit": None},
                "height": {"text": None, "value": None, "unit": None},
                "weight": {"text": None, "value": None, "unit": None},
                "bmi": {"text": None, "value": None, "unit": None},
                "ethnicity": {"text": None, "value": None, "unit": None},
                "insurance": {"text": None, "value": None, "unit": None},
                "physicalActivityExercise": {"text": None, "value": None, "unit": None},
                "bloodPressure": {"text": None, "value": None, "unit": None},
                "pulse": {"text": None, "value": None, "unit": None},
                "respiratoryRate": {"text": None, "value": None, "unit": None},
                "bodyTemperature": {"text": None, "value": None, "unit": None},
                "substanceAbuse": {"text": None, "value": None, "unit": None},
                "entities": {
                    "medications": [],
                    "symptoms": [],
                    "diseases": [],
                    "diagnoses": [],
                    "surgeries": [],
                    "tests": [],
                },
                "summaries": {
                    "subjectiveClinicalSummary": [],
                    "objectiveClinicalSummary": [],
                    "clinicalAssessment": [],
                    "carePlanSuggested": [],
                },
            }

            ai_preds_file_path = f"{conversation_id}/ai_preds.json"
            if s3.check_file_exists(ai_preds_file_path):
                entities = s3.get_json_file(ai_preds_file_path)

            if merged_segments:
                text = " ".join([_["text"] for _ in merged_segments])
                extracted_info = self.get_preds_from_open_ai(text)
                extracted_info = self.clean_pred(extracted_info)

                details = extracted_info.get("details", {})
                if details:
                    for k, v in details.items():
                        if k in entities:
                            if entities[k]["text"] is None:
                                entities[k]["text"] = v
                            # else:
                            #     entities[k]["text"] += ", " + v

                            entities[k]["value"] = entities[k]["text"]

                all_texts_and_types = []
                for k in ["medications", "symptoms", "diseases", "diagnoses", "surgeries", "tests"]:
                    if isinstance(extracted_info.get(k), str):
                        v = extracted_info.get(k, "").replace(" and ", " , ").split(",")
                        for _ in v:
                            if _.strip():
                                all_texts_and_types.append((_.strip(), k))
                    elif isinstance(extracted_info.get(k), list):
                        v = extracted_info.get(k, "")
                        for _ in v:
                            all_texts_and_types.append((_.strip(), k))

                text_to_codes = {}

                if all_texts_and_types:
                    try:
                        codes = \
                            requests.post(heconstants.AI_SERVER + "/code_search/infer",
                                          json=all_texts_and_types).json()[
                                'prediction']
                    except:
                        codes = [{"name": _, "code": None} for _ in all_texts_and_types]

                    for (text, _type), code in zip(all_texts_and_types, codes):
                        text_to_codes[text] = {"name": code["name"], "code": code["code"], "score": code.get("score")}

                for k in ["medications", "symptoms", "diseases", "diagnoses", "surgeries", "tests"]:
                    if isinstance(extracted_info.get(k), str):
                        v = extracted_info.get(k, "").replace(" and ", " , ").split(",")
                        v = [
                            {
                                "text": _.strip(),
                                "code": text_to_codes.get(_.strip(), {}).get("code", None),
                                "code_value": text_to_codes.get(_.strip(), {}).get("name", None),
                                "code_type": "",
                                "confidence": text_to_codes.get(_.strip(), {}).get("score", None),
                            }
                            for _ in v
                            if _.strip()
                        ]
                        entities["entities"][k] = v
                    elif isinstance(extracted_info.get(k), list):
                        v = extracted_info.get(k, "")
                        val = [
                            {
                                "text": _.strip(),
                                "code": text_to_codes.get(_.strip(), {}).get("code", None),
                                "code_value": text_to_codes.get(_.strip(), {}).get("name", None),
                                "code_type": "",
                                "confidence": text_to_codes.get(_.strip(), {}).get("score", None),
                            }
                            for _ in v
                            if _.strip()
                        ]
                        entities["entities"][k] = val

                # if entities:
                #     current_segment["ai_preds"] = entities
                # else:
                #     current_segment["ai_preds"] = None

                entities = self.clean_null_entries(entities)
                print("entities ::", entities)
                s3.upload_to_s3(f"{conversation_id}/ai_preds.json", entities, is_json=True)
                data = {
                    "es_id": f"{conversation_id}_SOAP",
                    "chunk_no": chunk_no,
                    "file_path": file_path,
                    "api_path": "asr",
                    "api_type": "asr",
                    "req_type": "encounter",
                    "executor_name": "SOAP_EXECUTOR",
                    "state": "Analytics",
                    "retry_count": retry_count,
                    "uid": None,
                    "request_id": conversation_id,
                    "care_req_id": conversation_id,
                    "encounter_id": None,
                    "provider_id": None,
                    "review_provider_id": None,
                    "completed": False,
                    "exec_duration": 0.0,
                    "start_time": str(start_time),
                    "end_time": str(datetime.utcnow()),
                }
                producer.publish_executor_message(data)

        except Exception as exc:
            msg = "Failed to get AI PREDICTION :: {}".format(exc)
            trace = traceback.format_exc()
            logger.error(msg, trace)
            if retry_count <= 2:
                data = {
                    "es_id": f"{conversation_id}_SOAP",
                    "chunk_no": chunk_no,
                    "file_path": file_path,
                    "api_path": "asr",
                    "api_type": "asr",
                    "req_type": "encounter",
                    "executor_name": "SOAP_EXECUTOR",
                    "state": "Analytics",
                    "retry_count": retry_count,
                    "uid": None,
                    "request_id": conversation_id,
                    "care_req_id": conversation_id,
                    "encounter_id": None,
                    "provider_id": None,
                    "review_provider_id": None,
                    "completed": False,
                    "exec_duration": 0.0,
                    "start_time": str(start_time),
                    "end_time": str(datetime.utcnow()),
                }
                producer.publish_executor_message(data)

    def string_to_dict(self, input_string):
        # Initialize an empty dictionary
        result = {}
        details = {}

        # Define the keys to be nested inside 'details'
        detail_keys = ["age_years", "gender", "height_cm", "weight_kg", "ethnicity",
                       "substanceAbuse", "bloodPressure", "pulseRate", "respiratoryRate",
                       "bodyTemperature_fahrenheit"]

        # Split the input string into lines
        lines = input_string.split(',\n')

        # Process each line
        for line in lines:
            # Split the line into key and value
            key, value = line.split(': ')

            # Clean up the key and value strings
            key = key.strip().replace('"', '')

            # Handle multiple values and 'not found' cases
            if ', ' in value:
                value = [item.strip().replace('"', '') for item in value.replace(" and ", " , ").split(', ')]
            else:
                value = value.strip().replace('"', '')

            # Check if the key should be nested inside 'details'
            if key in detail_keys:
                if key == "height_cm":
                    details["height"] = value
                elif key == "age_years":
                    details["age"] = value
                elif key == "weight_kg":
                    details["weight"] = value
                elif key == "bodyTemperature_fahrenheit":
                    details["bodyTemperature"] = value
                elif key == "pulseRate":
                    details["pulse"] = value
                else:
                    details[key] = value
            else:
                if key == "orders":
                    result["tests"] = value
                else:
                    result[key] = value

        # Add the 'details' dictionary to the result
        result['details'] = details

        return result

    def get_preds_from_open_ai(self,
                               transcript_text,
                               function_list=heconstants.faster_clinical_info_extraction_functions,
                               min_length=30,
                               ):
        try:
            transcript_text = transcript_text.strip()
            if not transcript_text or len(transcript_text) <= min_length:
                raise Exception("Transcript text is too short")

            template = """
            "medications": <text>,
            "symptoms": <text>,
            "diseases": <text>,
            "diagnoses": <text>,
            "surgeries": <text>,
            "orders": <text>,
            "age_years": <text>,
            "gender": <text>,
            "height_cm": <text>,
            "weight_kg": <text>,
            "ethnicity": <text>,
            "substanceAbuse": <text>,
            "bloodPressure": <text>,
            "pulseRate": <text>,
            "respiratoryRate": <text>,
            "bodyTemperature_fahrenheit": <text>
            """

            messages = [
                {
                    "role": "system",
                    "content": """ Don't make assumptions about what values to plug into functions. return not found if you can't find the information.
                    You are acting as an expert clinical entity extractor.
                    Extract the described information from given clinical notes or consultation transcript.
                    No extra information or hypothesis not present in the given text should be added. Separate items with , wherever needed.
                    All the text returned should be present in the given TEXT. no new text should be returned.""",
                },
                {
                    "role": "system",
                    "content": f"Use given template to return the response : {template}",
                },
                {"role": "user", "content": f"TEXT: {transcript_text}"},
            ]

            for model_name in ["gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k-0613", "gpt-4-0613"]:
                try:
                    response = openai.ChatCompletion.create(
                        model=model_name,
                        messages=messages,
                        # functions=function_list,
                        # function_call={"name": "ClinicalInformation"},
                        temperature=0.6,
                    )

                    extracted_info = response.choices[0]["message"]["content"]
                    converted_info = self.string_to_dict(extracted_info)
                    logger.info(f"extracted_info :: {converted_info}")
                    # extracted_info = json.loads(
                    #     response.choices[0]["message"]["function_call"]["arguments"]
                    # )
                    return converted_info

                except Exception as ex:
                    logger.error(ex)
                    pass
        except Exception as exc:
            msg = "Failed to get OPEN AI PREDICTION :: {}".format(exc)
            logger.error(msg)


# if __name__ == "__main__":
    # ai_pred = aiPreds()
    # stream_key = "test_new39"
    # start_time = datetime.utcnow()
    # message = {
    #     "es_id": f"{stream_key}_AI_PRED",
    #     "chunk_no": 2,
    #     "file_path": f"{stream_key}/{stream_key}_chunk2.wav",
    #     "api_path": "asr",
    #     "api_type": "asr",
    #     "req_type": "encounter",
    #     "executor_name": "AI_PRED",
    #     "state": "AiPred",
    #     "retry_count": None,
    #     "uid": None,
    #     "request_id": stream_key,
    #     "care_req_id": stream_key,
    #     "encounter_id": None,
    #     "provider_id": None,
    #     "review_provider_id": None,
    #     "completed": False,
    #     "exec_duration": 0.0,
    #     "start_time": str(start_time),
    #     "end_time": str(datetime.utcnow()),
    # }
    # ai_pred.execute_function(message=message, start_time=datetime.utcnow())
    # ai_pred.get_preds_from_open_ai(transcript_text)

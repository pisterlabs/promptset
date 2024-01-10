import tiktoken
# from openai import OpenAI
# client = OpenAI()

def get_token_length(prompt):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-instruct")
    return len(encoding.encode(prompt))

def get_filtered_poi_list(poi_list, preferences):
    filtered_poi_list = []
    for poi in poi_list:
        if poi['type'] is not None:
            poi_type_list = poi['type']  # Convert to lowercase for case-insensitive comparison
            for poi_type in poi_type_list:
                poi_type_lower = poi_type.lower()
                for pref in preferences:
                    pref_lower = pref.lower()  # Convert to lowercase for case-insensitive comparison
                    if pref_lower in poi_type_lower:
                        filtered_poi_list.append(poi)
                        break
        
    return filtered_poi_list

# def upload_training_file(file_name):
#     file_key = client.files.create(
#         file=open(file_name, "rb"),
#         purpose="fine-tune"
#         )
#     print("file_key : ", file_key)
#     return file_key

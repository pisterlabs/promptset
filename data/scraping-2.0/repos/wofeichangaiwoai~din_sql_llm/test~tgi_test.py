

from langchain.llms import HuggingFaceTextGenInference
llm = HuggingFaceTextGenInference(
    inference_server_url="http://colossal-llm-api.home-dev.ubix.io",
    max_new_tokens=650,
    #top_k=10,
    #top_p=0.95,
    #typical_p=0.95,
    temperature=0.01,
    #repetition_penalty=1.03,
)

print(llm.predict("hello,  I'm felix"))

"""
python test/tgi_test.py
"""

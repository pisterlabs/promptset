from flask import jsonify
import openai



def optimize_code(code):
    openai.api_key="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"

    text="Reduce the cognitive compleity of the following code"
    msg=text+code
    response=openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"user","content":msg}
        ]
    )

    # print(response)

    assistant_response=response['choices'][0]['message']['content']
    print(assistant_response)
    return jsonify({"code": assistant_response})
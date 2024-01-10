from flask import Flask, request, jsonify
from openai import OpenAI 
from flask_cors import CORS


app = Flask(__name__)



# Enable CORS for all routes
CORS(app, resources={r'/generate_response': {'origins': '*'}})



@app.route('/generate_response', methods=['POST'])
def post():
    data = request.get_json()
    user_message = data['text']
    print(user_message)
    response_text = generate_response(user_message)
    print(user_message)

    return jsonify({'response': response_text})

def generate_response(user_message):
    # Add your existing code here
    system_message = """
    انت مساعد شخصي لتلاميذ باكالوريا الجزائر
    عليك ان تجيب عللا اسئلتهم في حدود اليانات التي تم تدريبك عليها
    حاول ان تحعل رسالتك مطولة نوعا ما
    """
    test_messages = [{"role": "system", "content": system_message},
                     {"role": "user", "content": user_message}]
    client = OpenAI(api_key="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")
    
    
    response = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-1106:personal:bac-gpt:8Tcuc0GP",
        messages=test_messages,
        temperature=0.10,
        max_tokens=500
    )

    return response.choices[0].message.content

if __name__ == '__main__':
    app.run(debug=True)



# كيف اختار موضوع البكالوريا"
# كيف أختار التخصص الجامعي
# ما رأيك في الدروس الخصوصية و هل تنصحني أن ألتحق بها 
# كيف أتعامل مع المواد العلمية ؟"
# هل يمكنني أن أتغيب عن الدروس في في الثانوية ؟
#  كيف يمكنني اكتشاف اهتماماتي الأكاديمية واختيار تخصص مناسب؟"
# اذكر أهم الكتب الخارجية لمادة العربية / الادب العربي / اللغة العربية
from openai import OpenAI
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=OPENAI_API_KEY)

@app.route('/diacritize', methods=['POST'])
def diacritize():

    data = request.json
    non_diacritized_text = data.get("text")

    few_shot = '''
Non-Diacritized:

معلوم نہیں یہ بات معقول تھی یا غیر معقول، بہرحال دانش مندوں کے فیصلے کے مطابق ادھر ادھر اونچی سطح کی کانفرنسیں ہوئیں اور بالآخر ایک دن پاگلوں کے تبادلے کے لیے مقرر ہوگیا۔ اچھی طرح چھان بین کی گئی۔

Diacritized:

مَعْلُوم نَہِیں یِہ بات مَعْقُول تھی یا غَیْر مَعْقُول، بَہْرحال دانِش مندوں کے فیصلے کے مُطابِق اِدَھر اُدَھر اونچی سَطْح کی کانفرنسیں ہوئیں اور بِالْآخِر ایک دِن پاگلوں کے تبادلے کے لِیے مُقَرَّر ہوگیا۔ اَچّھی طَرْح چھان بَین کی گئی۔

Non-Diacritized:

اب ملاقات بھی نہیں آتی تھی۔ پہلے تو اسے اپنے آپ پتہ چل جاتا تھا کہ ملنے والے آرہے ہیں، پر اب جیسے اس کے دل کی آواز بھی بند ہوگئی تھی جو اسے ان کی آمد کی خبر دے دیا کرتی تھی۔ اس کی بڑی خواہش تھی کہ وہ لوگ آئیں جو اس سے ہمدردی کا اظہار کرتے تھے اور اس کے لیے پھل، مٹھائیاں اور کپڑے لاتے تھے۔

Diacritized:

اَب مُلاقات بھی نَہِیں آتی تھی۔ پَہْلے تو اُسے اَپْنے آپ پَتَہ چُل جاتا تھا کِہ مِلْنے والے آرہے ہیں، پر اَب جَیسے اُس کے دِل کی آواز بھی بَنْد ہوگئی تھی جو اُسے اُن کی آمَد کی خَبَر دے دِیا کرتی تھی۔ اُس کی بَڑی خواہِش تھی کِہ وہ لُوگ آئِیں جو اُس سے ہَمْدَرْدی کا اِظْہار کرتے تھے اور اُس کے لِیے پَھل، مِٹھَائیاں اور کَپْڑے لاتے تھے۔  

Non-Diacritized:

پنشن کے پچاس روپے جیب میں ڈال کر وہ برآمدہ طے کرتا اور چق لگے کمرے کے پاس جا کر اپنی آمد کی اطلاع کراتا۔ چھوٹے جج صاحب اس کو زیادہ دیر تک باہر کھڑا نہ رکھتے، فوراً اندر بلا لیتے اور سب کام چھوڑ کر اس سے باتیں شروع کردیتے۔ 

’’تشریف رکھیے منشی صاحب۔۔۔ فرمائیے مزاج کیسا ہے؟‘‘ 

’’اللہ کا لاکھ لاکھ شکر ہے۔۔۔آپ کی دعا سے بڑے مزے میں گزررہی ہے، میرے لائق کوئی خدمت؟‘‘ 

’’آپ مجھے کیوں شرمندہ کرتے ہیں۔ میرے لائق کوئی خدمت ہو تو فرمائیے۔ خدمت گزاری تو بندے کا کام ہے۔‘‘ 

’’آپ کی بڑی نوازش ہے۔‘‘
 
Diacritized:

پنشن کے پَچاس رُوپے جیب میں ڈال کر وہ بَرآمدہ طَے کَرْتا اور چَق لَگے کَمْرے کے پاس جا کر اَپْنی آمَد کی اِطَّلاع کراتا۔ چُھوٹے جج صاحب اُس کُو زِیَادہ دَیر تَک باہَر کَھڑا نَہ رکھتے فَوراً اِنْدَر بُلا لیتے اور سَب کام چھوڑ کر اُس سے باتیں شُرُوع کردیتے۔

’’تَشْرِیف رکھیے منشی صاحب۔۔۔ فرمائیے مِزَاج کیسا ہے؟‘‘ 

’’اللہ کا لاکھ لاکھ شُکْر ہے۔۔۔آپ کی دُعا سے بَڑے مَزے میں گزررہی ہے، میرے لائِق کوئی خِدْمَت؟‘‘

’’آپ مُجھے کِیُوں شَرْمِنْدَہ کرتے ہیں۔ میرے لائِق کوئی خِدْمَت ہو تو فَرْمائیے۔ خِدْمَت گُزاری تو بندے کا کام ہے۔‘‘ 

’’آپ کی بَڑی نَوازِش ہے۔‘‘ 
'''

    system_message = "The following are examples of Urdu sentences with and without diacritics. Please add diacritics to the non-diacritized sentence."

    user_message = few_shot + f"\nNon-Diacritized:\n\n{non_diacritized_text}\n\nDiacritized:"

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
    )

    # The response from the API will be in the form of a conversation, so we grab the last message
    diacritized = response.choices[0].message.content

    return jsonify({"diacritized_text": diacritized})

if __name__ == '__main__':
    app.run(debug=True)

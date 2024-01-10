from flask import Flask, request, render_template, url_for, redirect, session, jsonify
from FinanceOfficer import *
app = Flask(__name__)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        income = request.form['income']
        federal_taxes_withheld = request.form['federal_taxes_withheld']
        social_security_wage= request.form['social_security_wage']
        social_security_tax_withheld = request.form['social_security_tax_withheld']
        medicare_wages_and_tips = request.form['medicare_wages_and_tips']
        medicare_tax_withheld= request.form['medicare_tax_withheld']
        user_input_dict = {
            "income": income,
            "federal_taxes_withheld": federal_taxes_withheld,
            "social_security_wage": social_security_wage,
            "social_security_tax_withheld": social_security_tax_withheld,
            "medicare_wages_and_tips":  medicare_wages_and_tips,
            "medicare_tax_withheld": medicare_tax_withheld
        }
        refundVal = chatbot(user_input_dict)
        return {
        'refund': refundVal
        }
    else:
        return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)





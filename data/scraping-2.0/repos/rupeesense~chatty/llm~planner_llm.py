from langchain.llms.huggingface_text_gen_inference import HuggingFaceTextGenInference

llm = HuggingFaceTextGenInference(
    inference_server_url="http://localhost:8080",
    temperature=0.5,
    timeout=300,  # 5 minutes
)

house_loan_opt = '''
Different combinations of down payment and loan tenure are:

Down Payment: 20% (₹2800000.00), Loan Tenure: 10 years
Loan Amount: ₹11200000.00
Monthly EMI: ₹140365.96
Total Interest Paid: ₹5643915.26

Down Payment: 20% (₹2800000.00), Loan Tenure: 15 years
Loan Amount: ₹11200000.00
Monthly EMI: ₹111938.25
Total Interest Paid: ₹8948884.80

Down Payment: 20% (₹2800000.00), Loan Tenure: 20 years
Loan Amount: ₹11200000.00
Monthly EMI: ₹98975.60
Total Interest Paid: ₹12554143.86

Down Payment: 20% (₹2800000.00), Loan Tenure: 30 years
Loan Amount: ₹11200000.00
Monthly EMI: ₹88110.45
Total Interest Paid: ₹20519760.35

Down Payment: 25% (₹3500000.00), Loan Tenure: 10 years
Loan Amount: ₹10500000.00
Monthly EMI: ₹131593.09
Total Interest Paid: ₹5291170.56

Down Payment: 25% (₹3500000.00), Loan Tenure: 15 years
Loan Amount: ₹10500000.00
Monthly EMI: ₹104942.11
Total Interest Paid: ₹8389579.50

Down Payment: 25% (₹3500000.00), Loan Tenure: 20 years
Loan Amount: ₹10500000.00
Monthly EMI: ₹92789.62
Total Interest Paid: ₹11769509.86

Down Payment: 25% (₹3500000.00), Loan Tenure: 30 years
Loan Amount: ₹10500000.00
Monthly EMI: ₹82603.54
Total Interest Paid: ₹19237275.33

Down Payment: 30% (₹4200000.00), Loan Tenure: 10 years
Loan Amount: ₹9800000.00
Monthly EMI: ₹122820.22
Total Interest Paid: ₹4938425.85

Down Payment: 30% (₹4200000.00), Loan Tenure: 15 years
Loan Amount: ₹9800000.00
Monthly EMI: ₹97945.97
Total Interest Paid: ₹7830274.20
'''

# print(llm.predict(
#     "<s>[INST]You are a finance advisor and have to help the user with their finance queries. "
#     "Can you help me financially plan for a house purchase? I am planning to buy a house of cost 1 crore and 40 lakhs, have around 5 lakhs in savings and monthly income of 2 lakhs. "
#     + house_loan_opt + "[/INST]"))

print(llm.predict(
    "<s>[INST]You are a finance advisor and have to help the user with their finance queries. "
    "Do you know LIC's Jeevan Anand Policy?"
    + "[/INST]"))

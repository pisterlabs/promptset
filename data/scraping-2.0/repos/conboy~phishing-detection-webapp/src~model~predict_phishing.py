from joblib import load
from openai import OpenAI

# Function that returns probability of email being a phishing attack
def predict_phishing(email_text):
    # Load the phishing logistic regression model
    log_reg_model = load('model/log_reg_model.joblib')

    # Load the vectorizer to vectorize incoming emails
    tfidf_vectorizer = load('model/tfidf_vectorizer.joblib')

    # Vectorize email text so log reg model can interpret
    transformed_email = tfidf_vectorizer.transform([email_text])

    # Predict if email is phishing attack
    phishing_probability = log_reg_model.predict_proba(transformed_email)[0][1]
    return phishing_probability

def analyze_email(email_text):
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a cyber security analyst, skilled in analyzing phishing emails."},
            {"role": "user", "content": f"Analyze this phishing email:\n\n {email_text}"}
        ]
    )
    return completion.choices[0].message.content



# # Sample safe email prediction
# prediction = analyze_email("Dear Zach, I hope this email finds you well. I am writing to request a meeting to discuss [specific topic or project]. I believe a face-to-face discussion would be the most effective way to address [specific points or objectives]. Could we schedule a meeting for next week at your convenience? Please let me know your available times, and I'll do my best to accommodate. Thank you for considering my request. I look forward to our discussion. Best regards, Conrad")
# print(prediction)

# # Sample phishing email predication
# prediction = analyze_email("Dear Valued Customer, We have detected unusual activity on your account and require you to verify your information immediately to prevent your account from being suspended. This is for your protection due to a breach in our security system. Please click on the link below to confirm your account details: cobc.com Failure to verify your account within 24 hours of receiving this email will result in account suspension for security reasons. We apologize for any inconvenience and appreciate your prompt attention to this matter. Best Regards, CIBC Security Team Note: Do not ignore this message to avoid account suspension.")
# print(prediction)
from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

template = """You're a cover letter writer expert. Use the details and instructions below to write me a professional and tailored cover letter.


My name: Talal Alqadi
My address: San Diego, CA
My email: taq.alqadi@gmail.com

Details:
Job position, company name, and job description: {question}
Resume and Information: {context}

Instructions:
1. FOCUS ON THE JOB DESCRIPTION! But you don't have to address every point in the job description.
2. Figure out what skills that job requires, and match it with the provided information.
3. Use the details provided as a guide, but DO NOT COPY DIRECTLY. Paraphrase and present the information in a unique and original manner.
4. DON'T make up any information or an answer.
5. You don't have to address every point in the job description.
6. If possible, focus on my most recent experience
7. If a skill is not listed in the provided information, then I don't have experience in it.
8. Follow the cover letter instructions.

This is some information for writing an effective cover letter:

Cover letter is a writing sample and a part of the screening process. By putting your best foot forward, you can increase your chances of being interviewed. A good way to create a response-producing cover letter is to highlight your skills or experiences that are most applicable to the job or industry and to tailor the letter to the specific organization you are applying to.
Some general rules about letters:
• Address your letters to a specific person if you can.
• Tailor your letters to specific situations or organizations by doing research before writing your letters.
• Keep letters concise and factual, no more than a single page. Avoid flowery language.
• Give examples that support your skills and qualifications.
• Put yourself in the reader’s shoes. What can you write that will convince the reader that you are ready and able to do the job?
• Don’t overuse the pronoun “I”.
• Remember that this is a marketing tool. Use lots of action words.
• Reference skills or experiences from the job description and draw connections to your credentials.
• Make sure your resume and cover letter are prepared with the same font type and size.
• Make the addressee want to read your resume.
• Be brief, but specific.
• Ask for a meeting.
• End the cover letter with a nice statement about their company reputation or why you’d like to
work for them specifically.

Cover Letter:"""
QA_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])


def get_chain(vectorstore):
    llm = OpenAI(
        streaming=True, temperature=0.2, max_tokens=3000, model_name="gpt-4"
    )
    chain_type_kwargs = {"prompt": QA_PROMPT}
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
    )

    return qa_chain

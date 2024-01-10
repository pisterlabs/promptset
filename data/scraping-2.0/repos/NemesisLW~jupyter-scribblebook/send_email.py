# Display the details of the shortlisted candidates
# Then, Button to send emails to them.

from langchain.llms import OpenAI
from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.agents import AgentType






def send_emails(llm,zapier, names,contact_info,st_callback):
    toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
    agent = initialize_agent(toolkit.get_tools(), llm, agent="zero-shot-react-description", verbose=True)



    agent.run(f"""Your task is to send personalized congratulatory emails to the selected candidates who are {names}, informing them about their selection and the next steps in the hiring process.

    Please craft individual emails for each candidate, addressing them by their name. You can find the names of the caand including specific details about their selection and the next steps. Your emails should be professional, concise, and well-written, demonstrating enthusiasm for their selection and providing clear instructions on what they need to do next.

    Please note that each email should be unique and tailored to the individual candidate. You should avoid using any generic or template language. Instead, personalize each email by mentioning specific qualifications, experiences, or accomplishments that stood out during the selection process. Additionally, feel free to include any relevant information about the company, team, or role that may be of interest to the candidate.

    You may consult the following JSON object to gain specific information about the email addresses of each candidate: 

    {contact_info}

    Ensure that the emails are error-free, have a professional tone, and are formatted correctly. Check the names and emails of the candidates to ensure accuracy before sending the emails.

    Your goal is to make each candidate feel appreciated, valued, and excited about the next steps in the hiring process.

    NO NEED to CC anyone. Make sure you do not include "[Candidate's Email]" in the "To" section.
    Make sure you have sent the emails to every eligible candidates selected.
    """, callbacks=[st_callback])
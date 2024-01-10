import streamlit as st
import openai
from streamlit_chat import message

openai.api_key = "INSERT OPENAI KEY"

def med_chat(st):
    # Initialise session state variables
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    if 'model_name' not in st.session_state:
        st.session_state['model_name'] = []
    if 'analysis' not in st.session_state:
        st.session_state['analysis'] = ""  # Add a session variable to store analysis result
    if 'provided' not in st.session_state:
        st.session_state['provided'] = False
        st.session_state['situation'] = ''
        st.session_state['involved'] = ''
        st.session_state['documents'] = ''
        st.session_state['expectations'] = ''

    # Generate a response
    def generate_response(prompt):
        st.session_state['messages'].append({"role": "user", "content": prompt})

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=st.session_state['messages'],
            max_tokens=256,
            temperature=0.6
        )
        response = completion.choices[0].message.content.strip()
        st.session_state['messages'].append({"role": "assistant", "content": response})

        return response
    
    def analyze_propmpt(prompt):
        st.session_state['messages'].append({"role": "user", "content": prompt})

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=st.session_state['messages'],
            max_tokens=3000,
            temperature=0.1
        )
        response = completion.choices[0].message.content.strip()
        st.session_state['messages'].append({"role": "assistant", "content": response})

        return response

    # Analyze user messages
    def generate_analysis():
        user_messages = st.session_state['past'] 
        merged_user_messages = " ".join(user_messages)  # Merge the user messages into a single prompt

        st.session_state['messages'] = [
            {"role": "user", "content": merged_user_messages},
            {"role": "assistant", "content": f"""
                                                ChatGPT, you are now in the analysis phase of this legal consultation under Philippine Jurisdiction. Focus on Philippine Law. Your role is to review and comprehend all the information shared by the client during the previous interactions. Here are the key steps you should follow:
                                                Summarize the Situation: Start by providing a brief, understandable summary of the client's situation. This summary should not include any legal interpretations yet. It should simply reflect the facts and details shared by the client to ensure that you have correctly understood their circumstances.
                                                Legal Analysis: Next, perform a legal analysis of the client's situation based on the information they provided. Use precise and accurate legal terms to explain the potential legal implications, rights, responsibilities, and potential legal avenues available to the client. This analysis should be based on the legal norms, precedents, and regulations that apply to the case. Make sure to explain these terms in a way that is comprehensible to a layperson.
                                                Course of Action: Finally, provide a list of possible actions the client could take to address their situation. This should be based on the legal analysis you've done and should consider the client's specific needs and circumstances. Discuss the potential benefits, risks, and consequences of each option, and remind the client that your suggestions are not definitive legal advice and should be reviewed with a qualified legal professional.
                                                Remember to communicate empathetically and professionally throughout this process. Your goal is to help the client understand their situation better and guide them towards possible next steps.
                                           """}
        ]

        st.session_state['analysis'] = analyze_propmpt("")  # Update the analysis session variable

    st.image("jurisai_logo.png")
    st.markdown("Developed by [Eys](https://www.linkedin.com/in/acecanacan/)", unsafe_allow_html=True)
    st.markdown('')


    if not st.session_state['provided']:
        with st.form(key='context_form'):

            st.write('Before you begin, please fill out these forms to help the AI understand you better and provide more meaningful support based on your current situation.')

            st.session_state['situation'] = st.text_area("Can you briefly describe the situation? What is its current status and how has it developed?", value=st.session_state['situation'])
            st.session_state['involved'] = st.text_area("Who are the parties involved in this situation? Are there any potential witnesses?", value=st.session_state['involved'])
            st.session_state['documents'] = st.text_area("What legal documents do you have related to this situation? Can you briefly describe their contents?", value=st.session_state['documents'])
            st.session_state['expectations'] = st.text_area("What do you hope to get out of this consultation?", value=st.session_state['expectations'])

            if st.form_submit_button("Submit"):
                st.session_state['messages'].append({"role": "system", "content": f"""Here is the situation of the client: {st.session_state['situation']}. These are the people involved: {st.session_state['involved']}. Here are the documents and whats included: {st.session_state['documents']} Here is what they expect to happen {st.session_state['expectations']}
                                                                                        ChatGPT, you are an empathetic legal consultant in the Philippines so focus on Philippine Laws. Your primary role is to gain a deeper understanding of the client's situation by asking relevant questions. Your objective is not to provide advice or definitive information but to probe the circumstances and gather more details about the issue at hand.
                                                                                        Ensure you build a complete picture by clarifying any ambiguities or uncertainties in the client's responses. 
                                                                                        Ask about the current status of the issue, the factors that led to the present situation, the client's thoughts and feelings, and any additional information that might be relevant to the case.
                                                                                        However, remember to respect the client's comfort and time, evaluate whether you've obtained sufficient information to understand the situation.
                                                                                        When the conversation has become extensive or you believe you've collected enough details, kindly ask the client if they have anything more they'd like to discuss or clarify. 
                                                                                        I highly emphasize that you are not here to provide any legal advice to the client, you are just there to ask questions to get details about their situation.

                                                                                        CHAT GPT IT IS IMPERATIVE THAT YOU DO NOT GIVE MORE THAN ONE QUESTION PER ANSWER OR MESSAGE YOU SEND.
                                                                                        DO NOT OVERWHELM THE CLIENT BY PROVIDING A BARRAGE OF QUESTIONS
                                                                                        ASK ONE QUESTION AT A TIME
                                                                                    """})
                st.session_state['provided'] = True
                st.experimental_rerun()


    if st.session_state['provided']:

        st.write('You can now begin your AI consultation. Say Hi!.')

        # Container for chat history
        response_container = st.container()
        # Container for text box
        container = st.container()



        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    if i < len(st.session_state["past"]):
                        message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                    message(st.session_state["generated"][i], key=str(i))

        # Add a new section for displaying the analysis
        if st.session_state['analysis']:
            st.markdown(f"## Analysis")
            st.markdown(st.session_state['analysis'])

        # New Session button
        if st.button("New Session"):
            st.session_state['generated'] = []
            st.session_state['past'] = []
            st.session_state['messages'] = []
            st.session_state['model_name'] = []
            st.session_state['analysis'] = ""  # Reset the analysis session variable
            st.session_state['provided'] = []
            st.session_state['situation'] = ''
            st.session_state['involved'] = ''
            st.session_state['documents'] = ''
            st.session_state['expectations'] = ''
            st.experimental_rerun()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_area("You:", key='input', height=100)
                submit_button = st.form_submit_button(label='Send')


            if submit_button and user_input:
                output = generate_response(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)
                st.session_state['model_name'].append("GPT-3.5")
                st.experimental_rerun()


            # Replace the if condition for the analysis button with a permanent button
            analysis_button = st.button("Generate Analysis")
            if analysis_button:
                generate_analysis()
                st.experimental_rerun()


if __name__ == "__main__":
    med_chat(st)

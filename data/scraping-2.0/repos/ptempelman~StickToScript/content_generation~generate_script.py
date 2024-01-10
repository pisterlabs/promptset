from langchain.chat_models import ChatOpenAI


def generate_script(api_key, script_scenario):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", api_key=api_key)

    prompt = (
        """Please create a script for a salesperson. It should only reflect things the salesperson 
        should say. It should not take into account how the client might respond. It should just 
        be a one-sided sales script. Start each new piece of text with 'Salesperson: '. 
        Specifically, it should be a script for: """
        + script_scenario
    )
    return llm.predict(prompt)

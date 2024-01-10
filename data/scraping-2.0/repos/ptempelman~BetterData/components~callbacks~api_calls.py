from langchain.chat_models import ChatOpenAI
import re


def get_generated_graph(api_key, df, prompt):
    llm = ChatOpenAI(model_name="gpt-4", api_key=api_key)

    padded_prompt = f"""We will use your output directly to put into a dash plotly app, so only return code. 
        Only return a dcc.Graph component, nothing else. You have access to df which has columns:
        {df.columns}. Create everthing you need, including a px figure, inside the dcc.Graph, because we will 
        only extract that component. So the fig and everything else you need must be created INSIDE the Dcc.Graph. 
        The graph should be based on: {prompt}, use plotly express, px, to create the figure and give it 
        template='plotly_dark', also give the dcc.Graph component: 
        className="main-graph", and 
            config=
                "displaylogo": False,
                "modeBarButtonsToRemove": [
                    "zoom",
                    "pan",
                    "select2d",
                    "lasso2d",
                    "autoscale","""
    pred = llm.predict(padded_prompt)
    print(pred)
    trimmed_pred = extract_dcc_graph(pred)
    print(trimmed_pred)
    return trimmed_pred[0]


def extract_dcc_graph(code_str):
    # Pattern to find dcc.Graph(...
    pattern = r"dcc\.Graph\("

    # Find all matches of the pattern
    matches = [m for m in re.finditer(pattern, code_str)]

    if not matches:
        return "No dcc.Graph component found."

    components = []

    for match in matches:
        start_index = match.start()
        # Use a stack to find the matching closing bracket
        stack = []
        for i in range(start_index, len(code_str)):
            if code_str[i] == "(":
                stack.append("(")
            elif code_str[i] == ")":
                if stack:
                    stack.pop()
                if not stack:
                    # Found the matching closing bracket
                    components.append(code_str[start_index : i + 1])
                    break

    return components

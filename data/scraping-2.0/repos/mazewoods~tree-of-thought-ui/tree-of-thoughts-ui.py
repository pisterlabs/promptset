from tree_of_thoughts.treeofthoughts import OpenAILanguageModel, CustomLanguageModel, TreeofThoughts, OptimizedOpenAILanguageModel, OptimizedTreeofThoughts
import streamlit as st

use_v2 = False
api_key= st.text_input("Enter your API key")
api_base= "" # leave it blank if you simply use default openai api url
model = None

if not use_v2:
    #v1
    if api_key:
        model = OpenAILanguageModel(api_key=api_key, api_base=api_base)
else:
    #v2 parallel execution, caching, adaptive temperature
    if api_key:
        model = OptimizedOpenAILanguageModel(api_key=api_key, api_base=api_base)

search_algorithm = st.selectbox("Choose an algorithm", ["DFS", "BFS"])
strategy = st.selectbox("Choose strategy", ["propose", "cot"])
evaluation_strategy = st.selectbox("Choose evaluation strategy", ["vote", "value"])

if not use_v2:
    #create an instance of the tree of thoughts class v1
    if model:
        tree_of_thoughts = TreeofThoughts(model, search_algorithm)
else:
    #or v2 -> dynamic beam width -< adjust the beam width [b] dynamically based on the search depth quality of the generated thoughts
    if model:
        tree_of_thoughts= OptimizedTreeofThoughts(model, search_algorithm)

input_problem = st.text_area("Enter your problem here")

# input_problem = "tomorrow is my mothers birthday, she likes the following things: flowers, the color orange. she dislikes the following things: the color blue, and roses. what present should I get her?"
k = st.slider("Choose k", 1, 10, 5)
T = st.slider("Choose T", 1, 10, 3)
b = st.slider("Choose b", 1, 10, 5)
vth = st.slider("Choose vth", 0.1, 1.0, 0.5)


if st.button("Solve"):
    st.write("Solving...")
    #call the solve method with the input problem and other params
    solution = tree_of_thoughts.solve(input_problem, k, T, b, vth, )

    #use the solution in your production environment
    st.code(solution)

#call the solve method with the input problem and other params
# solution = tree_of_thoughts.solve(input_problem, k, T, b, vth, )

#use the solution in your production environment
#print(solution)

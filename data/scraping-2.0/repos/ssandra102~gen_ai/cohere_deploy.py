import cohere
import streamlit as st

co = cohere.Client("COHERE_API_KEY")

def generate_description(proj_idea, creativity):
    """
    Generate resume project description given project idea
    Arguments:
    project(str): the project idea
    temperature(str): the Generate model `temperature` value
    Returns:
    response(str): the resume description
    """
    idea_prompt = f"""Generate a resume project description given project idea. Here are a few examples.

                    --
                    project: Calculator App
                    project description: Developed and implemented a feature-rich Calculator app for iOS and Android platforms, showcasing advanced mathematical functionalities and a user-friendly interface.

                    --
                    project: Snake Game
                    project description: Designed, developed, and deployed a dynamic and engaging Snake Game for mobile platforms, showcasing expertise in game mechanics, user experience, and performance optimization.

                    --
                    project: Car price prediction
                    project description: Led the development of a machine learning-based Car Price Prediction system, leveraging predictive modeling techniques to estimate the market value of vehicles.

                    --
                    project:{proj_idea}
                    project description: """

    # Call the Cohere Generate endpoint
    response = co.generate(
        model="command",
        prompt=idea_prompt,
        max_tokens=50,
        temperature=creativity,
        k=0,
        stop_sequences=["--"],
    )
    
    description = response.generations[0].text
    print(idea_prompt)
    print("description - pre", description)
    description = description.replace("\n\n--", "").replace("\n--", "").strip()
    print("description - post", description)
    print("-------------")
    return description

# The front end code starts here

st.title("🚀 Resume Description Generator")
st.write("""
            Enter your project idea below, an generate a description that is resume worthy !!
            """)
st.markdown("""---""")

form = st.form(key="user_settings")
with form:
    # User input - project name
    proj_idea = st.text_input("Project", key="proj_idea")

    # Create a two-column view
    col1, col2 = st.columns(2)
    with col1:
        # User input - The number of ideas to generate
        num_input = st.slider(
            "Number of descriptions",
            value=3,
            key="num_input",
            min_value=1,
            max_value=10,
            help="Choose to generate between 1 to 10 ideas",
        )
    with col2:
        # User input - The 'temperature' value representing the level of creativity
        creativity_input = st.slider(
            "Creativity",
            value=0.5,
            key="creativity_input",
            min_value=0.1,
            max_value=0.9,
            help="Lower values generate more “predictable” output, higher values generate more “creative” output",
        )
    # Submit button to start generating ideas
    generate_button = form.form_submit_button("Generate Idea")

    if generate_button:
        if proj_idea == "":
            st.error("Project field cannot be blank")
        else:
            my_bar = st.progress(0.05)
            st.subheader("Project Descriptions:")

            for i in range(num_input):
                st.markdown("""---""")
                idea = generate_description(proj_idea, creativity_input)
                # name = generate_name(idea, creativity_input)
                # st.markdown("##### " + name)
                st.write(idea)
                my_bar.progress((i + 1) / num_input)

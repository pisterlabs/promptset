# Import necessary libraries
import streamlit as st
import pandas as pd
import openai
import re  # Importing the regular expressions module

# Title of the app
st.title('Auto-Clean')

def verify_api_key(api_key):
    return bool(api_key)

# If API key is not in session state, display input for API key
if 'api_key' not in st.session_state:
    entered_api_key = st.text_input("Enter OpenAI API Key:", type="password")
    if st.button("Verify API Key"):
        if verify_api_key(entered_api_key):
            st.session_state['api_key'] = entered_api_key
            openai.api_key = entered_api_key
            st.success("API Key verified successfully!")
            st.experimental_rerun()
        else:
            st.error("Invalid API Key!")

# If API key is in session state (i.e., verified), display the rest of the app's functionality
elif 'api_key' in st.session_state:
    # Function to generate a response from OpenAI
    def generate_response(prompt):
        if 'messages' not in st.session_state:
            st.session_state['messages'] = []

        st.session_state['messages'].append({"role": "system", "content": prompt})

        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=st.session_state['messages'],
            max_tokens=1000,
            temperature=0.1
        )
        return completion.choices[0].message['content']

    def analyze_column(column_data, column_name):
        # Convert a sample of the column data to string format for the prompt
        sample_data = '\n'.join(column_data.head(10).astype(str).tolist())

        # Construct the prompt for the model
        prompt = (f"Please analyze the following sample data from the column named '{column_name}':\n\n{sample_data}\n\n"
                "Do not create a summary. The sample should Identify key issues and provide cleaning suggestions for each issue"
                "Make multiple bullets that include the issue and suggested cleaning method"
                "The output should look exactly like this. The output should look exactly like this"
                "Issue: The value abc is not a valid integer.\nSuggested cleaning method:Remove or replace the non-integer value with a valid integer value")
        return prompt

    def generate_code_for_suggestion(suggestion):
        # Construct a prompt for the model to generate code based on the suggestion
        prompt = (f"Please provide a Python code block to implement the following data cleaning suggestion:\n\n{suggestion}"
                "Be concise with the output, just include the necessary code."
                "Do not put anything else, just the block of code. Put comments within"
                "Do not put anything else,")
        # Get the code block from the model
        response = generate_response(prompt)
        
        
        # Extract the code from between the triple backticks
        code_block = re.search(r"```python\n(.*?)\n```", response, re.DOTALL)
        if code_block:
            return code_block.group(1).strip()  # Return only the code without the backticks
        else:
            return response  # If no code block is found, return the entire response


    # Upload CSV functionality
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        # Load the CSV into a pandas DataFrame
        data = pd.read_csv(uploaded_file)

        # Displaying the column headers using radio buttons
        selected_column = st.radio(
            'Select the column you want to analyze:',
            options=data.columns.tolist()
        )

        # Display the selected column in a scrollable container with a set height
        with st.container():
            st.write(data[selected_column].head(100))
            st.markdown('<style>.stContainer { overflow-y: auto; max-height: 300px; }</style>', unsafe_allow_html=True)

        def parse_suggestions(response):
            # Split the response by lines to handle the situation where the issue and suggestion are on separate lines
            lines = response.split("\n")

            suggestions = []
            current_issue = None

            for line in lines:
                # Check if the line contains "Issue:" (with or without the bullet point)
                if re.search(r"(?:\* )?Issue:", line):
                    # If we already have an issue stored, but no corresponding suggestion was found, we add the issue to the suggestions list
                    if current_issue:
                        suggestions.append(current_issue)
                    current_issue = line.strip()
                # Check if the line contains "Suggested cleaning method:" (with or without the bullet point)
                elif re.search(r"(?:\* )?Suggested cleaning method:", line):
                    # If we have an issue stored, we combine it with the suggestion and add to the suggestions list
                    if current_issue:
                        combined_suggestion = f"{current_issue} {line.strip()}"
                        suggestions.append(combined_suggestion)
                        current_issue = None
                    else:
                        suggestions.append(line.strip())

            # If there's an issue left without a corresponding suggestion, we add it to the suggestions list
            if current_issue:
                suggestions.append(current_issue)

            return suggestions

        if st.button('Analyze Column'):
            try:
                # Analyze the selected column
                column_data = data[selected_column]
                prompt = analyze_column(column_data, selected_column)

                # Get the cleaning suggestions from the model
                response = generate_response(prompt)

                # Parse the suggestions
                suggestions = parse_suggestions(response)

                # Store the suggestions in the session state
                st.session_state['suggestions'] = suggestions
                st.session_state['ai_suggestions_done'] = True
            except Exception as e:
                st.write(f"Error: {e}")

        if 'suggestions' in st.session_state:
            # Display the suggestions and their corresponding code blocks
            for suggestion in st.session_state['suggestions']:
                # Display the suggestion in a text box
                user_input = st.text_area(f"'{selected_column}':", suggestion.strip())
                
                # Create a placeholder for the code block
                code_placeholder = st.empty()
                
                # Check if code for this suggestion is already generated and stored in session state
                if f'code_{user_input}' in st.session_state:
                    code_placeholder.code(st.session_state[f'code_{user_input}'])
                
                # Generate button for each text box-code block pair
                if st.button('Generate', key=user_input):  # Using the suggestion as a unique key for the button
                    # Generate code ONLY for the content of the text box
                    updated_code_block = generate_code_for_suggestion(user_input)
                    st.session_state[f'code_{user_input}'] = updated_code_block
                    code_placeholder.code(updated_code_block)  # Update the content of the placeholder with the new code block

                # Delete button for each text box-code block pair
                if st.button('Delete', key=f"delete_{user_input}"):  # Using a unique key for the delete button
                    # Remove the suggestion from the list
                    st.session_state['suggestions'].remove(suggestion)
                    # Refresh the display by rerunning the script
                    st.experimental_rerun()


        if 'ai_suggestions_done' in st.session_state and st.session_state['ai_suggestions_done']:
            # Allow users to add their own cleaning suggestions
            new_suggestion = st.text_input("Add your own cleaning suggestion:")

            # Initialize the dictionary to store generated code for each suggestion if it doesn't exist
            if 'generated_codes' not in st.session_state:
                st.session_state['generated_codes'] = {}

            # Check if code for this new suggestion is already generated
            if new_suggestion in st.session_state['generated_codes']:
                st.code(st.session_state['generated_codes'][new_suggestion])
            if st.button('Generate for new suggestion'):
                # Generate code for the new suggestion
                new_code_block = generate_code_for_suggestion(new_suggestion)
                st.session_state['generated_codes'][new_suggestion] = new_code_block
                st.code(new_code_block)

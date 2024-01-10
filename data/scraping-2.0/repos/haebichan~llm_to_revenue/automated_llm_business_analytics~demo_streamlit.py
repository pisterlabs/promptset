import streamlit as st
import pandas as pd
import random
import openai
import os

def sample_df_with_fully_unique_row_across_all_columns(df):
    # Initialize an empty DataFrame to store the sampled rows
    sampled_df = pd.DataFrame(columns=df.columns)
    
    # Initialize a set to keep track of sampled rows as tuples
    sampled_rows = set()
    
    # Loop through the rows and add them to the sampled_df while checking for uniqueness
    for index, row in df.iterrows():
        row_tuple = tuple(row)
        if row_tuple not in sampled_rows:
            sampled_df = pd.concat([sampled_df, row.to_frame().T])
            sampled_rows.add(row_tuple)
    
        # Check if you have enough unique rows; you can adjust the number as needed
        if len(sampled_df) >= 5:
            break
    
    return sampled_df


# Replace 'YOUR_API_KEY' with your actual OpenAI API key
api_key = os.environ['OPENAI_API_KEY']

# Initialize the OpenAI API client
openai.api_key = api_key

def generate_text(prompt, temperature=0.1):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # You can choose an appropriate engine
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        return str(e)



# Generate a list of departments
departments = ['HR', 'Finance', 'Marketing', 'Sales', 'Engineering']

# Create an empty list to store data
data = []

# Generate dummy data
for _ in range(100):
    month = random.choice(['January', 'February', 'March'])
    department = random.choice(departments)
    employee_name = f'Employee{_ + 1}'
    spending_amount = round(random.uniform(1000, 5000), 2)
    
    # Check for duplicate employee names in different departments
    while any((d['Employee Name'] == employee_name) and (d['Department'] != department) for d in data):
        employee_name = f'Employee{_ + 1}'
    
    data.append({
        'Month': month,
        'Department': department,
        'Employee Name': employee_name,
        'Spending Amount': spending_amount
    })

# Create a DataFrame
df = pd.DataFrame(data)


#####


sampled_data = sample_df_with_fully_unique_row_across_all_columns(df)


st.title("LLM Analytics Dashboard")

user_query = st.text_area("Type Request for Visual Chart Creation: ", height = 25)

prompt = f"""Generate code for this request "Create an appropriate visual to show {user_query}"

Sample Data:
{sampled_data}  # Define or replace sampled_data with your actual data

Instructions for Code Generation:
1. Write code to filter the data for the specified column and values when necessary.
2. Assume data is just called "df". Don't create any dummy data on your own
3. Use matplotlib (plt) to generate visuals
4. Just give me the code as output"""

if st.button('Execute Request'):
    if user_query:
        generated_ai_output = generate_text(prompt = prompt)

        exec_globals = {'df':df}
        
        exec(generated_ai_output, exec_globals)

        if 'plt' in exec_globals:
            figure = exec_globals['plt'].gcf()

            st.pyplot(figure)


st.write(sampled_data)











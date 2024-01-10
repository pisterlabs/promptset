import openai
import pandas as pd

# Set the OpenAI API Key
openai.api_key = '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'

def get_business_idea(country, countryOfResidence):
    # Generate a business idea based on the country
    prompt = f"Generate a business idea for people from {country} living in {countryOfResidence}, please provide three examples as to why they are great business ideas. "
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        temperature=0.7,
        max_tokens=100
    )

    idea = response.choices[0].text.strip()
    return idea

# Load the dataset
file_path = 'Datasets\Permanent Resident - Country of Citizenship.xlsx'
df = pd.read_excel(file_path)

# Clean the data
df = df.dropna().reset_index(drop=True)
df = df[['Unnamed: 1', 'Unnamed: 3', 'Unnamed: 4']]
df.columns = ['Country', 'Percentage', 'Base Count']
df = df.iloc[1:]
df['Percentage'] = pd.to_numeric(df['Percentage'], errors='coerce')

# Exclude the total count row and get the specific country with the max percentage
specific_country_df = df.iloc[1:]
specific_country_max_percentage = specific_country_df.loc[specific_country_df['Percentage'].idxmax()]

# Print the country with the max percentage and its base count
print(f"Country with max percentage: {specific_country_max_percentage['Country']}")
print(f"Percentage: {specific_country_max_percentage['Percentage']}")
print(f"Base Count: {specific_country_max_percentage['Base Count']}")
countryA = (f"{specific_country_max_percentage['Country']}").strip()
print(countryA)
# Example Usage
country = countryA
countryOfResidence = "Canada"
idea = get_business_idea(country, countryOfResidence)
print(f"Business idea for people from {country} living in {countryOfResidence}: {idea}")
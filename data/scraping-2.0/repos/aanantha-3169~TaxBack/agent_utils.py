import openai
import pandas as pd


def get_query(question):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=[
        {
        "role": "system",
        "content": "Given the list of items, classify a users item as either {item_category} if it meets the description of the items listed or \"False\" otherwise. For {item_category} cases, the output should return the category name eg. \"education\" where education is the category name from the mapping table provided below. Only have one output per user query and note that the category name is case sensitive.\nHere is the mapping table(category|Description):parent_care|Medical treatment, special needs, and carer expenses for parents(Includes non-cosmetic dental care as well as nursing home care and therapy.)\n education|Education fees(Any course of study up to tertiary level undertaken for law, accounting, Islamic finance, technical, vocational, industrial, scientific, or technical skills or Masters or Doctorate or A course of study undertaken for up-skilling)\n education|Education fees(Masters or Doctorate)\n education|Education fees(A course of study undertaken for up-skilling)\n medical|Medical expenses(Medical expenses on serious diseases eg.IDS, Parkinson’s disease, cancer, renal failure, leukaemia, heart attack, pulmonary hypertension, chronic liver disease, fulminant viral hepatitis, head trauma with neurological deficit, tumour in brain or vascular malformation, major burns, major organ transplant, and major amputation of limbs)\nmedical|Medical expenses( Medical expenses for fertility treatment)\n3.Medical expenses(Vaccination expenses)\nmedical|Medical expenses(Complete medical and mental health examination)\nlifestyle|Lifestyle purchases(Books, journals, magazines, printed newspapers, and other similar publications in both hardcopy and electronic forms; banned and offensive materials excluded)\nlifestyle|Lifestyle purchases(Personal computers, smartphones or tablets eg. iphone,macbook, asus laptop, galaxy fold etc.)\nlifestyle|Lifestyle purchases(sports equipment for sports activities defined under the Sports Development Act 1997, including golf balls and shuttlecocks, and payment for a gym membership)\nlifestyle|Lifestyle purchases(Internet subscription is paid through a monthly bill)\nsports|Expenses related to sports activity (Purchase of sports equipment for any sports activity or payment of rental or entrance fees to sports facilities or Payment of registration fees for sports competitions)\npersonal_computer|Purchase of personal computers, smartphones, or tablets\ntourism|Tourist accommodation, attractions, or tour package(Accommodation at premises registered with the Commissioner of Tourism or Entrance fees for tourist attractions)\ntourism|Tourist accommodation, attractions, or tour package(Purchase of domestic tour package through licensed travel agents, inclusive of fees for tour guide services, purchase of local handicraft products, F&B, and transportation)\ntourism|Tourist accommodation, attractions, or tour package(Accommodation at premises registered with the Commissioner of Tourism or Entrance fees for tourist attractions  or Purchase of domestic tour package through licensed travel agents, inclusive of fees for tour guide services, purchase of local handicraft products, F&B, and transportation)"
        },
            {"role": "user", "content":f"{question}"}
        ],
        temperature=0,
        max_tokens=256,
        top_p=0.8,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content

def get_rebate(list_category:list,income:int) -> str:
    list_cat = list_category

    data = {
    'category': ['education', 'lifestyle', 'sports', 'personal_computer', 'tourism', 'parent_care', 'medical'],
    'max_relief': [7000, 2500, 500, 2500, 1000, 8000, 8000]
    }

    df_relief = pd.DataFrame(data)

    list_relief = df_relief.loc[df_relief['category'].isin(list_cat),'max_relief'].to_list()
    total_relief = sum(list_relief)

    taxable_income = income - (9000 + total_relief)

    def get_tax(income:int) -> int:
        data = {
            'Category': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
            'Low_Range': [0, 5001, 20001, 35001, 50001, 70001, 100001, 400001, 600001, 2000001],
            'Upper_Range': [5000, 20000, 35000, 50000, 70000, 100000, 400000, 600000, 2000000, 2000000000],
            'Tax_Rate': [0, 0.01, 0.03, 0.06, 0.11, 0.19, 0.25, 0.26, 0.28, 0.3],
            'First_Tax': [0, 150, 150, 600, 1500, 3700, 9400, 84400, 136400, 528400]
        }

        df_tax = pd.DataFrame(data)
        """Takes a users annual income as input and return the tax that the user has to pay"""
        def get_category(income:int) -> str:
            """Get category of income level from df_tax"""
            if income <= 2000000:
                category = df_tax.loc[(df_tax['Low_Range'] <= income) & (df_tax['Upper_Range'] >= income), 'Category'].iloc[0]
                return category
            elif income > 2000000:
                return 'J'

        # First, determine the category based on income using the nested function
        category = get_category(income)

        # Then proceed with tax calculation
        tax_rate = df_tax.loc[df_tax['Category'] == category,'Tax_Rate'].iloc[0]
        lower_range = df_tax.loc[df_tax['Category'] == category,'Low_Range'].iloc[0] - 1
        first_rate = df_tax.loc[df_tax['Category'] == category,'First_Tax'].iloc[0]

        final_tax = ((income - lower_range) * tax_rate) + first_rate

        return final_tax

    initial_tax = get_tax(income)
    new_tax = get_tax(taxable_income)

    str_cat = ','.join(list_cat)


    return f'By maximising your reliefs for {str_cat}; a rough estimation of the reduction to your tax burner is {int(initial_tax - new_tax)}'

def get_tax(income:int) -> int:
    """Takes a users annual income as input and return the tax that the user has to pay"""

    data = {
        'Category': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
        'Low_Range': [0, 5001, 20001, 35001, 50001, 70001, 100001, 400001, 600001, 2000001],
        'Upper_Range': [5000, 20000, 35000, 50000, 70000, 100000, 400000, 600000, 2000000, 2000000000],
        'Tax_Rate': [0, 0.01, 0.03, 0.06, 0.11, 0.19, 0.25, 0.26, 0.28, 0.3],
        'First_Tax': [0, 150, 150, 600, 1500, 3700, 9400, 84400, 136400, 528400]
    }

    df_tax = pd.DataFrame(data)

    def get_category(income:int) -> str:
        if income <= 2000000:
            category = df_tax.loc[(df_tax['Low_Range'] <= income) & (df_tax['Upper_Range'] >= income), 'Category'].iloc[0]
            return category
        elif income > 2000000:
            return 'J'

    # First, determine the category based on income using the nested function
    category = get_category(income)

    # Then proceed with tax calculation
    tax_rate = df_tax.loc[df_tax['Category'] == category,'Tax_Rate'].iloc[0]
    lower_range = df_tax.loc[df_tax['Category'] == category,'Low_Range'].iloc[0] - 1
    first_rate = df_tax.loc[df_tax['Category'] == category,'First_Tax'].iloc[0]

    final_tax = ((income - lower_range) * tax_rate) + first_rate

    return f"If your income is {income}, your tax would be {int(final_tax)}. You would fall under category {category} which is taxed RM{first_rate} for the first {lower_range} and {tax_rate} percent on the remaining income"

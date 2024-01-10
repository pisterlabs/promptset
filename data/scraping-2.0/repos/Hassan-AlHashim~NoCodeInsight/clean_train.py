import openai
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key = api_key)
model = "gpt-4-1106-preview"

def suggest_cleaning_procedure(df, target_variable):

    # Prepare a string representation of the first few rows of the dataframe
    sample_data = df.sample(min(100, len(df))).to_string(index=False)
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are developed to look at a dataset and provide a python function to clean the data called 'clean_dataframe. The code \
                should be able to accept a pandas dataframe. The function should return a cleaned \
                data frame and must be ready to be executed for analysis such as logistic and linear regression, random forest, and gradient boost. Here are a few instructions to keep in mind to ensure consistency and error free \
                execution: \
                1) Function should be able to impute missing data, remove columns with large number of missing data, handle outliers, encode \
                    categorical variables, normalize required columns.\
                2) Function must check the number of missing variables in each column and drop columns with more than 80% missing values. \
                    for columns with less than 80%, impute the data such as replacing NA point with mean of column.\
                3) handle outliers by capping them at a defined threshold rather than removing.\
                4) Encode categorical variables using one hot encoding. to avoid errors in encoding, make sure you only encode the columns with a few categorical variables, \
                    for columns with large number of categorical values, just remove them as they are not important for the analysis. When encoding, make sure the resulting columns  are numerical, we dont want 'yes or no' or 'fals or true', ensure numerical '0 or 1 or 2 or 3,...' and so on.\
                5) Normalize the data using MinMaxScaler, be sure to exclude the target variable '" + target_variable + "',as we will be needing it in original scale for predictions.\
                    Also, make sure all numeric values are between 0 to 1.\
                6) function should preserve as many original columns as possible, avoid unnecessary removal of columns.\
                7) function should return a cleaned data frame ready for model training. ensure the function ends with 'return df'.\
                8) Use the `.copy()` method for all operations to prevent 'SettingWithCopyWarning'. \
                9) Include validation checks to ensure there are no NaN or infinite values post-cleaning. \
                10) The function should silently handle errors and ensure the DataFrame is always returned. \
                11) The function should not contain print statements or comments; it should be executable as is. If comments are included, dont forget to use '#'.\
                12) to avoid 'sparse' was renamed to 'sparse_output' in version 1.2 and will be removed in 1.4, 'sparse_output' is ignorded unless you leave 'sparse' to its default value, replace 'sparse' parameter with 'sparse_output'.\
                13) Avoid the 'TypeError:' in 'thresh' parameters in drop.na function. make sure you use an integer by using int() function. \
                14) Avoid 'UnboundLocalError: cannot access local variable 'df_clean' where it is not associated with a value' \
                15) Avoid syntax errors that include '<string>', do not put any comments, if comments are included, make sure to use # instead of")
        },{
            "role": "user",
            "content": (
            "The sample dataset for cleaning looks as follows:\n" + sample_data + "\n\n As Instructed, provide a python function to clean the data called 'clean_dataframe' suitable for  \
            for machine learning applications. Function should handle missing values by imputation or removal, cap outliers, normalize data \
            (excluding target variable: '"+ target_variable +"'), and convert data types where necessary. Please ensure the code preserves as much from the original \
            data possible. Return only the function code, ready for execution.\
            "
            )
        }
    ]

    # Make a request to the OpenAI API
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1000,
        temperature=0.1
    )


    # Accessing the message content
    suggested_code = response.choices[0].message.content.strip()

    # Remove the markdown code block delimiters
    cleaned_code = suggested_code.replace("```python", "").replace("```", "").strip()
    print(cleaned_code)

    return cleaned_code

def suggest_model_training_code(cleaned_df, original_df, target_column):
    # Prepare a string representation of the first few rows of the cleaned dataframe
    cleaned_sample_data = cleaned_df.head(100).to_string(index=False)

    # Prepare a string representation of the first few rows of the original dataframe
    original_sample_data = original_df.head(100).to_string(index=False)

    # Prepare the message
    messages = [
    {
        "role": "system",
        "content": ("You are a proficient data scientist tasked with guiding beginners in predictive analysis. "
                    "Create a Python function named 'train_predictive_model'. This function should accept exactly three parameters: "
                    "1) 'cleaned_df', a cleaned pandas DataFrame ready for model training, "
                    "2) 'original_df', the original pandas DataFrame before cleaning, and "
                    "Note that get_feature_names was updated to get_feature_names_out"
                    "3) 'target_variable', a string representing the name of the target variable for prediction in 'cleaned_df'. "
                    "First, split 'cleaned_df' into features (X) and target (y). Then, split these into training and testing sets using a 20% test size and a random state of 42. "
                    "Use MinMaxScaler to scale the target variable based on the original target values from 'original_df'. "
                    "For model selection, use Linear Regression for continuous targets (more than 2 unique values) and Logistic Regression for binary targets. "
                    "Train the model on the unscaled features and the scaled target variable. "
                    "Note that'OneHotEncoder' object has no attribute 'get_feature_names'"
                    "Handle exceptions silently and save the trained model and scaler for the target variable using pickle with filenames 'predictive_model.pkl' and 'target_scaler.pkl'. "
                    "Ensure the function returns the model, scaler, and the split datasets (X_train, X_test, y_train, y_test) and ends with this return statement, including no further text or explanation."
                    "End the function after the return statement without including any further text, explanations, notes, or example usages."
                    "Anything in the response that does not peretain to the excutable python function should start with # so it gets treated as a comment even if it was outside the fucntion")
    },
    {
        "role": "user",
        "content": (f"The cleaned dataset for training the model is as follows:\n{cleaned_sample_data}\n\n"
                    f"The original dataset is:\n{original_sample_data}\n\n"
                    f"The target variable for prediction is '{target_column}'. "
                    "Provide a Python function to train a predictive model on this cleaned dataset, "
                    "strictly following the structure and functionality as described, ending only with the return statement.")
    },
]


    # Make a request to the OpenAI API
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1000,
        temperature=0
    )
 
    # Accessing the message content
    suggested_code = response.choices[0].message.content.strip()

    # Remove the markdown code block delimiters
    training_code = suggested_code.replace("```python", "").replace("```", "").strip()
 
    return training_code

 
def suggest_advanced_model_training_code(cleaned_df, original_df, target_column):

    # Prepare a string representation of the first few rows of the cleaned and original dataframes
    cleaned_sample_data = cleaned_df.head(10).to_string(index=False)
    original_sample_data = original_df.head(10).to_string(index=False)

    # Prepare the message
    messages = [
        {
            "role": "system",
            "content": ("You are an advanced data scientist. Your task is to create a versatile Python function named 'train_advanced_model'. This function will train a Random Forest model, capable of handling both regression and classification tasks, and will be equipped with robust error handling. "
                        "The function must accept three parameters: 'cleaned_df' (the cleaned DataFrame), 'original_df' (the original DataFrame), and 'target_variable' (the name of the target variable in 'cleaned_df'). "
                        "Determine the nature of the task (regression or classification) based on the properties of the target variable. For regression, use RandomForestRegressor, and for classification, use RandomForestClassifier. Always set 'oob_score=True' to ensure the Out-of-Bag error estimate is available. "
                        "Split 'cleaned_df' into features and target, then into training and testing sets (test size 20%, random state 42). "
                        "Scale the target variable using MinMaxScaler based on original target values from 'original_df'. "
                        "Include error handling to ensure the function is resilient to potential issues during the model training process. "
                        "Save the trained model and scaler using pickle with filenames 'advanced_model.pkl' and 'advanced_scaler.pkl'. "
                        "Return only model, scaler, X_train, X_test, y_train, y_test"
                        "The function should return the model, the scaler, the split datasets, and it should also be capable of generating decision paths and OOB error estimates for the trained model. "
                        "Provide a clean, executable function without any additional comments or explanations."
                        "Anything in the response that does not peretain to the excutable python function should start with # so it gets treated as a comment even if it was outside the fucntion"
                        "Strictly follow those rules. The function should not contain print statements or comments; it should be executable as is")
        },
        {
            "role": "user",
            "content": (f"The cleaned dataset for training the model is as follows:\n{cleaned_sample_data}\n\n"
                        f"The original dataset is:\n{original_sample_data}\n\n"
                        f"The target variable for prediction is '{target_column}'. "
                        "Please provide a Python function capable of training a Random Forest model on this dataset, ensuring it handles both regression and classification, includes OOB error estimates, and can generate decision paths. The function should also include error handling and save the necessary components.")
        },
    ]

    # Make a request to the OpenAI API
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1000,
        temperature=0
    )

    # Accessing and leaning the suggested code
    suggested_code = response.choices[0].message.content.strip()
    training_code = suggested_code.replace("```python", "").replace("```", "").strip()
    print(training_code)
 
    return training_code

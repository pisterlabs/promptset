import streamlit as st
import openai

st.title("Guess the Prompt!")

def verify_api_key(api_key):
    return bool(api_key)

# Check if API key is already in session state
if 'api_key' in st.session_state and st.session_state['api_key']:
    api_key_verified = True
else:
    api_key_verified = False

# If API key is not verified, display input for API key
if not api_key_verified:
    entered_api_key = st.text_input("Enter OpenAI API Key:", type="password")
    if st.button("Verify API Key"):
        if verify_api_key(entered_api_key):
            st.session_state['api_key'] = entered_api_key
            openai.api_key = entered_api_key
            api_key_verified = True
        else:
            st.error("Invalid API Key!")

if api_key_verified:


    def generate_response(prompt):
        if 'messages' not in st.session_state:
            st.session_state['messages'] = []

        st.session_state['messages'].append({"role": "system", "content": prompt})

        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=st.session_state['messages'],
            max_tokens=500,
            temperature=0.
        )
        return completion.choices[0].message['content']

    # Multi-select box for data science tasks
    options = {
        'Data Acquisition': [
            "import requests\nresponse = requests.get('https://api.example.com/data1')\ndata1 = response.json()",
            "import requests\nresponse = requests.get('https://api.example.com/data2', params={'key': 'value'})\ndata2 = response.json()",
            "import requests\nresponse = requests.post('https://api.example.com/data', data={'key': 'value'})\ndata3 = response.json()",
            "import requests\nheaders = {'Authorization': 'Bearer YOUR_TOKEN'}\nresponse = requests.get('https://api.example.com/data3', headers=headers)\ndata4 = response.json()",
            "import requests\nresponse = requests.put('https://api.example.com/update', data={'key': 'updated_value'})\nupdated_data = response.json()"
        ],
        'Data Cleaning': [
            "import pandas as pd\ndf.dropna(inplace=True)",
            "df['column_name'].fillna(df['column_name'].mean(), inplace=True)",
            "df['column_name'].fillna(df['column_name'].median(), inplace=True)",
            "df['column_name'].fillna('Unknown', inplace=True)",
            "df = df.drop_duplicates()"
        ],
        'Data Manipulation': [
            "filtered_df = df[df['column_name'] > 100]",
            "aggregated_df = df.groupby('column_name').sum()",
            "merged_df = pd.merge(df1, df2, on='common_column')",
            "df['new_column'] = df['column1'] + df['column2']",
            "df = df.pivot(index='column1', columns='column2', values='column3')"
        ],
        'Data Visualization': [
            "import matplotlib.pyplot as plt\ndf['column_name'].hist()\nplt.show()",
            "plt.scatter(df['column1'], df['column2'])\nplt.show()",
            "df['column_name'].plot(kind='box')\nplt.show()",
            "df[['column1', 'column2']].plot(kind='bar')\nplt.show()",
            "df.plot(x='column1', y='column2', kind='line')\nplt.show()"
        ],
        'Feature Engineering': [
            "df_encoded = pd.get_dummies(df, columns=['categorical_column'])",
            "from sklearn.preprocessing import StandardScaler\nscaler = StandardScaler()\ndf['scaled_column'] = scaler.fit_transform(df[['column_name']])",
            "df['log_transformed'] = df['column_name'].apply(lambda x: np.log(x))",
            "df['squared_feature'] = df['column_name'] ** 2",
            "df['interaction'] = df['column1'] * df['column2']"
        ],
        'Statistical Analysis': [
            "from scipy.stats import ttest_ind\nstat, p = ttest_ind(df['group1'], df['group2'])",
            "correlation = df['column1'].corr(df['column2'])",
            "from scipy.stats import chi2_contingency\nchi2, p, dof, expected = chi2_contingency(pd.crosstab(df['column1'], df['column2']))",
            "from scipy.stats import f_oneway\nstat, p = f_oneway(df['group1'], df['group2'], df['group3'])",
            "from scipy.stats import pearsonr\ncoeff, p = pearsonr(df['column1'], df['column2'])"
        ],
        'Machine Learning': [
            "from sklearn.linear_model import LinearRegression\nmodel = LinearRegression()\nmodel.fit(X_train, y_train)",
            "from sklearn.ensemble import RandomForestClassifier\nmodel = RandomForestClassifier()\nmodel.fit(X_train, y_train)",
            "from sklearn.cluster import KMeans\nkmeans = KMeans(n_clusters=3)\nkmeans.fit(df)",
            "from sklearn.decomposition import PCA\npca = PCA(n_components=2)\ntransformed_data = pca.fit_transform(df)",
            "from sklearn.svm import SVC\nmodel = SVC()\nmodel.fit(X_train, y_train)"
        ],
        'Time Series Analysis': [
            "from statsmodels.tsa.arima_model import ARIMA\nmodel = ARIMA(df['time_series_column'], order=(1,1,1))\nmodel_fit = model.fit(disp=0)",
            "df['rolling_mean'] = df['time_series_column'].rolling(window=3).mean()",
            "df['lagged_value'] = df['time_series_column'].shift(1)",
            "from statsmodels.tsa.stattools import adfuller\nresult = adfuller(df['time_series_column'])",
            "from statsmodels.tsa.seasonal import seasonal_decompose\nresult = seasonal_decompose(df['time_series_column'], model='multiplicative')"
        ],
        'Natural Language Processing (NLP)': [
            "from nltk.tokenize import word_tokenize\ntokens = word_tokenize('Sample text here.')",
            "from nltk.corpus import stopwords\nstop_words = set(stopwords.words('english'))\nfiltered_tokens = [word for word in tokens if word not in stop_words]",
            "from nltk.stem import PorterStemmer\nstemmer = PorterStemmer()\nstemmed_words = [stemmer.stem(word) for word in tokens]",
            "from sklearn.feature_extraction.text import CountVectorizer\nvectorizer = CountVectorizer()\nX = vectorizer.fit_transform(corpus)",
            "from nltk.sentiment import SentimentIntensityAnalyzer\nsia = SentimentIntensityAnalyzer()\nsentiment = sia.polarity_scores('Sample text here.')"
        ]
    }

    # Combine tasks for each main category
    combined_options = {main_category: ', '.join(sub_options) for main_category, sub_options in options.items()}

    # Select box for data science categories and their combined tasks
    selected_option = st.selectbox("Select a data science category:", list(combined_options.keys()))

    # Get the combined tasks for the selected category
    selected_tasks = combined_options[selected_option]

    # Convert selected category and tasks into a prompt for the model
    initial_prompt = (f"{selected_option} ({selected_tasks}): These are examples of the code you can make. Just choose one specific function"
                    "Make sure it's not exactly similar to it, make variations of it"
                    "Do not include any explanation or anything. Just the code."
                    "Just choose one specific function")

    # Check if code_block exists in session state, if not, initialize it
    if 'code_block' not in st.session_state:
        st.session_state.code_block = ""

    # Check if the "Start" button is pressed
    if st.button("Start"):
        # Generate a new code block
        st.session_state.code_block = generate_response(initial_prompt)
        # st.write(initial_prompt)

    # Display the code block
    st.write(f"Code Block:\n{st.session_state.code_block}")

    # Get user's guessed prompt
    guessed_prompt = st.text_input("Guess the prompt that generated the above code:")
    if st.button("Submit"):
        verification_prompt = f"Given the prompt '{guessed_prompt}', would the expected output be the following code?\n\n{st.session_state.code_block}"
        verification_response = generate_response(verification_prompt)
        
        # Check the model's response to determine if the guessed prompt is correct
        if "yes" in verification_response.lower():
            st.success("Correct! You got it right.")
        else:
            st.session_state['incorrect_guess'] = True
            st.error("You didn't get it right. Press 'Hint' for a clue.")

    # Display the hint button only if the guessed prompt was incorrect
    if 'incorrect_guess' in st.session_state and st.session_state['incorrect_guess']:
        if st.button("Hint"):
            hint_prompt = (f"Based on the guessed prompt '{guessed_prompt}' and the generated code:\n\n"
                        f"{st.session_state.code_block}\n\n"
                        "Provide a hint to guide the user towards the correct prompt without giving away the answer.")
            hint_response = generate_response(hint_prompt)
            st.write(f"Hint: {hint_response}")
            
            # Remove the incorrect_guess state so the hint button won't appear again
            del st.session_state['incorrect_guess']

import streamlit as st
import pandas as pd
import numpy as np
from umap import UMAP
import plotly.express as px

from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

@st.cache_data
def load_data():
    media_bias_map = {
        'HuffPost': 'left',
        'Federalist': 'right',
        'Daily Beast': 'left',
        'Alternet': 'left',
        'Breitbart': 'right',
        'New Yorker': 'left',
        'American Greatness': 'right', # from https://mediabiasfactcheck.com/american-greatness/
        'Daily Caller': 'right',
        'Daily Wire': 'right',
        'Slate': 'left',
        'Reuters': 'center',
        'Hill': 'center', # from https://mediabiasfactcheck.com/the-hill/
        'USA Today': 'left',
        'CNBC': 'left',
        'Yahoo News - Latest News & Headlines': 'left',
        'AP': 'left',
        'Bloomberg': 'left',
        'Fox News': 'right',
        'MSNBC': 'left',
        'Daily Stormer': 'right', # from https://mediabiasfactcheck.com/the-hill/
        'New York Times': 'left'
        }

    new_topics_map = {
    '#metoo':'activism',
    'abortion':'abortion',
    'black lives matter':'activism',
    'blm':'activism',
    'coronavirus':'coronavirus-and-vaccines',
    'elections-2020':'politics',
    'environment':'environment',
    'gender':'socioeconomics',
    'gun control':'gun-control',
    'gun-control':'gun-control',
    'immigration':'immigration',
    'international-politics-and-world-news':'politics',
    'islam':'islam',
    'marriage-equality':'activism',
    'middle-class':'socioeconomics',
    'sport':'sport',
    'student-debt':'socioeconomics',
    'taxes':'socioeconomics',
    'trump-presidency':'politics',
    'universal health care':'universal-health-care',
    'vaccine':'coronavirus-and-vaccines',
    'vaccines':'coronavirus-and-vaccines',
    'white-nationalism':'white-nationalism',
    }
        
    df = pd.read_excel('data/final_labels_SG2.xlsx')
    df = df[df['label_bias']!='No agreement']
    embeddings = np.load('data/sentences_embeddings.npy')
    df['embedding'] = embeddings.tolist()
    
    df = df.rename(columns={'topic':'topic_original'})
    df['topic'] = df['topic_original'].map(new_topics_map)
    df['outlet_bias'] = df['outlet'].map(media_bias_map)
    return df

@st.cache_resource
def generate_embedding_umap():
    '''
    Computes and displays a UMAP 3D map of the media bias dataset ADA embeddings.
    '''
    df = load_data()

    if 'umap_3d_x' not in df.columns:
        with st.spinner('Calculating UMAP dimensionality reduction'):
            embeddings_3d = UMAP(n_components=3).fit_transform(np.vstack(df['embedding']))
            df['umap_3d_x'] = embeddings_3d[:,0]
            df['umap_3d_y'] = embeddings_3d[:,1]
            df['umap_3d_z'] = embeddings_3d[:,2]

    fig_umap = px.scatter_3d(df, x='umap_3d_x', y='umap_3d_y', z='umap_3d_z', color='topic', 
                            width=1000, height=680, hover_data=["outlet", "label_bias"])
    fig_umap.update_traces(marker_size=1.5)
    fig_umap.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False, title=''),
            yaxis=dict(showticklabels=False, title=''),
            zaxis=dict(showticklabels=False, title=''),
        )
    )

    st.plotly_chart(fig_umap)

def fit_model_and_create_report(df, column, clf):
    X_train, X_test, y_train, y_test = train_test_split(
        list(df.embedding.values), df[column], test_size=0.2, random_state=42
    )

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    report = classification_report(y_test, preds)
    st.code(report)

def model_testing():

    df = load_data()
    selected_column = st.selectbox("Select a variable", options=['topic', 'label_bias', 'outlet_bias'])
    # Add a model selection drop-down
    model_options = {
        "Logistic Regression": LogisticRegression,
        "Random Forest": RandomForestClassifier,
        "K-Nearest Neighbors": KNeighborsClassifier,
        "Neural Network": MLPClassifier,
    }
    selected_model = st.selectbox("Select a model", options=list(model_options.keys()))

    # Set default model hyperparameters
    hyperparameters = {}

    # Add hyperparameter input fields based on the chosen model
    if selected_model == "Logistic Regression":
        C = st.slider("C (Regularization strength)", min_value=0.0, max_value=4.0, value=1.0, step=0.1)
        penalty = st.selectbox("Penalty (Regularization type)", options=["none", "l2"])
        hyperparameters = {"C": C, "penalty": penalty}

    elif selected_model == "Random Forest":
        n_estimators = st.number_input("n_estimators (Number of trees)", min_value=10, max_value=1000, value=100, step=10)
        max_depth = st.slider("max_depth (Maximum depth of the trees)", min_value=1, max_value=30, value=5, step=1)
        hyperparameters = {"n_estimators": n_estimators, "max_depth": max_depth}

    elif selected_model == "K-Nearest Neighbors":
        n_neighbors = st.number_input("n_neighbors (Number of neighbors)", min_value=1, max_value=50, value=5, step=1)
        weights = st.selectbox("weights (Weight function)", options=["uniform", "distance"])
        hyperparameters = {"n_neighbors": n_neighbors, "weights": weights}

    elif selected_model == "Neural Network":
        hidden_layer_sizes = st.number_input("Hidden layer size", min_value=10, max_value=1000, value=100, step=10)
        learning_rate = st.slider("learning_rate (Learning rate)", min_value=0.0001, max_value=1.0, value=0.001, step=0.0001)
        max_iter = st.number_input("max_iter (Maximum number of iterations)", min_value=100, max_value=1000, value=200, step=10)
        hyperparameters = {"hidden_layer_sizes": (hidden_layer_sizes,), "learning_rate_init": learning_rate, "max_iter": max_iter}

    # Create an instance of the chosen model with the specified hyperparameters
    clf = model_options[selected_model](**hyperparameters)

    # Pass the classifier instance to the fit_model_and_create_report function
    if st.button('Fit model'):
        with st.spinner('Fitting model to training data and generating report...'):
            fit_model_and_create_report(df, selected_column, clf)   

def analysis_tab():
    st.header("Embeddings")
    st.write('To achieve the embedding similarities in the training data set comparable to how ChatGPT operates, I will use the OpenAI API to get sentence embeddings with the ADA second generation model, which is [recommended by OpenAI](https://openai.com/blog/new-and-improved-embedding-model) for all usecases.')

    st.write("Here's a snippet of the code needed to set up the openAI API key and get the emebeddings for our dataset.")
    st.code('''
    import openai
    import tiktoken
    from openai.embeddings_utils import get_embedding

    # read the API KEY from an environment variable and set it in openai
    api_key = os.getenv('OPENAI_API_KEY') 
    openai.api_key = api_key

    # load sentences dataframe
    df = pd.read_csv('../data/sentences_data.csv')

    # set the embedding model parameters
    embedding_model = "text-embedding-ada-002"
    embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
    max_tokens = 8000
    encoding = tiktoken.get_encoding(embedding_encoding)

    # omit reviews that are too long to embed
    df["n_tokens"] = df.text.apply(lambda x: len(encoding.encode(x)))
    df = df[df.n_tokens <= max_tokens]

    # get embedding
    df["embedding"] = df.text.apply(lambda x: get_embedding(x, engine=embedding_model))
    ''', language="python")

    st.write("The ADA embeddings span a vector space of ~1500 dimensions. To visualize it, we can use a dimensionality reduction algorithm like t-SNE or UMAP:")

    generate_embedding_umap()

    st.write("From the UMAP dimensionality reduction plots above it's clear that the main driver between similarity of the embedded data is the topic of the sentence. So, a spatial-distance type of model like kNN will be most likely biased towards detecting topics more so than bias. So next, we test different classification models on the embedding data in different variables, mainly the topic and bias.")
    
    # st.header("Classification models")
    # st.write("Here we'll focus on model selection and hyperparameter tuning for classification of three variables: topic, bias and political bias (inferred from media outlet).")
    
    # st.subheader("Topic classification")
    # st.write("As seen in the UMAP above, sentences spanning similar topics form distinct clusters in the embedding data, which makes a distance-based classifier suitable for this variable. To verify this, let's look at the performance of a kNN classifier compared to a Random Forest Classifier (the default one in the OpenAI embedding classification [example](https://github.com/openai/openai-python/blob/main/examples/embeddings/Classification.ipynb).)")
    # df=load_data()
    # with st.spinner('Fitting RFC and kNC...'):
    #     col1, col2 = st.columns(2)
    #     with col1:
    #         st.write("**Random Forest Classifier**")
    #         fit_model_and_create_report(df, 'topic', RandomForestClassifier())
    #     with col2:
    #         st.write("**kNeighbors Classifier**")
    #         fit_model_and_create_report(df, 'topic', KNeighborsClassifier())
    # # st.table()
    # st.write("We can see that the kNN classifier (f1 weighted = 0.64) performs better than the Random Forest one for topic classification (f1 weighted = 0.49).")
    # st.write("Let's also look at the confusion matrix heatmeap:")
    # st.image('data/topics_heatmap.png', use_column_width=True)
    # st.write("The heatmap unveils which topics are more commonly confused with each other and offers additional options for feature engineering. For example, the vaccine topic is most commonly confused  with the coronavirus (logical) and elections-2020 (circumstantial?) topic. Middle-class and taxes are also commonly confused, as well as black lives matter and blm, trump-presidency and international-politics-and-world-news, gun control and gun-control.This points to the fact that simpler topic labels may be more beneficial for this project, as we're aiming to first identify a broader topic a sentence belongs to and then identify whether the language used to discuss it is biased or not. In other uses-cases, like news article tagging and database sorting for recommendation systems, it may be more useful to keep the original detailed labels and find ways to improve the classifier on them instead.")
    # st.code('''
    # new_topics_map = {
    # '#metoo':'activism',
    # 'abortion':'abortion',
    # 'black lives matter':'activism',
    # 'blm':'activism',
    # 'coronavirus':'coronavirus-and-vaccines',
    # 'elections-2020':'politics',
    # 'environment':'environment',
    # 'gender':'socioeconomics',
    # 'gun control':'gun-control',
    # 'gun-control':'gun-control',
    # 'immigration':'immigration',
    # 'international-politics-and-world-news':'politics',
    # 'islam':'islam',
    # 'marriage-equality':'activism',
    # 'middle-class':'socioeconomics',
    # 'sport':'sport',
    # 'student-debt':'socioeconomics',
    # 'taxes':'socioeconomics',
    # 'trump-presidency':'politics',
    # 'universal health care':'universal-health-care',
    # 'vaccine':'coronavirus-and-vaccines',
    # 'vaccines':'coronavirus-and-vaccines',
    # 'white-nationalism':'white-nationalism',
    # }

    # df = df.rename(columns={'topic':'topic_original'})
    # df['topic'] = df['topic_original'].map(new_topics_map)
    # ''')

    # st.write("With this simple relabeling we have managed to increase the f1 score of the topic classifier by 10%!")
    
    # st.subheader("Bias classification")
    # st.write("As mentioned in the Introduction, this data set has already been used for bias detection, performing best with a pre-trained BERT model. Here we focus on evaluating whether using the ADA embeddings directly can achieve a comparable performance. The best performing model identified for the dataset used in this app (SG2) has an F1 score of 0.804.")
    # st.image('data/f1_paper.png',  caption="F1 scores of all models tested in 'Neural Media Bias Detection Using Distant Supervision With BABE - Bias Annotations By Experts' by T.Spinde et al.", width=400)

    # st.write('To find the best classification model for the bias labels, we ran grid search with cross validation with the following parameters:')
    # st.code('''
    # # set the parameter grid
    # pipeline_steps = [
    #     {
    #         'clf': [LogisticRegression()],
    #         'clf__penalty': ['none', 'l2'],
    #         'clf__C': np.logspace(-4, 4, 5),
    #     }, 
    #     {
    #         'clf': [SVC()],
    #         'clf__kernel': ['linear', 'rbf'],
    #         'clf__C': [0.1, 1, 10],
    #         'clf__gamma': [1, 0.1, 0.01],
    #     }, 
    #     {
    #         'clf': [RandomForestClassifier()],
    #         'clf__n_estimators': [100, 200, 500],
    #         'clf__max_depth': [None, 10, 20, 30],
    #     }, 
    #     {
    #         'clf': [GradientBoostingClassifier()],
    #         'clf__n_estimators': [100, 200, 500],
    #         'clf__learning_rate': [0.1, 0.05, 0.01],
    #         'clf__max_depth': [3, 4, 5],
    #     },
    #     {
    #         'clf': [MLPClassifier()],
    #         'clf__hidden_layer_sizes': [(32, 32), (64, 64), (128, 128)],
    #         'clf__activation': ['relu', 'tanh'],
    #         'clf__learning_rate': ['constant', 'invscaling', 'adaptive'],
    #         'clf__max_iter': [200, 300, 400],
    #     }
    # ]

    # # build the pipeline
    # pipe = Pipeline([
    #     ('preprocessing', StandardScaler()), 
    #     ('clf', None)  # Will be filled in by GridSearchCV
    # ])

    # # set up grid search
    # grid_search = GridSearchCV(estimator=pipe, param_grid=pipeline_steps, scoring='f1_weighted', n_jobs=-1, cv=5, verbose=2)

    # # split the data into training and test
    # X_train, X_test, y_train, y_test = train_test_split(
    #     list(df.embedding.values), df.label_bias, test_size=0.2, random_state=42
    # )

    # # run grid search
    # grid_search.fit(X_train, y_train)
    # ''')

    # st.write("The best fitting model (f1 weighted = 0.79) for the bias labels resulting from the grid search (with a subsequent more granular search across the Logistic Regression hyperparameters) was a Logistic Regression model with penalty='l2' and C=0.0002. Of note here is that we haven't managed to improve on the published version with the headline-pre-trained embeddings, but we're not too far off, so using ADA embeddings seems like a good substitute when short on resources to pre-train your own embedding!")
    # st.image('data/f1_scores_bias.png', use_column_width=True)

    # st.subheader("Political bias classification")
    # st.write("To classify on political bias, we first need to match the outlets with their own media bias / fact check rating. Using data from [AllSlides](https://www.allsides.com/media-bias/media-bias-chart) and [Media Bias / Fact Check](https://mediabiasfactcheck.com/) we generate an outlet bias map and apply it to the data:")
    # st.code(
    #     '''
    #     media_bias_map = {
    #     'HuffPost': 'left',
    #     'Federalist': 'right',
    #     'Daily Beast': 'left',
    #     'Alternet': 'left',
    #     'Breitbart': 'right',
    #     'New Yorker': 'left',
    #     'American Greatness': 'right', # from https://mediabiasfactcheck.com/american-greatness/
    #     'Daily Caller': 'right',
    #     'Daily Wire': 'right',
    #     'Slate': 'left',
    #     'Reuters': 'center',
    #     'Hill': 'center', # from https://mediabiasfactcheck.com/the-hill/
    #     'USA Today': 'left',
    #     'CNBC': 'left',
    #     'Yahoo News - Latest News & Headlines': 'left',
    #     'AP': 'left',
    #     'Bloomberg': 'left',
    #     'Fox News': 'right',
    #     'MSNBC': 'left',
    #     'Daily Stormer': 'right', # from https://mediabiasfactcheck.com/the-hill/
    #     'New York Times': 'left'
    #     }

    #     df['outlet_bias'] = df['outlet'].map(media_bias_map)
    #     '''
    # )
    # st.write("The search for best model is similar to the one for bias labels: we run a grid search with CV over several models and hyperparameter values and find that the best classifier for this variable is MLPClassifier(hidden_layer_sizes=(128, 128), max_iter=300) with a weighted F1 score = 0.72")

    # st.divider()
    # st.write("***NOTE**: An approach using sequential classification in each variable was also tested: first topic, then separate bias classifiers for each topic, then political bias classifier only on biased data. This approach has systematically resulted in a loss of accuracy so we're not showcasing it here. More can be found in the accompanying exploration notebooks in the git repository.*")

    st.header("Model explorer")
    st.write("You can use the interactive explorer to train and test different models on the dataset. Runs a little slow!")
    model_testing()

    
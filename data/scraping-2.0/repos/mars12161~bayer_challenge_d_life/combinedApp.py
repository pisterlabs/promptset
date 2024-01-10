import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import pickle
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
from streamlit_extras.switch_page_button import switch_page
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import openai
from dotenv.main import load_dotenv

load_dotenv()

api_key = os.getenv("API_TOKEN") 

export_path = os.path.join(os.getcwd(), 'exports')

st.set_page_config(
	page_title="Breast Cancer Dataset",
	page_icon="👩‍⚕️",
	layout="wide",
	initial_sidebar_state="expanded")

st.title('Breast Cancer Dataset')

st.markdown(
	"""
	<style>
		[data-testid=stSidebar] [data-testid=stImage]{
			text-align: center;
			display: block;
			margin-left: auto;
			margin-right: auto;
			width: 100%;
		[data-testid="stSidebar"] {
			width: 200px; 
			}
	</style>
	""", unsafe_allow_html=True
)

with st.sidebar:
	image = Image.open('images/bc_awareness.png')
	st.image(image, width=100)
	selected = option_menu("Menu", ['Information', 'Exploratory Analysis', 'Machine Learning', 'Predictions', 'Ask the AI', 'Sources'])
	selected

cd_2018 = pd.read_csv('./data/cd_2018.csv') #data needed for map
df = pd.read_csv('./data/dataset_factorised.csv') #data needed for EDA
model = pickle.load(open("./model/lr2.pkl", "rb"))
scaler = pickle.load(open("./model/trained_scaler.pkl", "rb"))
X_lr = pd.read_csv('./data/X_lr.csv',index_col= None)

y = df['diagnosis']
Malignant=df[df['diagnosis'] == 0]
Benign=df[df['diagnosis'] == 1]

def histplot(features):
	plt.figure(figsize=(10,15))
	for i, feature in enumerate(features):
		bins = 20
		plt.subplot(5, 2, i+1)
		sns.histplot(Malignant[feature], bins=bins, color='blue', alpha=0.6, label='Malignant');
		sns.histplot(Benign[feature], bins=bins, color='pink', alpha=0.5, label='Benign');
		plt.title(str(' Density Plot of: ')+str(feature))
		plt.xlabel(str(feature))
		plt.ylabel('Count')
		plt.legend(loc='upper right')
	plt.tight_layout()
	plt.show()
st.set_option('deprecation.showPyplotGlobalUse', False)

def plot_heatmap(confusion):
	
	plt.figure(figsize=(4,3))
	sns.heatmap(confusion, xticklabels = np.unique(y), yticklabels = np.unique(y),
				cmap = 'RdPu', annot=True, fmt='g')
	plt.xlabel('Predicted', fontsize=14)
	plt.ylabel('Actual', fontsize = 14)

def information_tab():
	st.subheader('Information')
	st.markdown("In 2018, an estimated 2.1 million individuals were confronted with \
		a breast cancer diagnosis across the globe. Regrettably, breast cancer stands as\
		a formidable contributor to female mortality rates. Particularly in developing nations, \
		the paucity of healthcare resources often impedes the prompt identification of this \
		disease.\n\n Though breast cancer incidence rates remain relatively subdued in less developed regions, \
		their mortality rates mirror those of more developed areas. This disconcerting finding suggests \
		a distressing probability: a substantial number of cases might be escaping diagnosis entirely. \
		This supports the urgency for improved detection methods.\n\nThe objective of our initiative \
		is to enhance the screening of entire populations, thereby mitigating medical expenses, while \
		leveraging computer-aided diagnosis. Additionally, the correlation between early detection and \
		increased chances of survival amplifies the significance of this endeavour.")
	image1 = Image.open('images/figure2.png')
	st.image(image1)
	st.write("Source: https://canceratlas.cancer.org")
	st.write("---")
	st.subheader('Breast Cancer Death Rate (per 100,000 Individuals) in 2018')
	st.write("Included in the hover data below is the rate of diagnosed cases and the rate of breast cancer deaths per 100,000 people, for both sexes and age-standardized.")
	fig = px.choropleth(cd_2018,
					 locations = "code", 
					 color = "death_rate", 
					 hover_name = "country", 
					 hover_data = ["diagnosed_rate"],
					 color_continuous_scale = px.colors.sequential.Sunsetdark)
	st.plotly_chart(fig)
	
def exploratory_analysis_tab():
	st.subheader('Exploratory Analysis')
	#divide feature names into groups
	mean_features= ['radius_mean','texture_mean','perimeter_mean',\
				'area_mean','smoothness_mean','compactness_mean',\
				'concavity_mean','concave_points_mean','symmetry_mean',\
				'fractal_dimension_mean']
	error_features=['radius_se','texture_se','perimeter_se',\
				'area_se','smoothness_se','compactness_se',\
				'concavity_se','concave_points_se','symmetry_se',\
				'fractal_dimension_se']
	worst_features=['radius_worst','texture_worst','perimeter_worst',\
				'area_worst','smoothness_worst','compactness_worst',\
				'concavity_worst','concave_points_worst',\
				'symmetry_worst','fractal_dimension_worst']
	option = st.selectbox(
		'What would you like to see?',
		('Density Graphs', 'Correlation or Heatmap'))

	if 'Density Graphs' in option: 
		option_1 = st.selectbox('Please select a group:', ('Mean Features', 'Standard Error Features', 'Worst Features'))
		if 'Mean Features' in option_1: 
			st.write(df[mean_features].describe())
			mf = histplot(mean_features)
			st.pyplot(mf)
		if 'Standard Error Features' in option_1: 
			st.write(df[error_features].describe())
			ef = histplot(error_features)
			st.pyplot(ef)
		if 'Worst Features' in option_1: 
			st.write(df[worst_features].describe())
			wf = histplot(worst_features)
			st.pyplot(wf)
	if 'Correlation or Heatmap' in option: 
		df_corr = df.drop(columns = ['id'])
		fig, ax = plt.subplots()
		option_2 = st.selectbox('Please select a group:', ('All', 'Mean Features', 'Standard Error Features', 'Worst Features'))
		if 'All' in option_2:
			sns.heatmap(df_corr.corr(), ax=ax)
			st.write(fig)
		if 'Mean Features' in option_2: 
			sns.heatmap(df_corr[mean_features].corr(), ax=ax)
			st.write(fig)
		if 'Standard Error Features' in option_2: 
			sns.heatmap(df_corr[error_features].corr(), ax=ax)
			st.write(fig)
		if 'Worst Features' in option_2: 
			sns.heatmap(df_corr[worst_features].corr(), ax=ax)
			st.write(fig)
def machine_learning_tab():
	st.subheader('Machine Learning')
	st.write("All machine learning models were trained using an 80/20 split on stratified data that was standardised using StandardScaler.")
# link to dashboard here
	st.subheader("Model Explainer Dashboard Using SHAP")
	st.markdown("A **hub of interactive dashboards** for analyzing and explaining the predictions.")
	st.components.v1.iframe("https://final-2znz-main-afectjcvzq-wm.a.run.app/", width=1300, height=700, scrolling=True)
		
def sources_tab():
	st.subheader('Dataset')
	st.markdown("http://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic")
	st.subheader('Sources')
	st.markdown("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8626596/,  \n\
		 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7950292/,\n\
		 https://canceratlas.cancer.org/,  \nhttps://ourworldindata.org/cancer  \n")

def add_info():
	st.markdown("<h3 style='text-align: center; color: orchid;'>Cell Nuclei Measurements </h3>", unsafe_allow_html=True)
	st.markdown("<font size='2'>You can also update the measurements by hand using the sliders in the sidebar. </font>", unsafe_allow_html=True)
	slider_labels = [
		("Concavity (mean)", "concavity_mean"),
		("Concave points (mean)", "concave_points_mean"),
		("Radius (se)", "radius_se"),
		("Perimeter (se)", "perimeter_se"),
			("Area (se)", "area_se"),
			("Radius (worst)", "radius_worst"),
			("Texture (worst)", "texture_worst"),
			("Perimeter (worst)", "perimeter_worst"),
			("Area (worst)", "area_worst"),
			("Concavity (worst)", "concavity_worst"),
			("Concave points (worst)", "concave_points_worst"),
			("Symmetry (worst)", "symmetry_worst"),
		]
	input_dict = {}

	for label, key in slider_labels:
		input_dict[key] = st.slider(label, min_value = float(0), max_value = float(X_lr[key].max()), 
			value = float(X_lr[key].mean())
		)
	return input_dict

def get_scaled_values(input_dict):
	scaled_dict = {}

	for key, value in input_dict.items():
		max_val = X_lr[key].max()
		min_val = X_lr[key].min()
		scaled_value = (value - min_val) / (max_val - min_val)
		scaled_dict[key] = scaled_value
	return scaled_dict

def get_radar_chart(input_data):
	input_data = get_scaled_values(input_data)

	categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
			'Concavity', 'Concave Points', 'Symmetry'
			]

	fig = go.Figure()

	fig.add_trace(go.Scatterpolar(
		r=[
		0,0,0,0,input_data['concavity_mean'], input_data['concave_points_mean'],0
		],
		theta=categories,
		fill='toself',
		name='Mean Value'
	))
	fig.add_trace(go.Scatterpolar(
		r=[
		input_data['radius_se'],0, input_data['perimeter_se'], input_data['area_se'], 0,0
		],
		theta=categories,
		fill='toself',
		name='Standard Error'
	))
	fig.add_trace(go.Scatterpolar(
		r=[
		input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
		input_data['area_worst'], input_data['concavity_worst'], input_data['concave_points_worst'], 
		input_data['symmetry_worst']
		],
		theta=categories,
		fill='toself',
		name='Worst Value'
	))
	fig.update_layout(
		polar = dict(radialaxis = dict(visible = True,range = [0, 1])),
		showlegend = True
		)
	return fig

def add_predictions(input_data): 
	input_array = np.array(list(input_data.values())).reshape(1, -1)
	input_array_scaled = scaler.transform(input_array)
	prediction = model.predict(input_array_scaled)
	st.subheader("**The cell cluster is:**")
	if prediction[0] == 0:
		st.write("<span class='diagnosis benign'>:blue[**Benign**]</span>", unsafe_allow_html=True)
	else:
		st.write("<span class='diagnosis malignant'>:blue[**Malignant**]</span>", unsafe_allow_html=True)
	st.write(f"Probability of being benign: {model.predict_proba(input_array_scaled)[0][0]: .3f}")
	st.write(f"Probability of being malignant: {model.predict_proba(input_array_scaled)[0][1]: .3f}")
	return (model.predict_proba(input_array_scaled)[0][0], model.predict_proba(input_array_scaled)[0][1])

def assistant(B, M):
	openai.api_key = api_key
	prompt = (
		"I build an app with Wisconsin breast cancer diagnosis and used machine learning to give you these results, "
		"now act as the role of assistant within that app and generate general guidelines and tell them what should they do act as you are talking to the patients directly"
		f"Prediction Results:\nMalignant Probability: {M}\nBenign Probability: {B}"
		)
	response = openai.Completion.create(
		model="text-davinci-003",
		prompt=prompt,
		temperature=0.6,
		max_tokens = 400
		)
	guidelines = response.choices[0].text.strip()
	return(guidelines)

def predictions_tab():
	st.subheader('Predictions')
	with st.container():
		st.write("Please connect this app to your cytology lab to help diagnose breast cancer form your tissue sample. This app uses a logistic regression machine learning model to predict whether a breast mass is benign or malignant based on the measurements provided from your cytosis lab. ")
		st.text("")
		st.markdown('**This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis**')
	st.write("---")
	col1, col2 = st.columns([1, 3])
	with col1:
		with st.form("Prediction Form"):
			input_data = add_info()
			submitted = st.form_submit_button("Submit")
	with col2:
		if submitted:
			st.markdown("<h2 style='text-align: center; color: orchid;'>Cell Cluster Prediction </h2>", unsafe_allow_html=True)
			radar_chart = get_radar_chart(input_data)
			st.plotly_chart(radar_chart)
			B, M = add_predictions(input_data)
			st.write("---")
			# if st.button('Receive tips from AI!'):
			st.write(assistant(B, M))

			st.write("---")

def find_exported_files(path):
	for root, dirs, files in os.walk(path):
		for file in files:
			if file.endswith(".png") and file != 'figure2.png' and file != 'bc_awareness.png':
				return os.path.join(root, file)
	return None

def ask_pandas():
	llm = OpenAI(api_key)
	pandasai = PandasAI(llm, save_charts=True, save_charts_path=export_path, verbose=True)
	st.markdown("<h2 style='color: DarkOrchid;'>Ask the AI </h2>", unsafe_allow_html=True)
	st.write("Here you can ask the AI a question about the data. The AI currently running in the background is OpenAI's GPT.")
	st.write("Other Large Language Models are available, such as HuggingFace's Falcon.")
	st.markdown('**Example questions:**')
	st.write("- What is the average radius of the cell clusters?")
	st.write("- What is the standard error of the mean of the cell clusters?")
	st.write("- Plot the mean radius of the cell clusters by diagnosis.")
	st.write("- Plot the distribution of the diagnosis in a pie chart.")
	st.markdown('**Note:** The AI is still learning, so it may not be able to answer all questions correctly.')

	with st.form("Question"):
		question = st.text_input("Question", value="", type="default")
		submitted = st.form_submit_button("Submit")
		if submitted:
			with st.spinner("PandasAI is thinking..."):
				answer = pandasai.run(df, prompt=question)
				st.write(answer)

				# Plotting
				chart_file = find_exported_files(export_path)
				if chart_file:
					st.image(chart_file)
					os.remove(chart_file)

def main():
	if 'Information' in selected:
		information_tab()
	if 'Exploratory Analysis' in selected:
		exploratory_analysis_tab()
	if 'Machine Learning' in selected:
		machine_learning_tab()
	if 'Sources' in selected:
		sources_tab()
	if 'Predictions' in selected:
		predictions_tab()
	if 'AI' in selected:
		ask_pandas()
	
if __name__ == '__main__':
	main()
import streamlit as st
# from forest_cover.predict import predict
# from forest_cover.area import area_dif
import cv2
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from ultralytics import YOLO
# from ultralytics import utails
import cv2

model_path = r"C:\Users\Lakshmi\Downloads\last.pt"

def predict(img):
    img = cv2.imread(img)
    H, W, _ = img.shape
    model = YOLO(model_path)
    results = model(img)
    for result in results:
        for j, mask in enumerate(result.masks.data):
            mask = mask.numpy() * 255
            mask = cv2.resize(mask, (W, H))
            cv2.imwrite('./pictures/output.jpg', mask)
    out_img = cv2.imread('./output.jpg')
    return out_img

# def area_dif(img):
def area_dif(img):
    pixel_area = 1.0  # square meter
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    new_count = cv2.countNonZero(img)
    new_area = new_count * pixel_area
    print(new_area)
    prev_img = cv2.imread(r"forest_cover/pictures/previous.png", cv2.IMREAD_GRAYSCALE)
    prev_count = cv2.countNonZero(prev_img)
    prev_area = prev_count * pixel_area
    print(prev_area)
    return (prev_area - new_area)/prev_area * 100


st.set_page_config(
    page_title="Green Check",
    page_icon="🌿",
)

st.markdown("<h1 style='color: green; font-style: italic; font-family: Comic Sans MS; font-size:5rem' >Green Check 🌿</h1> <h3 style='color: #004B49; font-style: italic; font-family: Comic Sans MS; font-size:2rem'>Tracking Forest Cover Changes via Advanced Computer Vision. AI tracks forest changes via images from satellites/drones. Detects cover increase/decrease, aids conservation efforts. </h3>", unsafe_allow_html=True)

pic = st.file_uploader("Upload the forest aerial view here")

info = None
if pic is not None:
    st.image(pic)
    st.write("image processing started")
    img = predict(pic)
    st.image(img)
    st.write("Predicted forest area")
    st.image(img)
    st.write("Previous forest area")
    prev_img = cv2.imread(r"D:\llm projects\Forest-Amazon\forest_cover\pictures\previous.png")
    st.image(prev_img)
    # ###########
    # area_dif = area_dif(img)
    area_dif = area_dif(img)
    st.write("Percentage decrease in the forest area is")
    st.write(area_dif)
    st.write("Data Analysis of the Amazon degradation dataset ")
    def_amazon_data = pd.read_csv(r"C:\Users\Lakshmi\Downloads\archive (1)\def_area_2004_2019.csv", parse_dates=True, encoding = "cp1252")
    def_amazon_data.rename({
    'ï»¿Ano/Estados':'Year',
    'AC':'Acre',
    'AM': 'Amazonas',
    'AP': 'Amapa',
    'MA':'Maranhao',
    'MT':'Mato Grosso',
    'PA':'Para',
    'RO':'Rondonia',
    'RR':'Roraima',
    'TO':'Tocantins',
    'AMZ LEGAL':'Total'
    }, axis=1, inplace=True)

    fig = px.bar(def_amazon_data, x="Year", y="Total")
    fig.update_layout(title_text='Total Deforested Area per Year')
    fig.update_xaxes(tickmode='linear')
    st.plotly_chart(fig)

    states=["Acre","Amazonas","Amapa","Maranhao","Mato Grosso","Para","Rondonia","Roraima","Tocantins"]
    data=[]
    for i in range(len(states)):
        data.append(def_amazon_data[states[i]].sum())
    colors = ['lightslategray',] * 9
    colors[5] = 'crimson'
    fig = go.Figure(data=[go.Bar(x=states, y=data, text=data, textposition='auto', marker_color=colors)])
    fig.update_layout(title_text='Total Deforested Area by State')
    st.plotly_chart(fig)
    deforestation = "Predicted deforestation for year 2024 in  'Acre','Amazonas','Amapa','Maranhao','Mato', 'Grosso','Para','Rondonia','Roraima','Tocantins'  respectively is  [[ 249.11  617.25   79.45 1034.42 2705.46 4773.25 1087.58  462.55   95.79]]"
    st.markdown("<p style='color: #004B49; font-style: italic; font-family: Comic Sans MS; ' >Result of ML model prediction for 2024 regading amazon degradation</p>", unsafe_allow_html=True)
    st.write(deforestation)
    info = "the percent drop in forest cover is "+str(area_dif) + " and "+ deforestation



# Know the consequences
deforestation_model = joblib.load(r"D:\llm projects\Forest-Amazon\models\forest_model.pkl")
deforestation = deforestation_model.predict(2023)


# 2 - llm
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


consequence_template = PromptTemplate(
        input_variables=['info'],
        template='in the image of a place in amazon forest the new and previous compared  {info}. consider the given info and let me know the consequences on the biodiversity loss, harm to climate, and all other consequences with emoji and in points'
    )

llm = OpenAI(temperature=0.9)

consequence_chain = LLMChain(llm=llm, prompt=consequence_template, output_key ='consequence')

if info is not None:
    consequence = consequence_chain.run(info)
    st.write(consequence)










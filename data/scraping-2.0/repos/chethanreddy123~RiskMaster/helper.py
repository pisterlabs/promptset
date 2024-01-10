import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder 
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import io
import fitz  
import pytesseract
from PIL import Image
from fastapi.responses import FileResponse



def structure_json(json_data):
    structured_data = {
        "input": json_data.get("input"),
        "output": json_data.get("output"),
        "intermediate_steps": []
    }

    for i in json_data.get("intermediate_steps"):
        structured_data['intermediate_steps'].append(i[0].log)

    return structured_data


def extract_text_from_first_page(pdf_file):
    # Open the PDF
    doc = fitz.open(stream=pdf_file, filetype="pdf")
    page = doc[0]  # Extract text from the first page

    # If the PDF contains text directly
    if page.get_text():
        return page.get_text()

    # If the PDF contains images, use OCR
    text = ""
    for img in page.get_images(full=True):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image = Image.open(io.BytesIO(image_bytes))
        text += pytesseract.image_to_string(image)

    return text

def predict_with_models(sample_input):

    model = joblib.load('XGBClassifier_model.joblib')
    
    data = pd.DataFrame([sample_input])
    print(sample_input)
    
    le = LabelEncoder()
    obj_col = ['HasCoSigner','LoanPurpose','HasDependents', 'HasMortgage','MaritalStatus', 'EmploymentType', 'Education']
    for col in obj_col:
        data[col] = le.fit_transform(data[col])

    data = data.drop(['LoanID'], axis=1)
    
    trained_models = joblib.load('XGBClassifier_model.joblib')
    y_pred = model.predict(data)
    return y_pred


def handle_query(agent, query):
    if "graph" in query.lower() or "plot" in query.lower():
        # Handle graph-related query
        # Assuming 'agent' can process the query and generate a graph
        graph_result = agent(query + " Note: save the plot as plot.png")
        refined_result = structure_json(graph_result)
        refined_result['intermediate_steps'] = str(refined_result['intermediate_steps'])
        return FileResponse("plot.png", headers=refined_result)
    else:
        return structure_json(agent(query))
    
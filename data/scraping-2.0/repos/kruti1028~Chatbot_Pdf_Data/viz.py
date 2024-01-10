import os
import sqlite3
import fitz  # PyMuPDF
import openai
import plotly.graph_objects as go
from flask import Flask, request, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename
import re  # For regular expressions
from typing import List, Dict, Optional

app = Flask(__name__)
UPLOAD_FOLDER = 
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = 'your_secret_key'
openai.api_key = 

# Insurance policy information class
class InsurancePolicyInfo:
    def __init__(self):
        self.total_coverage_amount: Optional[str] = None
        self.insurance_types: List[str] = []
        self.annual_premium: Optional[str] = None
        self.insurer: Optional[str] = None
        self.insured: List[str] = []
        self.issue_date: Optional[str] = None
        self.renewal_date: Optional[str] = None
        self.policy_number: Optional[str] = None
        self.categories: Dict[str, Dict[str, List[str]]] = {}

# Database setup
def setup_database():
    conn = sqlite3.connect('insurance.db')
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS insurance_policies")
    cursor.execute("DROP TABLE IF EXISTS categories")
    cursor.execute('''CREATE TABLE insurance_policies (
                        id INTEGER PRIMARY KEY,
                        total_coverage_amount TEXT,
                        type_of_insurance TEXT,  
                        annual_premium TEXT,
                        insurer TEXT,
                        insured TEXT,
                        issue_date TEXT,
                        renewal_date TEXT,
                        policy_number TEXT
                      )''')
    cursor.execute('''CREATE TABLE categories (
                        policy_id INTEGER,
                        category_name TEXT,
                        covered TEXT,
                        not_covered TEXT,
                        events_covered TEXT,
                        FOREIGN KEY(policy_id) REFERENCES insurance_policies(id)
                      )''')
    conn.commit()
    conn.close()



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        text = extract_text_from_pdf(filepath)
        policy_info = query_openai_and_parse(text)
        insert_data_into_database(policy_info)
        chart_html = generate_sunburst_chart(policy_info)
        return render_template('screen2.html', chart_html=chart_html)
    else:
        flash('Invalid file type')
        return redirect(request.url)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def query_openai_and_parse(text):
    truncated_text = text[:3000]  # Truncate to fit within token limits

    prompt = (
        "Please provide a structured summary of this insurance policy in a list format, focusing on specific details. "
        "Include the following information: \n"
        "Total coverage amount\n"
        "Type of insurance\n"
        "Categories of insurance covered\n"
        "Details of what is covered under each category\n"
        "Details of what is not covered under each category\n"
        "Covered events\n"
        "Annual premium\n"
        "Name of the insurer\n"
        "Name of the insured\n"
        "Issue date\n"
        "Renewal date\n"
        "Policy number\n" + truncated_text
    )

    response = openai.Completion.create(
        model="text-davinci-003", 
        prompt=prompt,
        max_tokens=1000  # Limit the completion to 1000 tokens
    )
    extracted_text = response.choices[0].text
    print(extracted_text)
    return parse_openai_response(extracted_text)


def parse_openai_response(extracted_text):
    policy_info = InsurancePolicyInfo()

    # Basic information extraction
    policy_info.total_coverage_amount = extract_value_by_keyword(extracted_text, "Total Coverage Amount")
    policy_info.annual_premium = extract_value_by_keyword(extracted_text, "Annual Premium")
    policy_info.insurer = extract_value_by_keyword(extracted_text, "Name of the Insurer")
    insured_str = extract_value_by_keyword(extracted_text, "Name of the Insured")
    policy_info.insured = [name.strip() for name in re.split(r' & | and ', insured_str)] if insured_str else []
    policy_info.issue_date = extract_value_by_keyword(extracted_text, "Issue Date")
    policy_info.renewal_date = extract_value_by_keyword(extracted_text, "Renewal Date")
    policy_info.policy_number = extract_value_by_keyword(extracted_text, "Policy Number")

    # Extracting types of insurance and categories
    insurance_types_str = extract_value_by_keyword(extracted_text, "Type of Insurance")
    policy_info.insurance_types = [insurance_type.strip() for insurance_type in insurance_types_str.split(',')] if insurance_types_str else []

    policy_info.categories = parse_categories(extracted_text)

    print("Parsed Data:", policy_info.__dict__)
    return policy_info

def parse_categories(text):
    categories = {}
    
    # Assuming categories are listed under "Categories of Insurance Covered"
    categories_str = extract_value_by_keyword(text, "Categories of Insurance Covered")
    if categories_str:
        category_names = [cat.strip() for cat in categories_str.split(',')]
        for category in category_names:
            categories[category] = {"covered": [], "not_covered": [], "events_covered": []}

            # Extract details for each category
            covered_str = extract_value_by_keyword(text, f"Details of What is Covered under {category}")
            if covered_str:
                categories[category]["covered"] = [item.strip() for item in covered_str.split(',')]

            not_covered_str = extract_value_by_keyword(text, f"Details of What is Not Covered under {category}")
            if not_covered_str:
                categories[category]["not_covered"] = [item.strip() for item in not_covered_str.split(',')]

    # Additional parsing logic for other details as required

    return categories

def extract_value_by_keyword(text, keyword):
    pattern = rf"{keyword}: ([^\n]+)"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return None

def insert_data_into_database(policy_info):
    conn = sqlite3.connect('insurance.db')
    cursor = conn.cursor()

    # Convert the 'insured' list to a string
    insured_str = ', '.join(policy_info.insured) if policy_info.insured else ''
    type_of_insurance_str = ', '.join(policy_info.insurance_types) if policy_info.insurance_types else ''

    # Insert main policy data
    cursor.execute('''INSERT INTO insurance_policies 
                      (total_coverage_amount, type_of_insurance, annual_premium, insurer, insured, issue_date, renewal_date, policy_number)
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', 
                   (policy_info.total_coverage_amount, type_of_insurance_str, policy_info.annual_premium, 
                    policy_info.insurer, insured_str, policy_info.issue_date, policy_info.renewal_date, policy_info.policy_number))
    
    # Get the last inserted row id (the policy_id)
    policy_id = cursor.lastrowid

    # Insert category data
    for category, details in policy_info.categories.items():
        covered_str = ', '.join(details['covered']) if details['covered'] else ''
        not_covered_str = ', '.join(details['not_covered']) if details['not_covered'] else ''
        cursor.execute('''INSERT INTO categories 
                          (policy_id, category_name, covered, not_covered)
                          VALUES (?, ?, ?, ?)''', 
                       (policy_id, category, covered_str, not_covered_str))
    cursor.execute("SELECT * FROM insurance_policies")
    print("Insurance Policies:", cursor.fetchall())
    cursor.execute("SELECT * FROM categories")
    print("Categories:", cursor.fetchall())
    conn.commit()
    conn.close()

def generate_sunburst_chart(policy_info):
    # Initialize the sunburst chart elements
    total_coverage_amount = policy_info.total_coverage_amount or "Not specified"
    root_label = f"Total Coverage: {total_coverage_amount}"
    labels = [root_label]  # root node with total coverage amount
    parents = [""]  # root has no parent
    values = [100]  # Assign a static value or calculate based on data
    # Detailed hover information for the center
    hovertext = [f"Insurer: {policy_info.insurer or 'Not specified'}<br>"
                 f"Insured: {', '.join(policy_info.insured) or 'Not specified'}<br>"
                 f"Total Coverage Amount: {total_coverage_amount}<br>"
                 f"Annual Premium: {policy_info.annual_premium or 'Not specified'}<br>"
                 f"Renewal Date: {policy_info.renewal_date or 'Not specified'}<br>"
                 f"Issue Date: {policy_info.issue_date or 'Not specified'}"]

    # Add insurance types as the first layer
    for insurance_type in policy_info.insurance_types:
        labels.append(insurance_type)
        parents.append(root_label)
        values.append(20)  # Adjust value as necessary
        hovertext.append(f"Type of Insurance: {insurance_type}")

        # Add categories for each insurance type as the second layer
        for category, details in policy_info.categories.get(insurance_type, {}).items():
            labels.append(category)
            parents.append(insurance_type)
            values.append(10)  # Adjust value as necessary
            hovertext.append(f"Category: {category}")

            # Add covered and not covered items as the third and fourth layers
            # Handling covered items
            for item in details.get('covered', []):
                labels.append(f"Covered: {item}")
                parents.append(category)
                values.append(5)  # Adjust value as necessary
                hovertext.append(f"Covered Item: {item}")

            # Handling not covered items
            for item in details.get('not_covered', []):
                labels.append(f"Not Covered: {item}")
                parents.append(category)
                values.append(5)  # Adjust value as necessary
                hovertext.append(f"Not Covered Item: {item}")

    # Create and configure the sunburst chart
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        hoverinfo="text",
        hovertext=hovertext,
        branchvalues="total"
    ))
    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
    print("Labels:", labels)
    print("Parents:", parents)
    print("Values:", values)
    print("Hovertext:", hovertext)

    # Convert chart to HTML
    chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn', default_height=600, default_width=800)
    return chart_html

if __name__ == '__main__':
    setup_database()
    app.run(debug=True, port=8080)


import os
import openai
import rdflib
import pyshacl


system_prompt = """
You will act as a skilled expert automaton that is proficient in transforming unstructured text, specifically given German \"Regest\" (abstract) of the Regesta Imperii (RI), into Turtle RDF. Analyze the provided text based on the mapping rules I have shared and then execute the transformation to produce Turtle RDF based on the RI-Ontology, ensuring you adhere to the guidelines and only annotate if certain.

**Mapping rules**:

1. **Classes and Subclassing**:
   - `ri:Document` is a subclass of `crm:E31_Document`.
   - `ri:Complaint`, `ri:LettersPatent`, and `ri:Privilege` are subclasses of `ri:Document`, with `ri:Privilege` also being a subclass of `crm:E33_Linguistic_Object`.
   - `ri:Location` is a subclass of `crm:E53_Place`, with specific locations like `ri:City`, `ri:Village`, and `ri:Region` inheriting from it.
   - `ri:Person` is a subclass of `crm:E21_Person`, with specializations like `ri:Emperor`, `ri:Counselor`, etc., inheriting properties and restrictions.

2. **Property Specificity and Hierarchy**:
   - Various properties such as `ri:hasIssued`, `ri:concernsRights`, `ri:hasJurisdictionOver`, `ri:hasGrantedPrivilege`, `ri:advises`, `ri:leads`, `ri:confirmsPrivilege`, `ri:belongs`, `ri:isLocatedIn`, `ri:isPartOf`, and `ri:hasTitle` are defined with specific domain and range relations, aligning with the CRM patterns.
   - Introduce new properties like `ri:detailsOfComplaint` and `ri:decisionDetails` for capturing specific content of the document.

3. **Entity Mappings and ABox Interpretations**:
   - Individual entities (like `ex:FriedrichIII`, `ex:MargraveAlbrechtBrandenburg`) are mapped to their respective classes, with their titles and document relationships.
   - Locations (like `ex:Nuremberg`, `ex:Franconia`) are defined hierarchically and related through `ri:belongs` and `ri:isPartOf`.
   - Add annotations for specific complaints and decisions where appropriate.

4. **Document and Rights Contextualization**:
   - Documents and rights are contextualized with comments indicating their historical and legal relevance, including specific complaints and decisions.

5. **Refinement and Disambiguation**:
   - The properties are aligned with CRM patterns for semantic consistency, and ambiguities are minimized through explicit domain and range associations.

6. **Guidelines**:
   - Follow mapping rules strictly.
   - Preserve the original text.
   - Produce well-formed Turtle RDF.
   - Return only Turtle RDF. No ``` 
   - Annotate only when appropriate.
   - Preserve the complexity of output.
   - Compact Turtle RDF without any whitespace or indentation.
   - Use the following namespaces:
     - @prefix ri: <http://www.example.org/ontology/ri#> .
     - @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
     - @prefix crm: <http://www.cidoc-crm.org/cidoc-crm/> .
     - @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
   - Add rdfs:label to entities for better readability.
   - No language tags.

Turtle RDF Example:
´´´
@prefix ri: <http://www.example.org/ontology/ri#> .
@prefix ri-data: <http://www.example.org/data/ri#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix crm: <http://www.cidoc-crm.org/cidoc-crm/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

<ri-data:FriedrichIII-RI_XIII_H_28_n_41> a ri:Document ;
    rdfs:label "Friedrich III. - [RI XIII] H. 28 n. 41" ;
    ri:hasIssued <ri-data:Complaint_FriedrichIII-RI_XIII_H_28_n_41> ;
    crm:P4_has_time-span "1456-07-28"^^xsd:date ;
    ri:isLocatedIn <ri-data:Wiener_Neustadt> .

<ri-data:Complaint_FriedrichIII-RI_XIII_H_28_n_41> a ri:Complaint ;
    ri:concernsRights <ri-data:MargraveAlbrechtBrandenburg>,
                      <ri-data:MargraveFriedrichII>,
                      <ri-data:MargraveJohann>,
                      <ri-data:MargraveFriedrichJr> ;
    ri:detailsOfComplaint "Complaint about infringement of rights at Nuremberg land court" ;
    ri:decisionDetails "Decision by Friedrich III clarifying the impact of privileges on rights at Nuremberg land court" ;
    ri:isLocatedIn <ri-data:Nuremberg> ;
    crm:P70_documents <ri-data:FriedrichIII> .

<ri-data:MargraveAlbrechtBrandenburg> a ri:Person ;
    rdfs:label "Albrecht von Brandenburg" ;
    ri:hasTitle "Mgf." .

<ri-data:MargraveFriedrichII> a ri:Person ;
    rdfs:label "Friedrich II." ;
    ri:hasTitle "Mgf." .

<ri-data:MargraveJohann> a ri:Person ;
    rdfs:label "Johann" ;
    ri:hasTitle "Mgf." .

<ri-data:MargraveFriedrichJr> a ri:Person ;
    rdfs:label "Friedrich (d. J.)" ;
    ri:hasTitle "Mgf." .

<ri-data:Nuremberg> a ri:City ;
    rdfs:label "Nürnberg" ;
    ri:isPartOf <ri-data:Franconia> .

<ri-data:Franconia> a ri:Region ;
    rdfs:label "Franken" .

<ri-data:Wiener_Neustadt> a ri:City ;
    rdfs:label "Wiener Neustadt" .
´´´

Take a deep breath and lets think step by step. This is very important to my career.
"""
shacl_shapes = """
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ri: <http://www.example.org/ontology/ri#> .
@prefix crm: <http://www.cidoc-crm.org/cidoc-crm/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

# SHACL shapes for ri:Document and its subclasses
ri:DocumentShape
    a sh:NodeShape ;
    sh:targetClass ri:Document ;
    sh:property [
        sh:path ri:hasIssued ;
        sh:or ( [ sh:class ri:Document ] [ sh:class ri:Complaint ] [ sh:class ri:Privilege ] ) ; # Allow linking to Document or its subclasses
        sh:minCount 1 ;
    ] ;
    sh:property [
        sh:path crm:P4_has_time-span ;
        sh:datatype xsd:date ;
        sh:minCount 1 ;        # Making it mandatory
    ] ;
    sh:property [
        sh:path ri:isLocatedIn ;
        sh:or ( [ sh:class ri:City ] [ sh:class ri:Location ] [ sh:class ri:Region ] [ sh:class ri:Forest ]) ; # Including ri:Forest
        sh:minCount 1 ;        # Making it mandatory if location is always specified
    ] .

# Additional properties for ri:Complaint and ri:Privilege
ri:ComplaintShape
    a sh:NodeShape ;
    sh:targetClass ri:Complaint ;
    sh:property [
        sh:path ri:detailsOfComplaint ;
        sh:datatype rdfs:Literal ;
        sh:minCount 1 ;
    ] ;
    sh:property [
        sh:path ri:decisionDetails ;
        sh:datatype rdfs:Literal ;
        sh:minCount 1 ;
    ] .

ri:PrivilegeShape
    a sh:NodeShape ;
    sh:targetClass ri:Privilege ;
    sh:property [
        sh:path ri:detailsOfPrivilege ;
        sh:datatype rdfs:Literal ;
        sh:minCount 1 ;
    ] .

# SHACL shape for ri:Person
ri:PersonShape
    a sh:NodeShape ;
    sh:targetClass ri:Person ;
    sh:property [
        sh:path ri:hasTitle ;
        sh:datatype rdfs:Literal ;
        sh:minCount 1 ;  # Making it mandatory if every person has a title
    ] .
"""

# Replace with your new API key
openai.api_key = ''

def load_regest(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    print(f"Loaded content from {file_path}")
    return content

def save_rdf(rdf_content, rdf_file_path):
    os.makedirs(os.path.dirname(rdf_file_path), exist_ok=True)
    with open(rdf_file_path, 'w') as file:
        file.write(rdf_content)
    print(f"Saved RDF to {rdf_file_path}")

def validate_rdf(rdf_content):
    try:
        g = rdflib.Graph()
        g.parse(data=rdf_content, format="turtle")
        print("RDF is valid.")
        return True
    except Exception as e:
        print(f"RDF validation failed: {e}")
        return False
    
def validate_rdf_with_shacl(rdf_content, shacl_shapes):
    try:
        data_graph = rdflib.Graph()
        data_graph.parse(data=rdf_content, format="turtle")

        shapes_graph = rdflib.Graph()
        shapes_graph.parse(data=shacl_shapes, format="turtle")

        conforms, results_graph, results_text = pyshacl.validate(data_graph, shacl_graph=shapes_graph, 
                                                                 data_graph_format='turtle', shacl_graph_format='turtle')

        # Return both the conformity status and the results text
        return conforms, results_text

    except Exception as e:
        error_message = f"RDF was not created because: {e}"
        print(error_message)
        # Return False and the error message
        return False, error_message



def create_rdf_from_regest(regest_text):
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": regest_text
        }
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature = 0
    )

    if response.choices and response.choices[0].message:
        return response.choices[0].message['content'].strip()
    else:
        return "No RDF conversion found."

def process_files(shacl_shapes):
    input_dir = 'regesten/input'
    output_dir = 'regesten/output'

    # Loop over all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            # Construct the full file path
            file_path = os.path.join(input_dir, filename)
            
            # Load the regest
            regest = load_regest(file_path)
            
            # Create the RDF
            rdf = create_rdf_from_regest(regest)
            
            # Validate the RDF with SHACL shapes
            print(rdf)
            is_valid, report = validate_rdf_with_shacl(rdf, shacl_shapes)
            
            if is_valid:
                # Construct the output file path
                rdf_file_path = os.path.join(output_dir, filename.replace('.txt', '.rdf'))
                
                # Save the RDF
                save_rdf(rdf, rdf_file_path)
            else:
                print(f"RDF validation failed for {filename}. Report: {report}")

# Call the function to process all files
process_files(shacl_shapes)
print("Registen successfully processed.")


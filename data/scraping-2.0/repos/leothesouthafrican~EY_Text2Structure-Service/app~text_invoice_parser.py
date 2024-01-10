from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate

# Function to read invoice text from a file
def read_invoice_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Define the function that encapsulates the invoice parsing logic
def parse_invoice(file_path, model):
    # Read the invoice text from the file
    invoice_text = read_invoice_text(file_path)

    # Define the response schema
    response_schemas = [
        ResponseSchema(name="Year", description="the year when the invoice was issued, if 2 digits, assume 20XX"),
        ResponseSchema(name="Month", description="the month when the invoice was issued, converted to a number from 01 to 12"),
        ResponseSchema(name="Supplier", description="the name of the company issuing the invoice"),
        ResponseSchema(name="Supplier Country", description="the country of the company issuing the invoice"),
    ]

    # Create the StructuredOutputParser
    invoice_text_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    # Get the format instructions
    format_instructions = invoice_text_parser.get_format_instructions()

    # Prepare the prompt template with the format instructions
    prompt_template = "Please extract the following information from the invoice text:\n{format_instructions}\n\n---\n\n{invoice_text}"
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["invoice_text"],
        partial_variables={"format_instructions": format_instructions}
    )

    # Format the prompt with the actual invoice text
    _input = prompt.format_prompt(invoice_text=invoice_text)

    # Get the model's output
    output = model(_input.to_string())

    # Parse the output
    parsed_invoice_output = invoice_text_parser.parse(output)

    # Return the parsed output
    return parsed_invoice_output
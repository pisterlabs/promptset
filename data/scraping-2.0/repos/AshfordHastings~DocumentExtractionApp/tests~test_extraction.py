from langchain_app.extractor import extract_data_inline

TEST_DATA = """Alex is 5 feet tall. He lives in Texas. He has blonde hair. He is 47 years old. Theresa lives in New York. She has brown hair. """
TEST_SCHEMA = {
    "properties": {
        "name": {"type": "string"},
        "height": {"type": "number"},
        "state": {"type": "string"},
        "hair_color": {"type": "string"}
    },
    "required": ["name"]
}

def test_extract_data_inline():
    results = extract_data_inline(TEST_DATA, TEST_SCHEMA)
    print(f"RESULTS: \n{results}")

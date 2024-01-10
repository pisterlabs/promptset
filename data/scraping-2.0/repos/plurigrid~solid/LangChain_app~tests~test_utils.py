```python
import unittest
import requests
from LangChain_app.utils import parse_contract_module, parse_lifecycle_event

class TestUtils(unittest.TestCase):
    api_endpoint = "http://localhost:5000/api"

    def test_parse_contract_module(self):
        contract_module_string = "Contract Module String"
        response = requests.post(self.api_endpoint, json={"contract_module": contract_module_string})
        contract_module = response.json()
        parsed_contract_module = parse_contract_module(contract_module_string)
        self.assertEqual(parsed_contract_module, contract_module)

    def test_parse_lifecycle_event(self):
        lifecycle_event_string = "Lifecycle Event String"
        response = requests.post(self.api_endpoint, json={"lifecycle_event": lifecycle_event_string})
        lifecycle_event = response.json()
        parsed_lifecycle_event = parse_lifecycle_event(lifecycle_event_string)
        self.assertEqual(parsed_lifecycle_event, lifecycle_event)

if __name__ == "__main__":
    unittest.main()
```
import unittest
import pandas as pd
from langchain import OpenAI
import llm_interface
from prompt_template import retrainer


class TestTDD(unittest.TestCase):
    """Integration tests to format a table."""

    def test_format_table(self):
        """
        transform a csv file table into pandas table with standardized values

        async llm per record calls required by streamlit
        """
        llm = OpenAI(temperature=0, max_tokens=1000)  # type: ignore
        csv_table = "table_B.csv"
        input_table = pd.read_csv(csv_table)
        output_table = llm_interface.format_table(llm, input_table)  # type: ignore
        print(output_table)

    def test_retrain(self):
        """
        retrain AI system by addin the corrected record as a new example in the
        prompt template
        """
        llm = OpenAI(temperature=0, max_tokens=1000)  # type: ignore
        csv_table = "table_B.csv"
        input_table = pd.read_csv(csv_table)
        output_table = llm_interface.format_table(llm, input_table)  # type: ignore
        print(output_table)

        # assume on UI, user submitted a fix for row 9 in the streamlit grid table

        fixed_records = [
            {
                "_selectedRowNodeInfo": {"nodeRowIndex": 0, "nodeId": "0"},
                "Date.fields": "PolicyDate OR StartDate",
                "Date.mapping_rationale": "date like",
                "Date.value": "010101",
                "Date.transform_rationale": "ambiguous input format (mm-dd or dd-mm)",
                "EmployeeName.fields": "Employee_Name OR Name",
                "EmployeeName.mapping_rationale": "name like",
                "EmployeeName.value": "John Doe",
                "EmployeeName.transform_rationale": "mapped col vals are same AND format correct",
                "Plan.fields": "PlanType OR Plan_Name",
                "Plan.mapping_rationale": "insurance plan like",
                "Plan.value": "Gold",
                "Plan.transform_rationale": "keep pattern (Gold, Bronze, Silver)",
                "PolicyNumber.fields": "Policy_ID OR PolicyID",
                "PolicyNumber.mapping_rationale": "policy number like",
                "PolicyNumber.value": "AB12345",
                "PolicyNumber.transform_rationale": "mapped cols are same AND format correct",
                "Premium.fields": "PremiumAmount OR Cost",
                "Premium.mapping_rationale": "premium cost like",
                "Premium.value": 150,
                "Premium.transform_rationale": "mapped cols are same AND format correct",
            }
        ]
        fixed_record = fixed_records[0]
        retrainer.add_streamlit_user_fix(input_table, fixed_record)  # type: ignore
        output_table = llm_interface.format_table(llm, input_table)  # type: ignore

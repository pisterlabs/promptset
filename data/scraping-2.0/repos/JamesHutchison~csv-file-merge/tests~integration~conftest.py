from pathlib import Path
from typing import Iterable, TextIO

import pytest
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.schema.language_model import BaseLanguageModel

from table_merger.table_mergers import ColumnInfo

sample_data = Path(__file__).parent / "sample_data"


@pytest.fixture()
def example_template_csv() -> Iterable[TextIO]:
    with (sample_data / "template.csv").open() as template_file:
        yield template_file


@pytest.fixture()
def table_a() -> Path:
    return sample_data / "table_A.csv"


@pytest.fixture()
def table_b() -> Path:
    return sample_data / "table_B.csv"


@pytest.fixture()
def gpt3() -> OpenAI:
    # gpt-3.5-turbo chat is slower
    return OpenAI(max_tokens=1000, temperature=0.0)


@pytest.fixture()
def gpt4() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4", max_tokens=1000, temperature=0.0)


@pytest.fixture()
def template_column_info() -> list[ColumnInfo]:
    return [
        ColumnInfo(
            name="Date",
            type="date",
            output_format="^(0[1-9]|1[0-2])-(0[1-9]|[1-2][0-9]|3[0-1])-[0-9]{4}$",
            empty_expected=False,
            example_values=[
                "01-05-2023",
                "02-05-2023",
                "03-05-2023",
                "04-05-2023",
                "05-05-2023",
                "06-05-2023",
                "07-05-2023",
                "08-05-2023",
                "09-05-2023",
                "10-05-2023",
            ],
        ),
        ColumnInfo(
            name="EmployeeName",
            type="string",
            output_format="[A-Za-z ]+",
            empty_expected=False,
            example_values=[
                "John Doe",
                "Jane Smith",
                "Michael Brown",
                "Alice Johnson",
                "Bob Wilson",
                "Carol Martinez",
                "David Anderson",
                "Eva Thomas",
                "Frank Jackson",
                "Grace White",
            ],
        ),
        ColumnInfo(
            name="Plan",
            type="string",
            output_format="[A-Za-z]+",
            empty_expected=False,
            example_values=["Gold", "Silver", "Bronze"],
        ),
        ColumnInfo(
            name="PolicyNumber",
            type="string",
            output_format="^[A-Z]{2}[0-9]{5}$",
            empty_expected=False,
            example_values=[
                "AB12345",
                "CD67890",
                "EF10111",
                "GH12121",
                "IJ13131",
                "KL14141",
                "MN15151",
                "OP16161",
                "QR17171",
                "ST18181",
            ],
        ),
        ColumnInfo(
            name="Premium",
            type="number",
            output_format="^[0-9]+$",
            empty_expected=False,
            example_values=["150", "100", "50"],
        ),
    ]


@pytest.fixture()
def incoming_column_info() -> list[ColumnInfo]:
    return [
        ColumnInfo(
            name="Date_of_Policy",
            type="date",
            output_format="^(0[1-9]|1[0-2])/(0[1-9]|[1-2][0-9]|3[0-1])/[0-9]{4}$",
            empty_expected=False,
            example_values=[
                "05/01/2023",
                "05/02/2023",
                "05/03/2023",
                "05/04/2023",
                "05/05/2023",
                "05/06/2023",
                "05/07/2023",
                "05/08/2023",
                "05/09/2023",
                "05/10/2023",
            ],
        ),
        ColumnInfo(
            name="FullName",
            type="string",
            output_format="[A-Za-z ]+",
            empty_expected=False,
            example_values=[
                "John Doe",
                "Jane Smith",
                "Michael Brown",
                "Alice Johnson",
                "Bob Wilson",
                "Carol Martinez",
                "David Anderson",
                "Eva Thomas",
                "Frank Jackson",
                "Grace White",
            ],
        ),
        ColumnInfo(
            name="Insurance_Plan",
            type="string",
            output_format="^(Gold Plan|Silver Plan|Bronze Plan)$",
            empty_expected=False,
            example_values=["Gold Plan", "Silver Plan", "Bronze Plan"],
        ),
        ColumnInfo(
            name="Policy_No",
            type="string",
            output_format="^[A-Z]{2}-[0-9]{5}$",
            empty_expected=False,
            example_values=[
                "AB-12345",
                "CD-67890",
                "EF-10111",
                "GH-12121",
                "IJ-13131",
                "KL-14141",
                "MN-15151",
                "OP-16161",
                "QR-17171",
                "ST-18181",
            ],
        ),
        ColumnInfo(
            name="Monthly_Premium",
            type="number",
            output_format="^[0-9]+(\\.[0-9]+)?$",
            empty_expected=False,
            example_values=["150.00", "100.00", "50.00"],
        ),
        ColumnInfo(
            name="Department",
            type="string",
            output_format="^(IT|HR|Marketing|Finance|Sales|Operations|Legal|Product|Engineering|Design)$",
            empty_expected=False,
            example_values=[
                "IT",
                "HR",
                "Marketing",
                "Finance",
                "Sales",
                "Operations",
                "Legal",
                "Product",
                "Engineering",
                "Design",
            ],
        ),
        ColumnInfo(
            name="JobTitle",
            type="string",
            output_format="^[A-Za-z ]+$",
            empty_expected=False,
            example_values=[
                "Software Engineer",
                "HR Manager",
                "Marketing Coordinator",
                "Financial Analyst",
                "Sales Executive",
                "Operations Manager",
                "Attorney",
                "Product Manager",
                "Engineer",
                "Graphic Designer",
            ],
        ),
        ColumnInfo(
            name="Policy_Start",
            type="date",
            output_format="^(0[1-9]|1[0-2])/(0[1-9]|[1-2][0-9]|3[0-1])/[0-9]{4}$",
            empty_expected=False,
            example_values=[
                "05/01/2023",
                "05/02/2023",
                "05/03/2023",
                "05/04/2023",
                "05/05/2023",
                "05/06/2023",
                "05/07/2023",
                "05/08/2023",
                "05/09/2023",
                "05/10/2023",
            ],
        ),
        ColumnInfo(
            name="Full_Name",
            type="string",
            output_format="[A-Za-z ]+",
            empty_expected=False,
            example_values=[
                "John Doe",
                "Jane Smith",
                "Michael Brown",
                "Alice Johnson",
                "Bob Wilson",
                "Carol Martinez",
                "David Anderson",
                "Eva Thomas",
                "Frank Jackson",
                "Grace White",
            ],
        ),
        ColumnInfo(
            name="Insurance_Type",
            type="string",
            output_format="^(Gold|Silver|Bronze)$",
            empty_expected=False,
            example_values=["Gold", "Silver", "Bronze"],
        ),
        ColumnInfo(
            name="Policy_Num",
            type="string",
            output_format="^[A-Z]{2}-[0-9]{5}$",
            empty_expected=False,
            example_values=[
                "AB-12345",
                "CD-67890",
                "EF-10111",
                "GH-12121",
                "IJ-13131",
                "KL-14141",
                "MN-15151",
                "OP-16161",
                "QR-17171",
                "ST-18181",
            ],
        ),
        ColumnInfo(
            name="Monthly_Cost",
            type="number",
            output_format="^\\d+(\\.\\d+)?$",
            empty_expected=False,
            example_values=["150.00", "100.00", "50.00"],
        ),
    ]

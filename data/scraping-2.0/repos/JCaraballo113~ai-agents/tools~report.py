import os
from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel


def write_report(filename, html):
    if not os.path.exists("reports"):
        os.makedirs("reports")
    with open(f"reports/{filename}", "w") as f:
        f.write(html)


class WriteReportArgsSchema(BaseModel):
    filename: str
    html: str


write_report_tool = StructuredTool.from_function(
    name="write_report",
    func=write_report,
    description="Write an HTML file to disk. Use this tool whenever someone asks for a report.",
    args_schema=WriteReportArgsSchema
)

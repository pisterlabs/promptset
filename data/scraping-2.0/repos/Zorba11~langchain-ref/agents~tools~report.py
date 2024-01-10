from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel

def write_report(filename, html):
  with open(filename, 'w') as f: #open the file in write mode and call it f for reference
    f.write(html)

class WriteReportArgsSchema(BaseModel):
    filename: str
    html: str

# use structured tool to intake multiple argumenst to a function
write_report_tool = StructuredTool.from_function(
  name="write_report",
  description="Write an html file to a disk. I want to use this tool whenever someone asks for a report",
  func=write_report,
  args_schema=WriteReportArgsSchema,
)
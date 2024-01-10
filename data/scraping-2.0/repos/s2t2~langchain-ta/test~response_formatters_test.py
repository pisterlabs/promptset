
from langchain.output_parsers import PydanticOutputParser

from app.response_formatters import Student, DocumentScoring, QuestionScoring, QuestionScorings


def test_models():

    student = Student(name="Sally Student", net_id="G123456")
    print(student)

    scoring = DocumentScoring(score=3.5, comments="Great work!", confidence=1.0)
    print(scoring)

    qs = QuestionScoring(question_id="5.1", score=0.75, comments="Great work!", confidence=1.0)
    scorings = QuestionScorings(scorings=[qs])
    print(scorings)


def test_formatting_instructions():

    parser = PydanticOutputParser(pydantic_object=Student)
    instructions = parser.get_format_instructions()
    assert instructions == 'The output should be formatted as a JSON instance that conforms to the JSON schema below.\n\nAs an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}\nthe object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is the output schema:\n```\n{"description": "A student.", "properties": {"name": {"description": "The student\'s full name.", "title": "Name", "type": "string"}, "net_id": {"description": "The student\'s identifier.", "title": "Net Id", "type": "string"}}, "required": ["name", "net_id"]}\n```'

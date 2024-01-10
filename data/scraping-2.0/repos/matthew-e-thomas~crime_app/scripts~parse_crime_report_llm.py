from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number,
from langchain.llms import OpenAI

llm = OpenAI()

schema = Object(
    id="crime_report",
    description="Descriptions of crimes.",
    attributes=[
        Text(
            id="location",
            description="Place where crime occurred",
            examples=[("Gold’s Gym parking lot, 2380 Plank Road, 2/4, A person reported his vehicle’s registration sticker was stolen.", "Gold's Gym parking lot")],
        ),
        Text(
            id="address",
            description="address where crime occurred",
            examples=[("900 block Lafayette Boulevard, 2/2, A person reported her vehicle was vandalized", "900 block Lafayette Boulevard")],
        ),
        Text(
            id="date",
            description="Date the crime occurred",
            examples=[("900 block Lafayette Boulevard, 2/2, A person reported her vehicle was vandalized", "2/2")],
        ),
        Text(
            id="description",
            description="Description of the crime",
            examples=[("900 block Lafayette Boulevard, 2/2, A person reported her vehicle was vandalized", "A person reported her vehicle was vandalized")],
        ),
    ],
    examples=[
        (
            "Wine & Design, 502 Sophia Street, 1/30, The manager reported a window was shattered",
            [
                {"location": "Wine & Design", "address": "502 Sophia Street", "date": '1/30', "description": "The manager reported a window was shattered"},
            ],
        )
    ],
    many=True,
)


chain = create_extraction_chain(llm, schema)
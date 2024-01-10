from langchain.output_parsers.pydantic import PydanticOutputParser, T


class StrictPydanticOutputParser(PydanticOutputParser[T]):
    def get_format_instructions(self) -> str:
        instructions = super().get_format_instructions()
        return (
            f"{instructions}\n\n"
            "Remember to output only the JSON instance described above. "
            "Do not add anything other than the JSON instance, such as "
            "additional descriptions or code blocks."
        )

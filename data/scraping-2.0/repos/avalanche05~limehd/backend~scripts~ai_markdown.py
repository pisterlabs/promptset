from openai import OpenAI

from limehd.dependencies import get_db
from limehd.models import Program


def get_program_genre(program_name: str, program_description: str) -> str:
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=f"Определи жанр теле-программы по названию и описанию. Выбери один жанр из [ток-шоу, здоровье, /"
               f"новости, реалити, детектив, документальный, спорт, драма, криминал, образовательный, развлекательное, /"
               f"музыкальный, комедия, разное, мультфильм], который подходит. /"
               f"Название: '{program_name}'. Описание: '{program_description}. Ответь одним словом без лишних символов.")

    return response.choices[0].text


client = OpenAI(api_key='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')
db = next(get_db())
programs = db.query(Program).all()
genre_list = []
with open("genres.txt", "a", encoding="utf-8") as f:
    for program in programs:
        genre = get_program_genre(program.name, program.description).strip('\n').lower()
        genre_list.append((program.id, program.name, genre))
        f.write(f'{program.id},{program.name},{genre}\n')
        print(f'{program.name=}, {get_program_genre(program.name, program.description)=}')

print('success')


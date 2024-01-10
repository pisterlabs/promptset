import openai
import sqlalchemy as sa
import sqlalchemy.orm as orm

from . import domain, completion


def _character_prompt(name: str, mbti: str, age: int, gender: domain.Gender, description: str) -> str:
    return f"""# 페르소나
- 이름: "{name}"
- 성격 유형(MBTI): "{mbti}"
- 나이: {age}
- 성별: "{gender}"
- 설명: "{description}"

위 페르소나를 보고 그에 어울리는 캐릭터의 특징을 자세하게 소개해줘."""


def _persona_prompt(theme: str):
    return f"""- 주제: "{theme}"

위 주제에 맞게 시놉시스를 만들어줘."""


class PersonaService:

    def __init__(
            self,
            db: orm.Session,
            completer: completion.Completer):
        self._completer = completer
        self._db = db

    def add_character(
            self,
            name: str,
            age: int,
            mbti: str,
            gender: domain.Gender,
            description: str,
    ) -> domain.Persona:
        prompt = _character_prompt(name, mbti, age, gender, description)

        context = self._completer.chat([{
            'role': 'user',
            'content': prompt,
        }])

        persona = domain.Persona(
            name=name,
            mbti=mbti,
            age=age,
            gender=gender,
            description=description,
            context=context,
        )
        with self._db.begin():
            self._db.add(persona)

        return persona

    def get_all_characters(self) -> list[domain.Persona]:
        stmt = sa.select(domain.Persona)
        personas = list(self._db.execute(stmt).scalars().all())

        return personas


class SynopsisService:
    def __init__(
            self,
            db: orm.Session,
            completer: completion.Completer,
    ):
        self._completer = completer
        self._db = db

    def add_synopsis(
            self,
            theme: str,
            persona_id: int,
    ) -> domain.Synopsis:
        persona = self._db.execute(sa.select(domain.Persona).where(domain.Persona.id == persona_id).limit(1)).scalar_one()

        content = self._completer.chat([{
            'role': 'user',
            'content': _character_prompt(
                persona.name,
                persona.mbti,
                persona.age,
                persona.gender,
                persona.description,
            ),
        }, {
            'role': 'system',
            'content': persona.context,
        }, {
            'role': 'user',
            'content': _persona_prompt(theme),
        }])

        synopsis = domain.Synopsis(
            theme=theme,
            character_id=persona_id,
            content=content,
        )

        with self._db.begin(nested=True):
            self._db.add(synopsis)

        return synopsis

    def get_all_synopses(self) -> list[domain.Synopsis]:
        stmt = sa.select(domain.Synopsis)
        synopses = list(self._db.execute(stmt).scalars().all())

        return synopses

    def get_synopsis(self, synopsis_id: int) -> domain.Synopsis:
        stmt = sa.select(domain.Synopsis).where(domain.Synopsis.id == synopsis_id).limit(1)
        synopsis = self._db.execute(stmt).scalar_one()

        return synopsis

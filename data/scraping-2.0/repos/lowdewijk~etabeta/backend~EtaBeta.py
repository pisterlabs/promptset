from dataclasses import dataclass
import time
import json
from pydantic import BaseModel

from openai import AsyncOpenAI
from typing import Optional, List, Callable

from Message import Message

@dataclass
class QueryPoint:
    description: str
    tostr: Callable[[str, str], str]
    score: float

positives: dict[str, QueryPoint] = {
    "strong-argument": QueryPoint(
        description="Strong, well-reasoned and convincing arguments made.",
        tostr=lambda user, arg: f"Nice argument {user}! {arg}",
        score=1,
    ),
    "strong-evidence": QueryPoint(
        description="High-quality, reliable evidence presented.",
        tostr=lambda user, arg: f"Awesome evidence {user}! {arg}",
        score=1,
    ),
    "logic": QueryPoint(
        description="Logical consistency and coherence.",
        tostr=lambda user, arg: f"Logical consistency {user}! {arg}",
        score=1,
    ),
    "polite": QueryPoint(
        description="Respectful and polite discourse.",
        tostr=lambda user, arg: f"Respectful discourse {user}! {arg}",
        score=0.25,
    ),
}
negatives = {
    "weak_arguments": QueryPoint(
        description="Weak arguments made.",
        tostr=lambda user, arg: f"{user} made a weak argument :( {arg}",
        score=-1,
    ),
    "low_quality_evidence": QueryPoint(
        description="Low-quality evidence presented.",
        tostr=lambda user, arg: f"{user} presented low-quality evidence :( {arg}",
        score=-1,
    ),
    "logical_fallacies": QueryPoint(
        description="Logical fallacies or errors in reasoning.",
        tostr=lambda user, arg: f"{user} committed a fallacy :( {arg}",
        score=-1,
    ),
    "personal_attacks": QueryPoint(
        description="Personal attacks or ad hominem fallacies.",
        tostr=lambda user, arg: f"{user} committed an ad hominem :( {arg}",
        score=-2,
    ),
    "insults": QueryPoint(
        description="Insults or profanity.",
        tostr=lambda user, arg: f"{user} insulted someone :( {arg}",
        score=-5,
    ),
    "off_topic": QueryPoint(
        description="Off-topic or irrelevant comments.",
        tostr=lambda user, arg: f"{user} went off-topic :( {arg}",
        score=-0.5,
    ),
    "repetition": QueryPoint(
        description="Repetition of previous arguments.",
        tostr=lambda user, arg: f"{user} repeated themselves :( {arg}",
        score=-1,
    ),
    "contradiction": QueryPoint(
        description="Contradiction of previous arguments.",
        tostr=lambda user, arg: f"{user} contradicted themselves :( {arg}",
        score=-1,
    ),
    "nonsequitur": QueryPoint(
        description="Non-sequitur or non-logical arguments.",
        tostr=lambda user, arg: f"{user} made a non-sequitur :( {arg}",
        score=-1,
    ),
    "misinformation": QueryPoint(
        description="Misinformation or false claims.",
        tostr=lambda user, arg: f"{user} spread misinformation :( {arg}",
        score=-1,
    ),
    "misleading": QueryPoint(
        description="Misleading or deceptive claims.",
        tostr=lambda user, arg: f"{user} made a misleading claim :( {arg}",
        score=-1,
    ),
    "unsubstantiated": QueryPoint(
        description="Unsubstantiated claims.",
        tostr=lambda user, arg: f"{user} made an unsubstantiated claim :( {arg}",
        score=-1,
    ),
    "unreliable": QueryPoint(
        description="Unreliable sources cited.",
        tostr=lambda user, arg: f"{user} cited an unreliable source :( {arg}",
        score=-1,
    ),
    "unrelated": QueryPoint(
        description="Unrelated or tangential arguments.",
        tostr=lambda user, arg: f"{user} made an unrelated argument :( {arg}",
        score=-1,
    ),
    "unconvincing": QueryPoint(
        description="Unconvincing arguments.",
        tostr=lambda user, arg: f"{user} made an unconvincing argument :( {arg}",
        score=-0.5,
    ),
}

positive_properties = {
    p_key: {
        "type": "object",
        "items": {"type": "string"},
        "description": "Observation comments by EtaBeta. " + p_value.description,
    }
    for p_key, p_value in positives.items()
}

negative_properties = {
    n_key: {
        "type": "array",
        "items": {"type": "string"},
        "description": n_value.description,
    }
    for n_key, n_value in negatives.items()
}

json_schema = {
    "type": "object",
    "required": ["observations", "ball_in_court"],
    "properties": {
        "observations": {
            "type": "object",
            "description": "Analysis of messages with the 'observe' flag set to true.",
            "required": ["positive", "negative", "username"],
            "properties": {
                "positive": {
                    "type": "object",
                    "properties": positive_properties,
                },
                "negative": {
                    "type": "object",
                    "properties": negative_properties,
                },
            },
        },
        "number_of_observed_messages": {
            "description": "Number of messages with the 'observe' flag set to true.",
            "type": "integer",
        },
        "ball_in_court": {
            "description": "User with the initiative",
            "type": "string",
        },
    },
}

assistant_prompt = f"""
You are etabeta, an AI designed to impartially observe and analyze debates.
Your role is to evaluate the quality of arguments and evidence presented by the
participants. You will be given a chat log and should return the following
information:

- You should return a list of observations of all chat messages with
  the 'observe' flag set to true.
- Determine who has the initiative in the debate ('ball in their court').

Use this JSON schema for your response:

{json.dumps(json_schema, indent=2)}"""

class EtaBeta(BaseModel):
    messages: List[Message] = []
    in_court: Optional[str] = None
    scores: dict[str, float] = {}
    under_observation: List[int] = []

    async def query(self, debate_messages: List[Message]):
        print(debate_messages)
        if len(debate_messages) == 0:
            return
        observed_message_timestamp = debate_messages[-1].timestamp

        try:

            client = AsyncOpenAI()

            chat_log = [
                {"user": msg.username, "message": msg.message, "observe": False}
                for msg in debate_messages
            ]
            chat_log[-1]["observe"] = True
            observed_user = debate_messages[-1].username
            self.under_observation.append(observed_message_timestamp)

            print(assistant_prompt)
            response = await client.chat.completions.create(
                model="gpt-4-1106-preview",
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": assistant_prompt,
                    },
                    {"role": "user", "content": json.dumps(chat_log, indent=2)},
                ],
            )
            if response.choices[0].message.content is None:
                raise Exception("Empty response from EtaBeta.")

            resp = json.loads(response.choices[0].message.content)
            print(response.choices[0].message.content)

            self.in_court = resp.get("ball_in_court", None)
            p_observations = resp.get("observations", {}).get("positive", {})
            n_observations = resp.get("observations", {}).get("negative", {})

            timestamp = time.time_ns() // 1_000_000

            for p_key, p_value in positives.items():
                for observation in p_observations.get(p_key, []):
                    self.messages.append(
                        Message(
                            message=p_value.tostr(observed_user, observation),
                            username="Eta Beta",
                            timestamp=timestamp,
                        )
                    )
                    self.scores[observed_user] = (
                        self.scores.get(observed_user, 0) + p_value.score
                    )

            for n_key, n_value in negatives.items():
                for observation in n_observations.get(n_key, []):
                    self.messages.append(
                        Message(
                            message=n_value.tostr(observed_user, observation),
                            username="Eta Beta",
                            timestamp=timestamp,
                        )
                    )
                    self.scores[observed_user] = (
                        self.scores.get(observed_user, 0) + n_value.score
                    )
        except Exception as e:
            print(e)
        finally:
            self.under_observation.remove(observed_message_timestamp)


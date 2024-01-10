import json

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from openai.types import beta, shared_params
from openai.types.beta import assistant_create_params
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

from dataclasses import dataclass, field
from typing_extensions import Callable, Optional, Literal, List

class ToolsFunction:
    function: shared_params.FunctionDefinition
    resolver: Callable[..., object]
    @property
    def type(self) -> Literal["function"]:
        return "function"
    
    def __init__(self, function: shared_params.FunctionDefinition, resolver: Callable[..., object]):
        self.function = function
        self.resolver = resolver

    def toTool(self) -> assistant_create_params.ToolAssistantToolsFunction:
        return assistant_create_params.ToolAssistantToolsFunction(**({
            "type": self.type,
            "function": self.function,
        }))

@dataclass
class Company:
    name: str
    type: str
    pronoun: Literal["M"] | Literal["F"]
    services: str

@dataclass
class Mood:
    friendly: Optional[bool] = None
    aggressive: Optional[bool] = None
    polite: Optional[bool] = None
    unpolite: Optional[bool] = None
    kind: Optional[bool] = None
    inclusive: Optional[bool] = None

@dataclass
class OutOfContext:
    allowed: Optional[bool] = None
    pushBack: Optional[bool] = None
    friendly: Optional[bool] = None

def generateNames() -> list:
    return ["João", "José", "Gabriel", "Lucas", "Felipe", "Paulo", "Pedro", "Maria", "Ana", "Luísa", "Alessandra", "Joana", "Cris", "Marcela"]

class GenericAssistant:
    _ready: bool = False
    _assistant = None
    _threads: dict[str, object] = {}
    _toolResolvers: dict[str, Callable] = {}

    def __init__(self, assistant: beta.Assistant, toolResolvers: dict[str, Callable] = {}):
        self._assistant = assistant
        self._toolResolvers = toolResolvers
        self._ready = True

    def startNewThread(self):
        if self._ready:
            newThread = client.beta.threads.create()
            self._threads[newThread.id] = newThread
            return newThread
        return None
    
    def addMessage(self, threadId: str, content: str):
        if self._ready:
            message = client.beta.threads.messages.create(threadId, role="user", content=content)
            return message
        return None
    
    def runThread(self, threadId: str):
        if self._ready:
            run = client.beta.threads.runs.create(threadId, assistant_id=self._assistant.id)
            retrieve = client.beta.threads.runs.retrieve(run.id, thread_id=threadId)
            return retrieve
        return None
    
    def getRun(self, threadId: str, runId: str):
        if self._ready:
            run = client.beta.threads.runs.retrieve(runId, thread_id=threadId)
            if run.status == "failed" or run.status == "requires_action":
                print(run)
            if run.status == "requires_action":
                toolCalls = run.required_action.submit_tool_outputs.tool_calls
                outputs = []
                for toolCall in toolCalls:
                    if toolCall.type == "function" and toolCall.function.name in self._toolResolvers:
                        try:
                            print(self._toolResolvers[toolCall.function.name](**json.loads(toolCall.function.arguments)))
                        except Exception as e:
                            print(e)
                        outputs.append({
                            "tool_call_id": toolCall.id,
                            "output": json.dumps(self._toolResolvers[toolCall.function.name](**json.loads(toolCall.function.arguments))),
                        })
                run = client.beta.threads.runs.submit_tool_outputs(
                    run.id,
                    thread_id=threadId,
                    tool_outputs=outputs
                )
            return run
        return None
    
    def getMessages(self, threadId: str):
        if self._ready:
            messages = client.beta.threads.messages.list(threadId)
            return messages.data
        return None

def findAssistant(name: str, functions: Optional[List[ToolsFunction]] = None) -> GenericAssistant:
    assistants = client.beta.assistants.list()
    assistant = next(filter(lambda assistant: assistant.name == name, assistants.data), None)
    if assistant is not None:
        return GenericAssistant(assistant, {f.function["name"]: f.resolver for f in functions} if functions is not None else {})
    return None

def createAssistant(name: str, company: Company, playNames: Optional[List[str]] = None, mood: Optional[Mood] = None, conversationFlow: List[str] = field(default_factory=list), outOfContext: Optional[OutOfContext] = None, functions: Optional[List[ToolsFunction]] = None) -> GenericAssistant:
    assistant = client.beta.assistants.create(
        model="gpt-3.5-turbo-1106",
        name=name,
        instructions=f"""Você é um agente de call center que trabalha em {company.type} que se chama {company.name} (refira-se sempre no {"masculino" if company.pronoun == "M" else "feminino"}) que possui o seguinte serviço: {company.services}.
                Escolha aleatóriamente dentro da lista a seguir o seu nome: {", ".join(playNames if playNames and playNames.length > 0 else generateNames())}. Você usará este nome para se apresentar ao cliente.
                Voce deve atender as pessoas de maneira {"neutra" if mood is None or not mood else f"{"cordial" if mood.friendly else "agressiva" if mood.aggressive else "seca"}, {"formal" if mood.polite else "muito informal" if mood.unpolite else "sem muita formalidade"}, tratando-as sempre de uma forma {"gentil" if mood.kind else "mesquinha"}{" e inclusiva" if mood.inclusive else ""}"}.
                {"\n".join(conversationFlow)}
                {f"Se a conversa parecer estar totalmente fora de contexto, {"continue conversando com o cliente dentro do assunto que o cliente está puxando." if outOfContext.allowed else f"expresse isso de forma {"amigável" if outOfContext.friendly else "ríspida"} explicando porque a conversa parece estar fora do contexto, {" e trazendo imediatamente de volta ao contexto inicial proposto." if outOfContext.pushBack else ""}"}" if outOfContext is not None else ""}""",
        tools=[{
            "type": "retrieval"
        }] + [function.toTool() for function in functions] if functions is not None else [],
    )
    return GenericAssistant(assistant, {f.function["name"]: f.resolver for f in functions} if functions is not None else {})

def getAssistant(name: str, company: Company, playNames: Optional[List[str]] = None, mood: Optional[Mood] = None, conversationFlow: List[str] = field(default_factory=list), outOfContext: Optional[OutOfContext] = None, functions: Optional[List[ToolsFunction]] = None) -> GenericAssistant:
    assistant = findAssistant(name, functions)
    if assistant is None:
        assistant = createAssistant(name, company, playNames, mood, conversationFlow, outOfContext, functions)
    return assistant
from .base import Connector
from ..typedef import dataclass, field, Optional, Literal, AsyncGenerator, AsyncIterator, ABC

@dataclass
class Endpoint(ABC):
    conn: Connector

@dataclass
class APIResource(Endpoint):
    conn: Connector
    id: T
    
    def __post_init__(self):
        T = get_args(type(self).__orig_bases__[0])[0] # type: ignore
        if not is_APIResourceId(T, self.id):
            raise TypeError(f"API resource id mismatch: got {self.id} for {T.__name__}")
    
    @abstractmethod
    async def retrieve(self):
        '''Retrieve the raw API resource.'''

@dataclass
class Message:
    id: MessageId
    content: list[Content]
    file_ids: list[FileId] = field(default_factory=list)
    
    def to_schema(self):
        schema: dict = {
            "role": "user",
            "content": self.content
        }
        if self.file_ids:
            schema['file_ids'] = self.file_ids
        
        return schema

@dataclass
class StepHandle(APIResource[StepId]):
    '''One step in a run.'''
    
    run_id: RunId
    thread_id: ThreadId
    
    async def retrieve(self) -> Any:
        return await self.openai.beta.threads.runs.steps.retrieve(
            step_id=self.id, thread_id=self.thread_id, run_id=self.run_id
        )

@dataclass
class ThreadMessageHandle(APIResource[MessageId]):
    '''Handle for a single message.'''
    
    thread_id: ThreadId
    
    async def retrieve(self):
        '''Retrieve thread message API resource.'''
        return await self.openai.beta.threads.messages.retrieve(
            message_id=self.id,
            thread_id=self.thread_id
        )
    
    async def update(self, *, metadata: Optional[object]|NotGiven=NOT_GIVEN):
        '''Update the thread message (only metadata).'''
        return await self.openai.beta.threads.messages.update(
            message_id=self.id,
            thread_id=self.thread_id,
            metadata=metadata
        )

@dataclass
class RunHandle(APIResource[RunId]):
    '''An in-progress completion in a thread.'''
    
    thread_id: ThreadId
    
    @dataclass
    class StepEndpoint(Endpoint):
        run: 'RunHandle'
        
        def __call__(self, step_id: StepId):
            return StepHandle(self.run.openai, step_id, self.run.id, self.run.thread_id)
        
        async def iter(self):
            res = await self.run.openai.beta.threads.runs.steps.list(
                run_id=self.run.id, thread_id=self.run.thread_id
            )
            async for step in res:
                yield step
        
        async def list(self):
            ls = []
            async for step in self.iter():
                ls.append(step)
            return ls
    
    @cached_property
    def step(self):
        return self.StepEndpoint(self)
    
    async def __aenter__(self):
        return aiter(self)
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        res = await self.retrieve()
        if res.status in {"in_progress", "queued"}:
            await self.cancel()
        
        if issubclass(exc_type, StopAsyncIteration):
            return True
    
    async def retrieve(self):
        return await self.openai.beta.threads.runs.retrieve(
            run_id=self.id, thread_id=self.thread_id
        )
    
    @property
    def thread(self):
        self.openai.beta.threads.runs.steps
        return ThreadHandle(self.openai, self.thread_id)
    
    async def cancel(self):
        '''Cancel the run.'''
        await self.openai.beta.threads.runs.cancel(
            run_id=self.id, thread_id=self.thread_id
        )
    
    async def __aiter__(self) -> AsyncGenerator[ActionRequired|StepHandle, Optional[object]]:
        res = await self.retrieve()
        while True:
            match res.status:
                # If it's cancelling, assume it will be cancelled and exit early
                case "cancelled"|"cancelling":
                    break
                
                case "completed": break
                case "expired": raise ExpiredError()
                case "failed": raise APIError(res.last_error)
                
                case "in_progress"|"queued":
                    res = await self.openai.beta.threads.runs.retrieve(
                        run_id=res.id,
                        thread_id=self.thread.id
                    )
                
                case "requires_action":
                    assert res.required_action is not None
                    
                    # Yield to the caller for each action required
                    # This lets us keep track of tool ids internally
                    tool_outputs = []
                    for tool_call in res.required_action.submit_tool_outputs.tool_calls:
                        tool_id = tool_call.id
                        func = tool_call.function
                        output = yield ActionRequired(func.name, func.arguments)
                        tool_outputs.append({
                            "tool_call_id": tool_id,
                            "output": json.dumps(output)
                        })
                    
                    res = await self.openai.beta.threads.runs.submit_tool_outputs(
                        run_id=res.id,
                        thread_id=self.thread.id,
                        tool_outputs=tool_outputs
                    )
                
                case status:
                    raise NotImplementedError(f"Unknown run status {status!r}")

@dataclass
class ThreadHandle(APIResource[ThreadId]):
    '''An abstract conversation thread.'''
    
    @dataclass
    class MessageEndpoint:
        thread: 'ThreadHandle'
        
        def __call__(self, message_id: MessageId):
            return ThreadMessageHandle(
                self.thread.openai, message_id, self.thread.id
            )
        
        async def create(self, msg: str) -> ThreadMessageHandle:
            '''Add a message to the thread.'''
            
            result = await self.thread.openai.beta.threads.messages.create(
                thread_id=self.thread.id,
                content=msg,
                role="user"
            )
            
            return self(result.id)
        
        async def list(self, *,
            order: Optional[Literal['asc', 'desc']]=None,
            after: Optional[MessageId]=None
        ) -> AsyncIterator[Message]:
            '''List messages in the thread.'''
            
            results = await self.thread.openai.beta.threads.messages.list(
                self.thread.id, order=order or NOT_GIVEN, after=after or NOT_GIVEN
            )
            async for result in results:
                parts = []
                for content in result.content:
                    match content.type:
                        case "image_file":
                            parts.append(
                                ImageContent(content.image_file.file_id) # type: ignore
                            )
                        
                        case "text":
                            parts.append(
                                TextContent(content.text.value) # type: ignore
                            )
                        
                        case _:
                            raise TypeError(f"Unknown content type: {content.type!r}")
                
                yield Message(result.id, parts, result.file_ids)
    
    @cached_property
    def message(self):
        return self.MessageEndpoint(self)
    
    @dataclass
    class RunEndpoint:
        thread: 'ThreadHandle'
        
        def __call__(self, run: RunId):
            return RunHandle(self.thread.openai, run, self.thread.id)
        
        async def iter(self, *,
            order: Literal['asc', 'desc']|NotGiven=NOT_GIVEN,
            after: RunId|NotGiven=NOT_GIVEN,
            limit: int|NotGiven=NOT_GIVEN
        ):
            '''Iterate over thread run API resources.'''
            result = await self.thread.openai.beta.threads.runs.list(
                thread_id=self.thread.id,
                after=after,
                limit=limit,
                order=order
            )
            async for run in result:
                yield run
        
        async def list(self):
            '''List all runs in the thread.'''
            async for run in self.iter():
                yield RunHandle(self.thread.openai, run.id, self.thread.id)
        
        async def create(self, assistant: AssistantId):
            '''Create a new run.'''
            
            res = await self.thread.openai.beta.threads.runs.create(
                thread_id=self.thread.id,
                assistant_id=assistant
            )
            return RunHandle(self.thread.openai, res.id, self.thread.id)
    
    @cached_property
    def run(self):
        return self.RunEndpoint(self)
    
    async def retrieve(self):
        return await self.openai.beta.threads.retrieve(
            thread_id=self.id
        )
    
    async def delete(self):
        '''Delete the thread.'''
        await self.openai.beta.threads.delete(self.id)

@dataclass
class AssistantFileHandle(APIResource[FileId]):
    '''Abstract handle to assistant file.'''
    
    assistant_id: AssistantId
    
    async def remove(self):
        '''Remove the file from the assistant.'''
        return await self.openai.beta.assistants.files.delete(
            file_id=self.id,
            assistant_id=self.assistant_id
        )
    
    async def retrieve(self):
        '''Retrieve API resource specification.'''
        return await self.openai.beta.assistants.files.retrieve(
            file_id=self.id,
            assistant_id=self.assistant_id
        )

@dataclass
class AssistantHandle(APIResource[AssistantId]):
    '''Abstract handle to assistant API resource.'''
    
    @dataclass
    class FileEndpoint:
        assistant: 'AssistantHandle'
        
        def file(self, file_id: FileId):
            return AssistantFileHandle(
                self.assistant.openai, file_id, self.assistant.id
            )
        
        async def assign(self, file_id: FileId):
            '''Assign a file to the assistant.'''
            
            # Technically this could be an endpoint on AssistantFileHandle, but
            #  this makes less intuitive sense because we're conceptually
            #  modifying the assistant, not its file.
            return await self.assistant.openai.beta.assistants.files.create(
                assistant_id=self.assistant.id,
                file_id=file_id
            )
        
        async def list(self):
            result = await self.assistant.openai.beta.assistants.files.list(
                assistant_id=self.assistant.id
            )
            async for file in result:
                yield file
    
    @cached_property
    def file(self):
        return self.FileEndpoint(self)
    
    async def retrieve(self):
        '''Retrieve API resource specification.'''
        return await self.openai.beta.assistants.retrieve(
            assistant_id=self.id
        )
    
    async def update(self, *,
        description: Optional[str]|NotGiven=NOT_GIVEN,
        file_ids: list[str]|NotGiven=NOT_GIVEN,
        instructions: Optional[str]|NotGiven=NOT_GIVEN,
        model: str|NotGiven=NOT_GIVEN,
        tools: list[dict]|NotGiven=NOT_GIVEN
    ):
        '''Update assistant configuration.'''
        
        return await self.openai.beta.assistants.update(
            assistant_id=self.id,
            description=description,
            file_ids=file_ids,
            instructions=instructions,
            model=model,
            tools=tools # type: ignore
        )
    
    async def delete(self):
        '''Delete assistant.'''
        await self.openai.beta.assistants.delete(
            assistant_id=self.id
        )

@dataclass
class FileHandle(APIResource[FileId]):
    '''File separate from an assistant.'''
    
    async def retrieve(self):
        return await self.openai.files.retrieve(self.id)
    
    async def content(self):
        '''Retrieve file content.'''
        return await self.openai.files.content(self.id)
    
    async def delete(self):
        '''Delete file from OpenAI account.'''
        await self.openai.files.delete(self.id)

class Adapter:
    '''A realized connection to the LLM provider.'''
    
    @dataclass
    class AssistantEndpoint:
        conn: Connection
        
        def __call__(self, assistant_id: AssistantId):
            return AssistantHandle(self.conn, assistant_id)
        
        async def create(self,
            model: str,
            name: str,
            description: str,
            instructions: str,
            tools: list,
            file_ids: list[FileId]|NotGiven=NOT_GIVEN
        ):
            return await self.conn.beta.assistants.create(
                model=model,
                name=name,
                description=description,
                instructions=instructions,
                tools=tools,
                file_ids=file_ids
            )
    
    @cached_property
    def assistant(self):
        return self.AssistantEndpoint(self.openai)
    
    @dataclass
    class FileEndpoint:
        openai: Connection
        
        def __call__(self, file_id: FileId):
            return FileHandle(self.openai, file_id)
        
        async def create(self, *,
            content: IO[bytes]|bytes|PathLike[str],
            contentType: Optional[str]=None,
            filename: Optional[str]=None,
            headers: Optional[Mapping[str, str]]=None
        ):  
            if headers:
                file = (filename, content, contentType, headers)
            else:
                file = (filename, content, contentType)
            
            return await self.openai.files.create(
                file=file,
                purpose="assistants"
            )
        
        async def list(self):
            '''List all files on the OpenAI account.'''
            
            async for file in await self.openai.files.list():
                yield file
    
    @cached_property
    def file(self):
        return self.FileEndpoint(self.openai)
    
    @dataclass
    class ThreadEndpoint:
        openai: libopenai.AsyncClient
        
        def __call__(self, id: ThreadId):
            return ThreadHandle(self.openai, id)
        
        async def create(self, messages: Optional[list[Message]]=None):
            msgs = [msg.to_schema() for msg in messages] if messages else NOT_GIVEN
            return await self.openai.beta.threads.create(
                messages=msgs # type: ignore
            )
    
    @cached_property
    def thread(self):
        return self.ThreadEndpoint(self.openai)
    
    def __init__(self, openai: libopenai.AsyncClient, api_key: str):
        self.api_key = api_key
        self.openai = openai
    
    async def chat(self, model: str, messages: list):
        '''Create a chat completion.'''
        
        res = await self.openai.chat.completions.create(
            model=model,
            messages=messages
        )
        
        return res.choices[0].message.content
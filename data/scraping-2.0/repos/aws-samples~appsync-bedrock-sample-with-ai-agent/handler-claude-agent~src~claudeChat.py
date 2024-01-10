"""OpenAI chat wrapper."""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
)

from langchain.callbacks.manager import (
    CallbackManagerForLLMRun,
)
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    BaseMessage,
    ChatGeneration,
    ChatMessage,
    ChatResult,
)
from .claudeInvoker import claude_bedrock
from .chatResponder import ChatResponder

class ChatBedrockClaude(BaseChatModel):
   
    system_role: str = ''

    def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> ChatResult:
        raise "Unimplemented"

    def _llm_type(self) -> str:
        return "claude-chat"

    def _generate( self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> ChatResult:

        escaped_system_role = self.system_role.replace('Human: ', 'user:').replace('Assistant: ', 'agent:')
        claude_prompt = f'''Human: {escaped_system_role}\n{messages[0].content}\nAssistant: '''
        claude_result = claude_bedrock(claude_prompt, stop)

        # Some custom parsing to get inner dialog for chat.
        if 'Final Answer:' in claude_result.lower():
            claude_result = claude_result.lower().split('final answer:')[1]

        if'Action:' in claude_result:
            claude_thoughts = claude_result.split('Action:')[0]
            if ('Thought:' in claude_result):
                claude_thoughts = claude_thoughts.split('Thought:')[1]
            ChatResponder.instance.publish_agent_dialog(claude_thoughts)

        if "Thought:" not in claude_result and "Final Answer:" not in claude_result and "Action:" not in claude_result:
            claude_result = f'Final Answer: {claude_result}'

        return ChatResult(generations=[ChatGeneration(message=ChatMessage(content=claude_result, role='assistant'))])

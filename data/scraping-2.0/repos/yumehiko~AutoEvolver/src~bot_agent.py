from .role import Role
import openai

class BotAgent:
    """
    要求に応じたテキストを返す、柔軟なChatBotエージェント。
    """

    def __init__(self) -> None:
        self.context: list[dict[str, str]] = []

    def response(self, prompt: str, role: Role = Role.system, model: str = "gpt-3.5-turbo") -> str:
        """
        文脈を記憶せず、promptに対する応答を取得する。
        """
        persona_message = {"role": role.name, "content": prompt}

        response_data = openai.ChatCompletion.create(  # type: ignore[no-untyped-call]
            model=model,
            messages=[persona_message],
        )

        response: str = response_data["choices"][0]["message"]["content"]
        return response
    
    def add_context(self, context: str, role: Role = Role.system) -> None:
        """
        contextを記憶する。
        """
        context_message = {"role": role.name, "content": context}
        self.context.append(context_message)

    def response_to_context(self, model: str="gpt-3.5-turbo") -> str:
        """
        contextに対して応答を返す。応答も記憶する。
        """
        response_data = openai.ChatCompletion.create(  # type: ignore[no-untyped-call]
            model=model,
            messages=self.context,
        )

        response: str = response_data["choices"][0]["message"]["content"]
        response_context = {"role": Role.assistant.name, "content": response}
        self.context.append(response_context)
        return response
    
    def compress_to_summary(self, model: str="gpt-3.5-turbo") -> None:
        
        content = "Please compress the following text as much as possible while preserving its meaning. It can be in a form that is not human readable. You are free to use any characters and expressions you wish. You may use emoji and symbols."
        system_command = {"role": Role.system.name, "content": content}
        self.context.append(system_command)

        summary_data = openai.ChatCompletion.create(  # type: ignore[no-untyped-call]
            model=model,
            messages=self.context,
        )
        
        # 要約を整形する。
        summary = summary_data["choices"][0]["message"]["content"]
        memorable_response = {"role": "assistant", "content": summary}

        # 文脈をクリアし、要約を追記する
        self.clear_context()
        self.context.append(memorable_response)
        print("要約化しました。")

    def clear_context(self) -> None:
        """
        文脈をクリアする。
        """
        self.context = []
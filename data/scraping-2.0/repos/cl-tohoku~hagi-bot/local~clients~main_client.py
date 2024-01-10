from .models.openai_model import OpenAIChat
from .prompts.system_instructions import main_instruction_template
from .models.slots import Slot
from .models.roles import Role
from .models.config import ModelConfig

class MainClient(OpenAIChat):

    def __init__(self, *, model_config: ModelConfig) -> None:
        super().__init__(model_config=model_config)
        self.system_instruction_template = main_instruction_template
        self.slot_contents = Slot.initial_slot_contents()

    @property
    def remaining_slots(self) -> list[Slot]:
        return [slot for slot in Slot if self.slot_contents[slot] is None]

    @property
    def conversation_done(self) -> bool:
        return not self.remaining_slots

    @property
    def instruction_remaining_slots(self) -> str:
        if not self.conversation_done:
            instruction_lines = [f"会話を進め、残る以下の項目を決定してください。"]
            for slot in self.remaining_slots:
                instruction_lines.append(f"- {slot.text}")

            instruction = "\n".join(instruction_lines)
        else:
            instruction = "全ての項目が決定しました。会話を締めてください。"
        
        return instruction
    
    @property
    def filled_slots(self) -> list[Slot]:
        return [slot for slot in Slot if self.slot_contents[slot] is not None]

    @property
    def no_slots_filled(self) -> bool:
        return not self.filled_slots

    @property
    def instruction_filled_slots(self) -> str:
        if not self.no_slots_filled:
            instruction_lines = []
            for slot in self.filled_slots:
                instruction_lines.append(f"- {slot.text}: {self.slot_contents[slot]}")

            instruction = "\n".join(instruction_lines)
        else:
            instruction = "まだ何も決まっていません。会話を進めてください。"

        return instruction

    @property
    def system_instruction(self) -> str:
        return self.system_instruction_template.format(
            instruction_remaining_slots=self.instruction_remaining_slots,
            instruction_filled_slots=self.instruction_filled_slots,
        )
    
    @classmethod
    def character_name(role: Role) -> str:
        if role == Role.SYSTEM:
            raise ValueError("SYSTEM role has no character name.")
        return {
            Role.USER: "ユウキ",
            Role.ASSISTANT: "シズカ"
        }[role]
    

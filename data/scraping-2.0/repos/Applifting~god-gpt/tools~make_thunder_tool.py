from langchain.tools.base import BaseTool
import simpleaudio as sa

class MakeThunderTool(BaseTool):
    name = "make_thunder"
    description = "When you want to make lightning and thunder"

    def _run(self, query: str, run_manager = None) -> str:
        thunder = sa.WaveObject.from_wave_file("./sound/thunder.wav")
        thunder.play()
        return f"Thunder has been made. (Majke some witty comment about it)"
    
    async def _arun(self, query: str, run_manager = None) -> str:
        raise NotImplementedError("custom_search does not support async")

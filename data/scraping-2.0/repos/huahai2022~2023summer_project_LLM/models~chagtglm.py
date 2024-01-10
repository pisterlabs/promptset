import gc
import time
from abc import ABC
import torch
from langchain.llms.base import LLM
from typing import Optional ,List

from models.loader import LoadCheckpoint
from models.base import (BaseAnswer,AnswerResult)
class ChatGLM(BaseAnswer,LLM,ABC):
	max_token:int=2048
	temperature:float=0.01
	top_p=0.9
	checkPoint:LoadCheckpoint=None
	history_len:int=10
	def __init__(self,checkPoint:LoadCheckpoint=None):
		super().__init__()
		self.checkPoint=checkPoint
	@property
	def _llm_type(self) -> str:
		return "ChatGLM"
	@property
	def _check_point(self) -> LoadCheckpoint:
		return self.checkPoint
	@property
	def _history_len(self) -> int:
		return self.history_len

	def set_history_len(self, history_len: int=10) -> None:
		self.history_len=history_len
	def _call(self,prompt:str,stop:Optional[List[str]]=None)->str:
		print(f"_call:{prompt}")
		response,_=self.checkPoint.model.chat(
			self.checkPoint.tokenizer,
			prompt,
			history=[],
			max_length=self.max_token,
			temperature=self.temperature
		)
		print(f"response:{response}")
		return response

	def generatorAnswer(self, prompt: str,
						history: List[List[str]] = [],
						streaming: bool = False):
		if streaming:
			history+=[[]]
			for inum,(stream_resp,_) in enumerate(self.checkPoint.model.stream_chat(
				self.checkPoint.tokenizer,
				prompt,
				history=history[-self.history_len:-1] if self.history_len>1 else [],
				# 如果self.history_len大于1，则history变量的取值是history[-self.history_len:-1]，
				# 即取history列表中倒数第self.history_len个元素到倒数第2个元素（不包括最后一个元素）。
				# 这样做的目的是将history列表限制为固定的长度，以便在模型输入时使用。
				max_length=self.max_token,
				temperature=self.temperature
			)):
				history[-1]=[prompt,stream_resp]
				answer_result=AnswerResult()
				answer_result.history=history
				#历史记录全部存起来了
				answer_result.llm_output={"answer":stream_resp}
				yield answer_result
				#yield answer_result是用于返回聊天模型的输出结果，以便在调用方中逐步获取答案
		else:
			response,_ =self.checkPoint.model.chat(
				self.checkPoint.tokenizer,
				prompt,
				history=history[-self.history_len:-1] if self.history_len>0 else [],
				max_length=self.max_token,
				temperature=self.temperature
			)
			#清理内存和显存
			gc.collect()
			device_id = "0" if torch.cuda.is_available() else None
			#这个地方默认使用三块卡来部署模型
			for i in range(3):
				CUDA_DEVICE = f"{self.checkPoint.llm_device}:{device_id}"
				with torch.cuda.device(CUDA_DEVICE):
					torch.cuda.empty_cache()
					torch.cuda.ipc_collect()
			history+=[[prompt,response]]
			answer_result=AnswerResult()
			answer_result.history=history
			answer_result.llm_output={"answer":response}
			yield answer_result


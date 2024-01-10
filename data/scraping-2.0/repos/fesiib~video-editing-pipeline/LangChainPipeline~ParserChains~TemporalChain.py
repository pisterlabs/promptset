import os
import ast
import json

from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser

from langchain.chains import LLMChain

from LangChainPipeline.PromptTemplates.temporal_position_prompt import get_temporal_position_prompt_chat as get_temporal_position_prompt
from LangChainPipeline.PromptTemplates.temporal_transcript_prompt import get_temporal_transcript_prompt_chat as get_temporal_transcript_prompt
from LangChainPipeline.PromptTemplates.temporal_visual_prompt import get_temporal_visual_prompt_chat as get_temporal_visual_prompt
from LangChainPipeline.PydanticClasses.TemporalSegments import TemporalSegments
from LangChainPipeline.PydanticClasses.ListElements import ListElements

from LangChainPipeline.DataFilters.general_filters import filter_metadata_skipped
from LangChainPipeline.DataFilters.semantic_filters import filter_metadata_by_semantic_similarity

class TemporalChain():
    def __init__(
        self,
        verbose=False,
        top_k = 10,
        neighbors_left = 0,
        neighbors_right = 0,
        video_id="4LdIvyfzoGY",
        interval=10,
        temperature=0.1,
        model_name="gpt-4-1106-preview",
    ):
        self.visual_metadata = None
        self.transcript_metadata = None
        self.context = None
        self.interval = None
        self.video_id = None
        self.set_video(video_id, interval)
        self.position = TemporalPositionChain(
            verbose,
            temperature=temperature,
            model_name=model_name,
        )
        self.transcript = TemporalTranscriptChain(
            verbose=verbose,
            top_k=top_k,
            neighbors_left=neighbors_left,
            neighbors_right=neighbors_right,
            temperature=temperature,
            model_name=model_name,
        )
        self.visual = TemporalVisualChain(
            verbose=verbose,
            top_k=top_k,
            neighbors_left=neighbors_left,
            neighbors_right=neighbors_right,
            temperature=temperature,
            model_name=model_name,
        )

        self.top_k = top_k
        self.neighbors_left = neighbors_left
        self.neighbors_right = neighbors_right
        print("Initialized TemporalChain")

    def set_video(self, video_id, interval):
        if self.video_id == video_id and self.interval == interval:
            return

        self.video_id = video_id
        
        self.is_combined = True
        metadata_filepath = f"metadata/{video_id}_{str(interval)}_combined.txt"
        if os.path.exists(metadata_filepath) == False:
            print("ERROR: Metadata file does not exist: ", metadata_filepath)
            metadata_filepath = f"metadata/{video_id}_{str(interval)}.txt"
            self.is_combined = False

        self.visual_metadata = []
        self.transcript_metadata = []
        with open(metadata_filepath) as f:
            raw_lines = f.readlines()
            for line in raw_lines:
                interval = ast.literal_eval(line.rstrip())
                visual_data = {
                    "action": interval["action_pred"],
                    "abstract_caption": interval["synth_caption"],
                    "dense_caption": interval["dense_caption"],
                }
                visual_data_str = (interval["synth_caption"].strip() + ", " 
                    + interval["dense_caption"].strip() + ", " 
                    + interval["action_pred"].strip())
                
                if self.is_combined:
                    visual_data = {
                        "action": interval["action_pred"],
                        "abstract_caption": interval["synth_caption"],
                        "objects": interval["objects"],
                        "dense_caption": interval["dense_caption_2"],
                    }
                    visual_data_str = (json.dumps(interval["synth_caption"]).strip() + ", "
                        + json.dumps(interval["dense_caption_2"]).strip() + ", "
                        + json.dumps(interval["action_pred"]).strip() + ", "
                        + json.dumps(interval["objects"]).strip())

                transcript = interval["transcript"].strip()
                self.visual_metadata.append({
                    "start": interval["start"],
                    "end": interval["end"],
                    "data": visual_data_str,
                    "structured_data": visual_data,
                })
                self.transcript_metadata.append({
                    "start": interval["start"],
                    "end": interval["end"],
                    "data": transcript,
                })
            self.context = [f'The video ends at {self.visual_metadata[-1]["end"]}']
        print("Set video TemporalChain")

    def set_parameters(self, top_k, neighbors_left, neighbors_right):
        self.top_k = top_k
        self.neighbors_left = neighbors_left
        self.neighbors_right = neighbors_right
        self.transcript.set_parameters(top_k, neighbors_left, neighbors_right)
        self.visual.set_parameters(top_k, neighbors_left, neighbors_right)
        print("Set parameters")

    def run(self, original_command, command, label, offsets,
        current_player_position = 0, skipped_segments=[]
    ):
        additional_context = [
            f'The original command was: {original_command}',
        ]
        if label == "position":
            current_position_context = [f'You are at {current_player_position} seconds']
            return self.position.run(
                context=self.context + additional_context + current_position_context,
                command=command,
                offsets=offsets,
            )
        elif label == "transcript":
            return self.transcript.run(
                context=self.context + additional_context,
                command=command,
                metadata=filter_metadata_skipped(self.transcript_metadata, skipped_segments),
                offsets=offsets,
            )
        elif label == "video":
            return self.visual.run(
                context=self.context + additional_context,
                command=command,
                metadata=filter_metadata_skipped(self.visual_metadata, skipped_segments),
                offsets=offsets,
            )
        elif label == "other":
            print("Detected 'other' temporal reference: ", command)
            return []
        else:
            print("ERROR: Invalid label: ", label)
            return []


class TemporalPositionChain():
    def __init__(
            self,
            verbose=False,
            temperature=0.1,
            model_name="gpt-4-1106-preview",
    ):
        self.llm = ChatOpenAI(temperature=temperature, model_name=model_name)
        self.parser = PydanticOutputParser(pydantic_object=TemporalSegments)

        self.prompt_template = get_temporal_position_prompt({
            "format_instructions": self.parser.get_format_instructions(),
        })

        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template,
            verbose=verbose,
            output_parser=self.parser,
        )
        print("Initialized TemporalPositionChain")

    def run(self, context, command, offsets):
        try:
            result = self.chain.predict(
                context=json.dumps(context),
                command=json.dumps(command),
            )
        except:
            print("ERROR: Failed to parse command: ", command)
            return []

        segments = []
        for segment in result.segments:
            segments.append({
                "start": segment.start,
                "finish": segment.finish,
                "explanation": ["Explicit reference to time"],
                "source": command.copy(),
                "offsets": offsets.copy(),
            })
        return segments

class TemporalTranscriptChain():
    def __init__(
        self,
        verbose=False,
        top_k = 10,
        neighbors_left = 0,
        neighbors_right = 0,
        temperature=0.1,
        model_name="gpt-4-1106-preview",
    ):
        self.llm = ChatOpenAI(temperature=temperature, model_name=model_name)
        self.parser = PydanticOutputParser(pydantic_object=ListElements)

        self.prompt_template = get_temporal_transcript_prompt({
            "format_instructions": self.parser.get_format_instructions(),
        })

        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template,
            verbose=verbose,
            output_parser=self.parser,
        )

        self.top_k = top_k
        self.neighbors_left = neighbors_left
        self.neighbors_right = neighbors_right
        print("Initialized TemporalTranscriptChain")

    def set_parameters(self, top_k, neighbors_left, neighbors_right):
        self.top_k = top_k
        self.neighbors_left = neighbors_left
        self.neighbors_right = neighbors_right

    def run(self, context, command, metadata, offsets):
        filtered_metadata = filter_metadata_by_semantic_similarity(
            targets=command,
            candidates=metadata,
            k=self.top_k,
            neighbors_left=self.neighbors_left,
            neighbors_right=self.neighbors_right,
        )

        try: 
            result = self.chain.predict(
                context=json.dumps(context),
                metadata=json.dumps([data["data"] for data in filtered_metadata]),
                command=command,
            )
        except:
            print("ERROR: Failed to parse command: ", command)
            return []

        segments = []
        for element in result.list_elements:
            index = int(element.index)
            explanation = element.explanation
            if index >= len(filtered_metadata) or index < 0:
                print("ERROR: Invalid index: ", index, len(filtered_metadata), command)
                continue
            segments.append({
                "start": filtered_metadata[index]["start"],
                "finish": filtered_metadata[index]["end"],
                "explanation": [explanation],
                "source": command.copy(),
                "offsets": offsets.copy(),
            })

        return segments
    

class TemporalVisualChain():
    def __init__(
        self,
        verbose=False,
        top_k = 10,
        neighbors_left = 0,
        neighbors_right = 0,
        temperature=0.1,
        model_name="gpt-4-1106-preview",
    ):
        self.llm = ChatOpenAI(temperature=temperature, model_name=model_name)
        self.parser = PydanticOutputParser(pydantic_object=ListElements)

        self.prompt_template = get_temporal_visual_prompt({
            "format_instructions": self.parser.get_format_instructions(),
        })

        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template,
            verbose=verbose,
            output_parser=self.parser,
        )

        self.top_k = top_k
        self.neighbors_left = neighbors_left
        self.neighbors_right = neighbors_right
        print("Initialized TemporalTranscriptChain")

    def set_parameters(self, top_k, neighbors_left, neighbors_right):
        self.top_k = top_k
        self.neighbors_left = neighbors_left
        self.neighbors_right = neighbors_right

    def run(self, context, command, metadata, offsets):
        filtered_metadata = filter_metadata_by_semantic_similarity(
            targets=command,
            candidates=metadata,
            k=self.top_k,
            neighbors_left=self.neighbors_left,
            neighbors_right=self.neighbors_right,
        )

        print("HERE: ", command)

        try:
            # #dummy
            
            # result = ListElements.get_dummy_instance()
            result = self.chain.predict(
                context=json.dumps(context),
                metadata=json.dumps([data["structured_data"] for data in filtered_metadata]),
                command=json.dumps(command),
            )
        except:
            print("ERROR: Failed to parse command: ", command)
            return []

        segments = []
        for element in result.list_elements:
            index = int(element.index)
            explanation = element.explanation
            if index >= len(filtered_metadata) or index < 0:
                print("ERROR: Invalid index: ", index, len(filtered_metadata), command)
                continue
            segments.append({
                "start": filtered_metadata[index]["start"],
                "finish": filtered_metadata[index]["end"],
                "explanation": [explanation],
                "source": command.copy(),
                "offsets": offsets.copy(),
            })

        return segments
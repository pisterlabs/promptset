from abc import ABC, abstractmethod
import openai
from dataclasses import dataclass
from typing import Union, Dict, Any
import os
import json
import jsonlines
from dotenv import load_dotenv
load_dotenv("../.env")

from finetuna.utils import timestr, dataclass_to_dict, copy_file, dict_without_nones, write_to_jsonl
from finetuna.consts import OPENAI_API_KEY, FINETUNES_PATH, DATA_PATH
from finetuna.datagen.gen import DataHolder

openai.api_key = OPENAI_API_KEY

@dataclass
class FTConfig:
    model : str

@dataclass
class OpenAI_FTConfig(FTConfig):
    n_epochs : Union[int, None] = None
    batch_size : Union[int, None] = None
    learning_rate_multiplier : Union[float, None] = None

def openai_finetune_file_upload(datagen : Union[DataHolder, str]):
    """
    Uploads a file to the OpenAI API for fine-tuning.
    """
    if isinstance(datagen, str):
        jsonl_filepath = datagen
        upload_response = openai.File.create(
            file=open(jsonl_filepath, "rb"), purpose="fine-tune"
        )
    else:
        # assume it is a DataHolder
        dataset = datagen.dataset
        temp_filepath = f"{DATA_PATH}/tmp/{datagen.name}.jsonl"
        # make sure the tmp directory exists:
        if not os.path.exists(f"{DATA_PATH}/tmp"):
            os.mkdir(f"{DATA_PATH}/tmp")
        write_to_jsonl(
            dataset,
            temp_filepath,
            only_keys=["prompt", "completion"]
        )
        upload_response = openai.File.create(
            file=open(temp_filepath, "rb"), purpose="fine-tune"
        )
        # delete file:
        os.remove(temp_filepath)
    file_id = upload_response.id # type: ignore
    return file_id


@dataclass
class FTState:
    name : str
    description : str
    data_generator_name : str
    ft_config : FTConfig
    created : str
    model_ptr : Any 

@dataclass
class OpenAI_FTState(FTState):
    file_id : Union[None, str]
    response_id : Union[None, str]
    response_json : Dict
    result_file_id: Union[None, str]


class Finetuning(ABC):
    def __init__(
        self,
        datagen_or_path_or_name : Union[DataHolder, str],
        ft_config : FTConfig,
        name : Union[None, str] = None,
        description : str = "",
        custom_dir = None,
        skip_exists_check = False,
        skip_save = False
    ):
        if name == None:
            name = "unnamed_fintune_" + timestr()
        
        self.finetunes_path = Finetuning.get_path(custom_dir)
        self.custom_dir = custom_dir
        
        # make the finetunes file if it doesn't exist:
        if not os.path.exists(self.finetunes_path):
            with open(self.finetunes_path, "w") as f:
                json.dump({}, f)
        
        if not skip_exists_check:
            if Finetuning.name_exists(name, custom_dir):
                raise Exception(
                    f"Finetuning with name {name} already exists{f' in custom directory {custom_dir}' if custom_dir is not None else ''}. Aborted creation of new finetuning."
                )
        datagen_name = ""
        if isinstance(datagen_or_path_or_name, str):
            path_or_name = datagen_or_path_or_name
            if path_or_name[-6:] != ".jsonl":
                # assume it's a name
                datagen_name = path_or_name
                assert DataHolder.name_exists(datagen_name, custom_dir=custom_dir), f"Assertion failure: Finetuning.__init__ parsed '{datagen_name}' as a DataHolder name, but no DataHolder with name {datagen_name} exists."
            else:
                # assume it's a path to a .jsonl file
                datagen = DataHolder(
                    path_or_name,
                    name=path_or_name.split("/")[-1][:-6]
                )
                datagen.save(custom_dir=custom_dir)
        else:
            # it's a datagen object
            datagen_name = datagen_or_path_or_name.name
        # At this point, datagen_name links to an existing and saved DataHolder
        assert DataHolder.name_exists(datagen_name, custom_dir=custom_dir), f"Finetuning.__init__ failed to ensure DataHolder called {datagen_name} exists."
        
        self.state : FTState = FTState(
            name = name,
            description = description,
            data_generator_name = datagen_name,
            ft_config = ft_config,
            model_ptr = None,
            created = timestr()
        )
        
        if not skip_save:
            self.save()
        
        assert isinstance(self.state.ft_config, FTConfig), f"Instead of FTConfig, self.state.ft_config is {type(self.state.ft_config).__name__}"
        assert isinstance(self.state, FTState), f"Instead of FTState, self.state is  {type(self.state).__name__}"
    
    @staticmethod
    def get_path(dir = None) -> str:
        path = FINETUNES_PATH
        if dir is not None:
            path = f"{dir}/{FINETUNES_PATH}"
        return path
    
    @staticmethod
    def name_exists(name : str, custom_dir = None) -> bool:
        finetunes_path = Finetuning.get_path(custom_dir)
        with open(finetunes_path, "r") as f:
            finetunes = json.load(f)
        if name in finetunes.keys():
            return True
        return False
    
    @staticmethod
    def load(
        name : str,
        constructor,
        ftstate_constructor = None,
        ftconfig_constructor = None,
        custom_dir = None
    ):
        """
        Child classes should override load, passing in `constructor`
        and `ftstate_constructor` and `ftconfig_constructor` 
        (the latter two are only necessary if they use more than just
        the base FTState / base FTConfig configs).
        """
        finetunes_path = Finetuning.get_path(dir=custom_dir)
        with open(finetunes_path, "r") as f:
            finetunes = json.load(f)
        if name not in finetunes.keys():
            raise Exception(f"Finetuning with name {name} not found.")
        finetuning_state = finetunes[name]
        if ftstate_constructor is None:
            ftstate_constructor = FTState
        if ftconfig_constructor is None:
            ftconfig_constructor = FTConfig
        ft_config = ftconfig_constructor(**finetuning_state["ft_config"])
        finetuning_state["ft_config"] = ft_config
        ft_state = ftstate_constructor(**finetuning_state)
        finetuning = constructor(
            finetuning_state["data_generator_name"],
            ft_config,
            finetuning_state["name"],
            finetuning_state["description"],
            custom_dir=custom_dir,
            skip_exists_check=True,
            skip_save=True
        ) # type: ignore
        finetuning.state = ft_state
        # The new finetune already got saved in the constructor
        # but since then we overwrote properties, which means that
        # without saving again it will lose the properties:
        # finetuning.save() # <--- no longer necessary because skip_save=True
        return finetuning
    
    @staticmethod
    def edit_finetune_file(fn, custom_dir = None):
        finetunes_path = Finetuning.get_path(dir=custom_dir)
        with open(finetunes_path, "r") as f:
            finetunes = json.load(f)
        copy_file(finetunes_path, finetunes_path + ".bak")
        # Maximum paranoia backup:
        # (every state the finetunes file has ever been in is saved)
        history_file = finetunes_path + "-history.jsonl"
        if not os.path.exists(history_file):
            with open(history_file, "w") as f:
                pass
        with jsonlines.open(history_file, "a") as writer:
            writer.write(timestr()) # type: ignore
            writer.write(finetunes) # type: ignore
        new_finetunes = fn(finetunes)
        with open(finetunes_path, "w") as f:
            json.dump(new_finetunes, f, indent=4)
    
    @staticmethod
    def delete(name : str, custom_dir = None):
        Finetuning.edit_finetune_file(
            lambda finetunes : {
                k: v for k, v in finetunes.items() if k != name
            },
            custom_dir = custom_dir
        )
    
    def save(self):
        #json_for_state = json.dumps(dataclass_to_dict(self.state))
        def save_self(finetunes):
            finetunes[self.state.name] = dataclass_to_dict(self.state)
            return finetunes
        Finetuning.edit_finetune_file(save_self, custom_dir=self.custom_dir)
    
    @abstractmethod
    def start(self):
        pass
    
    @abstractmethod
    def check(self):
        pass
    
    @abstractmethod
    def is_done(self):
        pass



class OpenAI_Finetuning(Finetuning):
    def __init__(
        self,
        datagen_or_path_or_name : Union[DataHolder, str],
        ft_config : OpenAI_FTConfig,
        name,
        description = "",
        custom_dir = None,
        skip_exists_check = False,
        skip_save = False
    ):
        super().__init__(
            datagen_or_path_or_name,
            ft_config,
            name,
            description,
            custom_dir = custom_dir,
            skip_exists_check = skip_exists_check,
            skip_save = True # <--- we'll save it at the end of __init__
        )
        
        self.state : OpenAI_FTState = OpenAI_FTState(
            #name = name,
            #description = description,
            #data_generator_name = datagen_name,
            #ft_config = ft_config,
            file_id = None,
            response_id = None,
            response_json = {},
            #model_ptr = None,
            result_file_id = None,
            #created = timestr(),
            **self.state.__dict__
        )
        
        self.custom_dir = custom_dir
        
        if not skip_save:
            self.save()
    
    def start(self):
        self.state.file_id = openai_finetune_file_upload(
            DataHolder.load(
                self.state.data_generator_name,
                dir=self.custom_dir
            )
        )
        print(f"Uploaded file {self.state.file_id}.")
        assert self.state.file_id != None, "File ID cannot be none."
        response = openai.FineTune.create(
            training_file=self.state.file_id,
            **dict_without_nones(self.state.ft_config.__dict__)
        )
        # OpenAI FT API documentation sucks, so here's an example:
        """
        {'object': 'fine-tune',
        'id': 'ft-bdN4jjCZTb1wv02KhkVGYF67',
        'hyperparams': {'n_epochs': 4,
        'batch_size': None,
        'prompt_loss_weight': 0.01,
        'learning_rate_multiplier': None},
        'organization_id': 'org-e9eNgnHQJbr7PCGwAv88ygUA',
        'model': 'curie',
        'training_files': [{'object': 'file',
            'id': 'file-0F22b3inQnhkbZLJ1kfTsrxi',
            'purpose': 'fine-tune',
            'filename': 'file',
            'bytes': 16932,
            'created_at': 1690750242,
            'status': 'processed',
            'status_details': None}],
        'validation_files': [],
        'result_files': [],
        'created_at': 1690750299,
        'updated_at': 1690750299,
        'status': 'pending',
        'fine_tuned_model': None,
        'events': [{'object': 'fine-tune-event',
        'level': 'info',
        'message': 'Created fine-tune: ft-bdN4jjCZTb1wv02KhkVGYF67',
        'created_at': 1690750299}]}
        """
        self.state.response_id = response["id"] # type: ignore
        self.state.response_json = json.loads(json.dumps(response))
        self.save()
    
    def check(self):
        # OpenAI FT API documentation sucks, so here's an example:
        """
        <OpenAIObject list at 0x7fa51356c110> JSON: {
        "object": "list",
        "data": [
            {
            "object": "fine-tune-event",
            "level": "info",
            "message": "Created fine-tune: ft-bdN4jjCZTb1wv02KhkVGYF67",
            "created_at": 1690750299
            },
            {
            "object": "fine-tune-event",
            "level": "info",
            "message": "Fine-tune costs $0.04",
            "created_at": 1690758138
            },
            {
            "object": "fine-tune-event",
            "level": "info",
            "message": "Fine-tune is in the queue. Queue number: 1",
            "created_at": 1690758315
            },
            {
            "object": "fine-tune-event",
            "level": "info",
            "message": "Fine-tune is in the queue. Queue number: 0",
            "created_at": 1690758324
            },
            {
            "object": "fine-tune-event",
            "level": "info",
            "message": "Fine-tune started",
            "created_at": 1690758332
            },
            {
            "object": "fine-tune-event",
            "level": "info",
            "message": "Completed epoch 1/4",
            "created_at": 1690758414
            },
            {
            "object": "fine-tune-event",
            "level": "info",
            "message": "Completed epoch 2/4",
            "created_at": 1690758436
            },
            {
            "object": "fine-tune-event",
            "level": "info",
            "message": "Completed epoch 3/4",
            "created_at": 1690758458
            },
            {
            "object": "fine-tune-event",
            "level": "info",
            "message": "Completed epoch 4/4",
            "created_at": 1690758480
            },
            {
            "object": "fine-tune-event",
            "level": "info",
            "message": "Uploaded model: curie:ft-dcevals-kokotajlo-2023-07-30-23-08-14",
            "created_at": 1690758495
            },
            {
            "object": "fine-tune-event",
            "level": "info",
            "message": "Uploaded result file: file-w61OA9x6iZzZyKUSBl2gjJKy",
            "created_at": 1690758495
            },
            {
            "object": "fine-tune-event",
            "level": "info",
            "message": "Fine-tune succeeded",
            "created_at": 1690758496
            }
        ]
        }
        """
        fine_tune_events_response = openai.FineTune.list_events(id=self.state.response_id) # type:ignore
        events = fine_tune_events_response["data"] # type: ignore
        for event in events:
            if event["message"].split(":")[0] == "Uploaded model":
                self.state.model_ptr = event["message"].split(": ")[1]
            if event["message"].split(":")[0] == "Uploaded result file":
                self.state.result_file_id = event["message"].split(": ")[1]
        self.save()
        return fine_tune_events_response
    
    @staticmethod
    def load(name, custom_dir = None):
        return Finetuning.load(
            name,
            constructor=OpenAI_Finetuning,
            ftstate_constructor=OpenAI_FTState,
            ftconfig_constructor=OpenAI_FTConfig,
            custom_dir=custom_dir
        )
    
    def is_done(self):
        self.check()
        return self.state.model_ptr is not None
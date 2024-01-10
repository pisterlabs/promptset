import openai
#Make the persona of an electric expert who knows about electricity and all security measures
from lollms.paths import LollmsPaths
from lollms.main_config import LOLLMSConfig
from pathlib import Path
import yaml
custom_global_paths_cfg_path= Path(__file__).parent.parent.parent/"global_paths_cfg.yaml"
custom_default_cfg_path= Path(__file__).parent.parent.parent/"configs/config.yaml"
lollms_paths:LollmsPaths = LollmsPaths.find_paths(force_local=True, custom_default_cfg_path=custom_default_cfg_path, custom_global_paths_cfg_path=custom_global_paths_cfg_path)
config = LOLLMSConfig.autoload(lollms_paths)
open_ai_cfg = LOLLMSConfig(lollms_paths.personal_configuration_path/"bindings"/"open_ai"/"config.yaml")
openai_key = open_ai_cfg.openai_key
openai.api_key = openai_key
models = []
for model in openai.models.list():
    print(model)
    if "gpt" in model.id:
        md = {
            "category": "generic",
            "datasets": "unknown",
            "icon": '/bindings/open_ai/logo.png' if "3.5" in model.id else '/bindings/open_ai/logo2.png',
            "last_commit_time": model.created,
            "license": "commercial",
            "model_creator": "openai",
            "model_creator_link": "https://openai.com",
            "name": model.id,
            "quantizer": None,
            "rank": 1.0,
            "type": "api",
            "variants":[
                {
                    "name":model.id,
                    "size":999999999999
                }
            ]
        }
        models.append(md)
        with open(Path(__file__).parent/"openai.yaml", 'w') as f:
            yaml.dump(models, f)        



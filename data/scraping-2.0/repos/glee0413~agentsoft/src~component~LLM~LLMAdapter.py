import langchain

class LLMAdapter:
    _registered_models = {}  # 用于存储注册的模型

    def __init__(self, config_path):
        self.config_path = config_path
        self.selected_model = None
        self.load_config()
        self.register_models()

    def load_config(self,config):
        # 从配置文件加载配置，选择要使用的模型
        # 在这里读取配置的代码...
        pass

    def register_models(self):
        # 注册所有可用的LLM模型
        for name, cls in self._registered_models.items():
            # 利用langchain的注册功能自动注册模型
            langchain.register(cls, name)

    def select_model(self, model_name):
        # 选择要使用的模型
        # 在这里选择模型的代码...
        pass

    def generate_text(self, input_text):
        # 通过选定的模型生成文本
        # 在这里调用选定模型的代码...
        pass
    
    # 装饰器用于注册模型
    @classmethod
    def register_model(cls, name):
        def decorator(model_cls):
            cls._registered_models[name] = model_cls
            return model_cls
        return decorator

def main():
    adapter = LLMAdapter("config_file.yaml")
    adapter.select_model("openai-gpt3")
    output_text = adapter.generate_text("你的输入文本")
    print(output_text)

if __name__ == "__main__":
    main()
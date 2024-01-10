from nft_tools import NFT_info
from langchain.agents.tools import Tool

class_name = 'NFT_info'
device = 'cpu'
models = {}
models[class_name] = globals()[class_name](device=device)
print(models)
tools = []
for instance in models.values():
    # 获取类中所有的属性和方法名称
    for e in dir(instance):
        if e.startswith('get_nft'):
            # getattr(obj, name)时，Python会尝试获取obj对象的name属性或方法。
            func = getattr(instance, e)
            tools.append(Tool(name=func.name, description=func.description, func=func))
print(tools)
# print(get_nft_by_contract(inputs='0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d,4495'))
# a = NFT_info('cpu')

# print(a.get_nft_by_contract(inputs='0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d,4495'))
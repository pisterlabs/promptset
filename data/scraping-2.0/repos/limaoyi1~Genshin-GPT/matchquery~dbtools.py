# Import things that are needed generically
from langchain import LLMMathChain, WikipediaAPIWrapper
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool

from matchquery.dbmatch import CharacterWrapper, AchievementsWrapper, AdventurerRankWrapper, AnimalsWrapper, \
    ArtifactsWrapper, ConstellationsWrapper, DomainsWrapper, EnemiesWrapper, FoodWrapper, GeographiesWrapper, \
    MaterialsWrapper, NameCardsWrapper, OutfitsWrapper, TalentsWrapper, WeaponWrapper, WindgliderWrapper, \
    AchievementgroupsWrapper
from readconfig.myconfig import MyConfig


# 根据角色查询
character_wrapper = CharacterWrapper()

# 根据成就分组查询
achievement_groups_wrapper = AchievementgroupsWrapper()

# 根据成就查询
achievements_wrapper = AchievementsWrapper()

# 根据冒险等级查询
adventurer_rank_wrapper = AdventurerRankWrapper()

# 根据动物查询
animals_wrapper = AnimalsWrapper()

# 根据圣遗物查询
artifacts_wrapper = ArtifactsWrapper()

# 根据命座查询
constellations_wrapper = ConstellationsWrapper()

# 根据秘境副本查询
domains_wrapper = DomainsWrapper()

# 根据敌人查询
enemies_wrapper = EnemiesWrapper()

# 根据食物查询
food_wrapper = FoodWrapper()

# 根据地理信息查询
geographies_wrapper = GeographiesWrapper()

# 根据材料查询
materials_wrapper = MaterialsWrapper()

# 根据名片查询
name_cards_wrapper = NameCardsWrapper()

# 根据服饰查询
outfits_wrapper = OutfitsWrapper()

# 根据角色天赋查询
talents_wrapper = TalentsWrapper()

# 根据武器查询
weapon_wrapper = WeaponWrapper()

# 根据风之翼查询
windglider_wrapper = WindgliderWrapper()

# 维基百科
wikipedia_wrapper = WikipediaAPIWrapper(lang='zh', top_k_results=1)

tools = [
    Tool.from_function(
        func=character_wrapper.run,
        name="角色信息查询",
        description="通过角色名称查询角色的基本信息,在和角色对话的时候很有用",
        # return_direct=True
    ),
    # Tool.from_function(
    #     func=achievement_groups_wrapper.run,
    #     name="成就分组查询",
    #     description="通过成就分组名称查询,需要知道成就分组时会很有用",
    #     return_direct=True
    # ),
    # Tool.from_function(
    #     func=achievements_wrapper.run,
    #     name="成就查询",
    #     description="通过成就名称查询,需要知道成就信息时会很有用",
    #     return_direct=True
    # ),
    # Tool.from_function(
    #     func=adventurer_rank_wrapper.run,
    #     name="冒险等级查询",
    #     description="通过等级获取冒险等级信息,传入1-60中的一个整数,需要知道冒险等级时会很有用",
    #     return_direct=True
    # ),
    Tool.from_function(
        func=animals_wrapper.run,
        name="动物查询",
        description="通过动物名称查询动物信息",
        return_direct=True
    ),
    Tool.from_function(
        func=artifacts_wrapper.run,
        name="圣遗物查询",
        description="通过圣遗物名称查询圣遗物信息",
        return_direct=True
    ),
    Tool.from_function(
        func=constellations_wrapper.run,
        name="命座查询",
        description="通过角色名称获取命座信息,通过角色名称查询命座信息",
        return_direct=True
    ),
    Tool.from_function(
        func=domains_wrapper.run,
        name="副本秘境查询",
        description="通过副本秘境名称获取副本秘境信息,需要知道副本秘境信息时会很有用",
        return_direct=True
    ),
    Tool.from_function(
        func=enemies_wrapper.run,
        name="敌人查询",
        description="通过敌人名称获取敌人信息,需要知道敌人信息时会很有用",
        return_direct=True
    ),
    Tool.from_function(
        func=food_wrapper.run,
        name="食物信息查询",
        description="通过食物名称获取食物信息,或者通过角色名称获取角色最喜欢的食物",
        return_direct=True
    ),
    Tool.from_function(
        func=geographies_wrapper.run,
        name="地理位置查询",
        description="通过地名进行地理位置查询,需要知道地理信息时会很有用",
        return_direct=True
    ),
    Tool.from_function(
        func=materials_wrapper.run,
        name="材料信息查询",
        description="通过材料名称获取详细信息,包含游戏中所有的素材,需要知道材料信息时会很有用",
        return_direct=True
    ),
    # Tool.from_function(
    #     func=name_cards_wrapper.run,
    #     name="名片信息查询",
    #     description="通过游戏名片查询详细信息,需要知道名片信息时会很有用",
    #     return_direct=True
    # ),
    Tool.from_function(
        func=outfits_wrapper.run,
        name="服饰信息查询",
        description="通过服饰名称或者角色名称获取角色衣服信息",
        return_direct=True
    ),
    Tool.from_function(
        func=talents_wrapper.run,
        name="角色战斗天赋查询",
        description="通过角色名称获取角色战斗天赋信息",
        return_direct=True
    ),
    Tool.from_function(
        func=weapon_wrapper.run,
        name="武器信息查询",
        description="通过武器名称获取武器详细信息",
        return_direct=True
    ),
    # Tool.from_function(
    #     func=windglider_wrapper.run,
    #     name="风之翼查询",
    #     description="通过风之翼名称获取风之翼详细信息",
    #     return_direct=True
    # ),
    Tool.from_function(
        func=wikipedia_wrapper.run,
        name="维基百科查询",
        description="作为其他工具的补充,在使用其他工具没有查询到结果时作为补充使用",
        return_direct=True
    )

]

if __name__ == '__main__':
    config = MyConfig("../")

    llm = ChatOpenAI(temperature=0, openai_api_key=config.OPENAI_API_KEY,
                     openai_api_base=config.OPENAI_BASE_URL)

    llm_math_chain = LLMMathChain(llm=llm, verbose=True)

    agent = initialize_agent(
        tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True  # 自我询问模式似乎更好用
    )

    agent.run(
        "甘雨的饮食习惯和文化背景是什么样的？"
    )

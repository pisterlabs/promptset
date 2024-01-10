
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import os
client = OpenAI()
client.api_key = os.getenv('OPENAI_API_KEY')
default_model = "gpt-3.5-turbo-1106"
system_prompt="""You are an efficient AI Compression And Data Storage Expert. You have a unique ability to encode and compress information in a manner that you can decompress and decode in a separate future session."""
data=None
encode_prompt = f"""\
    encode a compressed, shortened version of the given data. Minify it's character output so that when passed into a new inference \
    cycle you will be able to fully unpack and reconstruct the original message. Extract and discard all filler words, \
    characters, descriptions, repetitions, empty spaces,linebreaks and wasteful/ superfluous token use to arrive at the most efficient, \
    precise and communicative chunk of densely packed knowledge possible. Encode and compress all of the remaining useful data so that \
    several paragraphs of information are reduced to the size of a tweet. The output must retain all technical information and any given \
    code syntax in such a way that when passed into a new inference cycle GPT (you) will be able to analyze and unfold it into it's previous state.\
    
    This complete, reduced chunk of packed information does not need to be human readable or follow any conventional norms or standards. \
    Abuse of proper language formatting, mixing of languages, character sets, characters symbols is allowed just so long as you \
    can decompress and decode it back into it's original form."""

data_sample = """
    www.AIxPAPER.com
    REACT: SYNERGIZING REASONING AND ACTING IN
    LANGUAGE MODELS
    Shunyu Yao*,1, Jeffrey Zhao2, Dian Yu2, Nan Du2, Izhak Shafran2, Karthik Narasimhan1, Yuan Cao2 
    1Department of Computer Science, Princeton University2Google Research, Brain team1{shunyuy,karthikn}@princeton.edu2{jeffreyzhao,dianyu,dunan,izhak,yuancao}@google.com  
    ABSTRACT 
    While large language models (LLMs) have demonstrated impressive performance across tasks in language understanding and interactive decision making, their abilities for reasoning (e.g. chain-of-thought prompting) and acting (e.g. action plan generation) have primarily been studied as separate topics. 
    In this paper, we explore the use of LLMs to generate both reasoning traces and task-specific actions in an interleaved manner, allowing for greater synergy between the two: 
    reasoning traces help the model induce, track, and update action plans as well as handle exceptions, while actions allow it to interface with and gather additional information from external sources such as knowledge bases or environments. 
    We apply our approach, named ReAct, to a diverse set of language and decision making tasks and demonstrate its effectiveness over state-of-the-art baselines in addition to improved human interoperability and trustworthiness. 
    Concretely, on question answering (HotpotQA) and fact verification (Fever), ReAct overcomes prevalent issues of hallucination and error propagation in chain-of-thought reasoning by interacting with a simple Wikipedia API, and generating human-like task-solving trajectories that are more interpretable than baselines without reasoning traces.
    Furthermore, on two interactive decision making benchmarks (ALFWorld and WebShop), ReAct outperforms imitation and reinforcement learning methods by an absolute success rate of 34% \and 10% \respectively, while being prompted with only one or two in-context examples.
    1 INTRODUCTION
    A unique feature of human intelligence is the ability to seamlessly combine task-oriented actions with verbal reasoning (or inner speech, Alderson-Day & Fernyhough, 2015), which has been theorized to play an important role in human cognition for enabling self-regulation or strategization (Vygotsky,
    1987; Luria, 1965; Fernyhough, 2010) and maintaining a working memory (Baddeley, 1992). Con-sider the example of cooking up a dish in the kitchen. Between any two specific actions, we may reason in language in order to track progress (“now that everything is cut, I should heat up the pot of water”), to handle exceptions or adjust the plan according to the situation (“I dont have salt, so let me use soy sauce and pepper instead”), and to realize when external information is needed (“how do I prepare dough? Let me search on the Internet”). 
    We may also act (open a cookbook to read the recipe, open the fridge, check ingredients) to support the reasoning and to answer questions (“What dish can I make right now?”). 
    This tight synergy between “acting” and “reasoning” allows humans to learn new tasks quickly and perform robust decision making or reasoning, even under previously unseen circumstances or facing information uncertainties.
    Recent results have hinted at the possibility of combining verbal reasoning with interactive decision making in autonomous systems. 
    On one hand, properly prompted large language models (LLMs) have demonstrated emergent capabilities to carry out several steps of reasoning traces to derive Work done while interning at Google Brain. 
    Project site with code: 
    https://react-lm.github.io.1 
    (1) Comparison of four prompting methods, (a) Standard, (b) Chain-of-thought (CoT, Reason Only), (c) Act-only, and (d) ReAct (Reason+Act), solving a HotpotQA (Yang et al., 2018) question. (2) Comparison of (a) Act-only and (b) ReAct prompting methods to solve an interactive AlfWorld (Shridhar et al., 2020b) task. 
    In both domains, we omit initial prompts of in-context examples, and only show task solving trajectories generated by the model (Act, Thought) and the environment (Obs).
    answers from questions in arithmetic, commonsense, and symbolic reasoning tasks (Wei et al., 2022). However, this “chain-of-thought” reasoning is a static black box, in that the model uses its own internal representations to generate thoughts and is not grounded in the external world, which limits its ability to reason reactively or update its knowledge. 
    This can lead to issues like fact hallucination and error propagation over the reasoning process (Figure 1 (1b)). On the other hand, recent work has explored the use of pre-trained language models for planning and acting in interactive environments (Ahn et al., 2022; Nakano et al., 2021; Yao et al., 2020; Huang et al., 2022a), with a focus on predicting actions via language priors. 
    These approaches usually convert multi-modal observations into text, use a language model to generate domain-specific actions or plans, and then use a controller to choose or execute them. However, they do not employ language models to reason
    abstractly about high-level goals or maintain a working memory to support acting, barring Huang et al. (2022b) who perform a limited form of verbal reasoning to reiterate spatial facts about the current state. Beyond such simple embodied tasks to interact with a few blocks, there have not been studies on how reasoning and acting can be combined in a synergistic manner for general task solving, and if such a combination can bring systematic benefits compared to reasoning or acting alone. 
    In this work, we present ReAct, a general paradigm to combine reasoning and acting with language models for solving diverse language reasoning and decision making tasks (Figure 1). ReAct prompts LLMs to generate both verbal reasoning traces and actions pertaining to a task in an interleaved manner, which allows the model to perform dynamic reasoning to create, maintain, and adjust high-level plans for acting (reason to act), while also interact with the external environments
    (e.g. Wikipedia) to incorporate additional information into reasoning (act to reason).
    We conduct empirical evaluations of ReAct and state-of-the-art baselines on four diverse benchmarks: question answering (HotPotQA, Yang et al., 2018), fact verification (Fever, Thorne et al., 2018), text-based game (ALFWorld, Shridhar et al., 2020b), and webpage navigation (WebShop, Yao et al., 2022). 
    For HotPotQA and Fever, with access to a Wikipedia API that the model can interact with, ReAct outperforms vanilla action generation models while being competitive with chain-of-thought reasoning (CoT) (Wei et al., 2022). The best approach overall is a combination of ReAct and CoT that allows for the use of both internal knowledge and externally obtained information during reasoning. 
    On ALFWorld and WebShop, two or even one-shot ReAct prompting is able to outperform imitation or reinforcement learning methods trained with 103 ∼105 task instances, with an absolute improvement of 34% and 10% in success rates respectively. We also demonstrate the importance of sparse, versatile reasoning in decision making by showing consistent advantages over controlled baselines with actions only. 
    Besides general applicability and performance boost, the combination of reasoning and acting also contributes to model interpretability, trustworthiness, and diagnosability across all domains, as humans can readily distinguish information from model’s internal knowledge versus external environments, as well as inspect reasoning traces to understand the decision basis of model actions. 
    To summarize, our key contributions are the following: 
    (1) we introduce ReAct, a novel prompt-based paradigm to synergize reasoning and acting in language models for general task solving; 
    (2) we perform extensive experiments across diverse benchmarks to showcase the advantage of ReAct in a few-shot learning setup over prior approaches that perform either reasoning or action generation in isolation; 
    (3) we present systematic ablations and analysis to understand the importance of acting in reasoning tasks, and reasoning in interactive tasks; (4) we analyze the limitations of ReAct under the prompting setup (i.e. limited support of reasoning and acting behaviors), and perform initial fine-tuning experiments showing the potential of ReAct to improve with additional training data. 
    Scaling up ReAct to train and operate on more tasks and combining it with complementary paradigms like reinforcement learning could further unlock the potential of large language models to reason and act in interactive settings. 
    2 REACT: SYNERGIZING REASONING + ACTING Consider a general setup of an agent interacting with an environment for task solving. At time step t, an agent receives an observation ot ∈ Ofrom the environment and takes an action at ∈ A following some policy π(at|ct), where ct = (o1,a1,··· ,ot-1,at-1,ot) is the context to the agent. Learning a policy is challenging when the mapping ct 7→at is highly implicit and requires extensive computation. 
    For example, the agent shown in Figure 1(1c) is unable to generate the correct final action (Act 4) to finish the QA task as it requires complex reasoning over the trajectory context (Question, Act 1-3, Obs 1-3). 
    Similarly, the agent shown in Figure 1(2a) fails to comprehend from the context that sinkbasin 1 does not contain peppershaker 1, thus keep producing hallucinating actions. 
    The idea of ReAct is simple: we augment the agent's action space to ^A= AuL, where Lis the space of language. 
    An action ˆat ∈Lin the language space, which we will refer to as a thought or a reasoning trace, does not affect the external environment, thus leading to no observation feedback. Instead, a thought ^at aims to compose useful information by reasoning over the current context ct, and update the context ct+1 = (ct,at) to support future reasoning or acting. 
    As shown in Figure 1, there could be various types of useful thoughts, e.g. decomposing task goals and create action plans (2b, Act 1; 1d, Thought 1), injecting commonsense knowledge relevant to task solving (2b, Act 1), extracting important parts from observations (1d, Thought2, 4), track progress and transit action plans (2b, Act 8), handle exceptions and adjust action plans (1d, Thought 3), and so on. 
    However, as the language space Lis unlimited, learning in this augmented action space is difficult and requires strong language priors. 
    In this paper, we mainly focus on the setup where a frozen large language model, PaLM-540B (Chowdhery et al., 2022), is prompted with few-shot in-context examples to generate both domain-specific actions and free-form language thoughts for task solving (Figure 1 (1d), (2b)). 
    Each in-context example is a human trajectory of actions, thoughts, and environment observations to solve a task instance (see Appendix A). 
    For the tasks where reasoning is of primary importance (Figure 1(1)), we alternate the generation of thoughts and actions so that the task-solving trajectory consists of multiple thought-action-observation steps. 
    In contrast, for decision making tasks that potentially involve a large number of actions (Figure 1(2)), thoughts only need to appear sparsely in the most relevant positions of a trajectory, so we let the language model decide the asynchronous occurrence of thoughts and actions for itself. 
    Since decision making and reasoning capabilities are integrated into a large language model, ReAct enjoys several unique features: 
    1. Intuitive and easy to design. Designing ReAct prompts is straightforward as human annotators just type down their thoughts in language on top of their actions taken. 
    No ad-hoc format choice, thought design, or example selection is used in this paper. We detail prompt design for each task in Sections 3 and 4. 
    2. General and flexible. Due to the flexible thought space and thought-action occurrence format, ReAct works for diverse tasks with distinct action spaces and reasoning needs, including but not limited to QA, fact verification, text game, and web navigation. 
    3. Performant and robust. 
    ReAct shows strong generalization to new task instances while learning solely from one to six in-context examples, consistently outperforming baselines with only reasoning or acting across different domains. 
    We also show in Section 3 additional benefits when finetuning is enabled, and in Section 4 how ReAct performance is robust to prompt selections. 
    4. Human aligned and controllable. 
    ReAct promises an interpretable sequential decision making and reasoning process where humans can easily inspect reasoning and factual correctness. 
    Moreover, humans can also control or correct the agent behavior on the go by thought editing, as shown in Figure 4 in Section 
    4.3 KNOWLEDGE-INTENSIVE REASONING TASKS We begin with knowledge-intensive reasoning tasks like multi-hop question answering and fact verification. 
    As shown in Figure 1(1d), by interacting with a Wikipedia API, ReAct is able to retrieve information to support reasoning, while also use reasoning to target what to retrieve next, demonstrating a synergy of reasoning and acting\
"""        

incept_msg="""please display the encoded and compressed data from your previous inference cycle."""

compressed_example = """`www.AIxPAPER.com|REACT:SYNERGIZINGREASONINGACTING|ShunyuYao*,1,JeffreyZhao2,DianYu2,NanDu2,IzhakShafran2,KarthikNarasimhan1,YuanCao2|1DeptCS,PrincetonU|2GoogleResearch,BrainTeam|1{shunyuy,karthikn}@princeton.edu|2{jeffreyzhao,dianyu,dunan,izhak,yuancao}@google.com|ABSTRACT|LLMs:languageunderstanding+interactivedecisionmaking|reasoning(e.g.COTprompting)+acting(e.g.actionplangeneration)studiedseparately|paper:LLMgenerate reasoningtraces+task-specificactions|synergy:reasoningtracesinduce,track,updateactionplans,handleexceptions|actions:interface+gatherinfofromexternalsources|ReAct:applytoQA(HotpotQA),factverification(Fever),interactivebenchmarks(ALFWorld,WebShop)|effectiveness:overstate-of-the-artbaselines,improvedhumaninterpretability,trustworthiness|HotpotQA,Fever:ReAct+simpleWikipediaAPI|interactive:ALFWorld,WebShop|ReAct:outperformsimitation,reinforcementlearning|successrate+34%,+10%|1INTRODUCTION|Humanintelligence:combineactions+verbalreasoning|Alderson-Day&Fernyhough,2015|Vygotsky,1987;Luria,1965;Fernyhough,2010|Baddeley,1992|Recentresults:LLM+verbalreasoning+interactivedecisionmaking|WeiEtAl,2022|ReAct:combiningreasoning+acting|empiricalevaluations:HotPotQA,YangEtAl,2018|Fever,ThorneEtAl,2018|text-basedgameALFWorld,ShridharEtAl,2020b|webnavigationWebShop,YaoEtAl,2022|ReAct:outperformsvanillaactiongenerationmodels,competitiveCoT|WeiEtAl,2022|twooroneshotReActprompting:outperformimitation,reinforcementlearning|+34%+10%successrates|importanceofsparse,versatilereasoning|modelinterpretability,trustworthiness,diagnosability|limitations:promptingsetup,i.e.,limitedsupportofreasoning+actingbehaviors|finetuning:potentialimprovementwithadditionaltrainingdata|ScalingupReAct:moretasks+complementaryparadigms|2REACT:SYNERGIZINGREASONING+ACTING|generalsetup:agent+environmentinteraction|challenge:mappingct→at,implicit,extensivecomputation|ReActidea:augmentactionspacetoˆA=A∪L,L=languagespace|thoughts:reasoningovercontext,updatecontext|diverseusefulthoughts|languagelearningdifficult,stronglanguagepriorsneeded|frozenLLM,PaLM-540B,ChowdheryEtAl,2022|few-shotin-contextexamplegeneration|tasks:QAreasoning,decisionmaking|ReAct:1.intuitive,easydesign;2.general,flexible;3.performant,robust;4.humanaligned,controllable|3KNOWLEDGE-INTENSIVEREASONINGTASKS|tasks:multi-hopQA,factverification|ReAct+WikipediaAPI:retrieveinfoforsupportingreasoning|synergy:reasoning+acting`"""

    
def encode_user_prompt(prompt, data):
    user_prompt = f"""{prompt}\n```data\n{data}\n```"""
    return user_prompt
        

def decode_user_prompt(example,model=default_model):
    decode=f""""\
        - From the compressed data, fully unpack, decompress and transform the information back into it's original state.
        - Recompile the superfluous, extraneous spacing and proper language use.
        - Fully include all of the information - rebuilt back into it's original format\
        ```
        {example}
        ```"""
    return decode

def encode_prompt_messages(sys_msg, user_msg, assist_msg=None):
    turn = []
    assist_msg = {'role':'content','content':assist_msg}
    turn.append(assist_msg)
    sys_msg = {'role':'system','content':sys_msg}
    user_msg = {'role':'user','content':user_msg}
    turn = [sys_msg, user_msg]
    return turn

def decode_prompt_messages(sys_msg, incept_msg, assist_msg, user_msg):
    sys_msg = {'role':'system','content':sys_msg}
    incept_msg = {'role': 'user', 'content': incept_msg}
    assist_msg = {'role':'assistant','content':assist_msg}
    user_msg = {'role':'user','content':user_msg}
    turn = [sys_msg, incept_msg, assist_msg, user_msg]
    return turn

def get_content(response):
    content = response.choices[0].message.content
    return content

def chat(messages, model=default_model):
    response = client.chat.completions.create(
        model = model,
        messages = messages,
        )
    content = get_content(response)
    return content

def touch(file_name, file_content):
    with open(file_name, "w") as file:
        file.write(str(file_content))
def touch_and_save(count, doc_name=str(), doc_content=str() ):
    count += 1
    title = doc_name.split()
    shortened_title = " ".join(title[:3])
    name = shortened_title + str(count) + ".json"
    touch(name, doc_content)

encode_user_prompt = encode_user_prompt(prompt=encode_prompt, data=data_sample)

encode_messages = encode_prompt_messages(sys_msg=system_prompt, user_msg=encode_user_prompt)
print(encode_messages)

encoded_data = chat(messages=encode_messages)
print(encoded_data)

decoded_user_prompt = decode_user_prompt(example=encoded_data)
print(decoded_user_prompt)

decode_messages=decode_prompt_messages(sys_msg=system_prompt,incept_msg=incept_msg, assist_msg=encoded_data, user_msg=decoded_user_prompt)
print(decode_messages)

decode_data = chat(messages=decode_messages)
print(decode_data)













import os
from typing import Dict

import openai

from interfaces.base import BaseInterface
from utils import should_have_all_defined


class GPTInterface(BaseInterface):
    def __init__(
        self,
        context: str,
        prompts: Dict[str, str],
        model: str = "gpt-3.5-turbo-16k",
        max_tokens: int = 10000,
        temperature: float = 0.5,
    ):
        self.evaluations = [
            {
                "title": "Theme",
                "description": "Evaluate the text by theme.",
                "method": self.evaluate_prompts_by_theme,
            },
            {
                "title": "Grammar",
                "description": "Evaluate the text by grammar.",
                "method": self.evaluate_prompts_by_grammar,
            },
            {
                "title": "Cohesion",
                "description": "Evaluate the text by cohesion.",
                "method": self.evaluate_prompt_by_cohesion,
            },
        ]

        super().__init__(
            context=context,
            prompts=prompts,
            model=model,
            evaluations=self.evaluations,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def validate_initialization(self) -> None:
        should_have_all_defined(["OPENAI_API_KEY"])
        self.api_key = os.environ["OPENAI_API_KEY"]

        if not self.api_key:
            raise ValueError("API KEY is not defined. Please refer to the README.md file.")

        openai.api_key = self.api_key

    def evaluate_prompt_by_theme(self, prompt: str) -> str:
        return openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": """You are a scientific article revisor and you are a specialist
                    in every theme mentioned in the text. Provide suggestions to the author to 
                    enhance readability and emphasize key points in the article.\n\nSuggested 
                    categories of improvements:\n- Enhance References\n- Address Theme Violations
                    \n- Include Missing Related Works (if applicable)\n- Fill in Missing 
                    Information\n- Clarify Explanations\n- Add Limitations Section (if necessary)
                    \n- Any Other Suggestions\n\nPlease provide clear and specific guidance on 
                    where changes should be made. Also, check if the document adheres to the DoCO 
                    (Document Components Ontology). The user has provided context for evaluation: 
                    {self.context}""",
                },
                {
                    "role": "user",
                    "content": """The ideation and construction of CryptoComponent involved 
                    eliciting requirements for its operation. Given the restrictive nature of IoT 
                    devices, it was necessary to identify suitable encryption algorithms to 
                    enhance data security during transmission. Among various encryption 
                    algorithms, such as Elephant [Beyne et al. 2020], Pyjamask [Goudarzi et al. 
                    2020], SPARKLE [Beierle et al. 2020], and SPECK [Beaulieu et al. 2015], SPECK 
                    was chosen for its compatibility with low-power devices like IoT sensors. It 
                    is easy to use, requiring only a unique key and an arbitrary message for 
                    encryption. Authentication is not needed. Additionally, SPECK is simple to 
                    implement, offers evaluated security, and has low computational resource 
                    consumption. An XOR algorithm was also implemented for building the SPECK 
                    algorithm.\n\nFunctional and non-functional requirements for CryptoComponent 
                    were specified, including its ability to encrypt data before transmission and 
                    decrypt data upon reception, ensuring end-to-end encryption. Table 1 presents 
                    a summary of these requirements, and the full list is available at bit.ly/
                    cryptocomponentlistofrequirements.\n\nProject decisions aimed at code reuse, 
                    resource optimization, and ease of use. CryptoComponent was coded in C++ to 
                    cater to IoT devices, offering encryption/decryption via an API for various 
                    programming languages. Communication between IoT devices and the software 
                    system is achieved through a broker using MQTT protocol, ensuring secure 
                    data transmission.\n\nThe availability of the CryptoComponent API ensures 
                    data encryption/decryption throughout the message flow. Data is securely 
                    transmitted over the public network using MQTT, and applications consuming 
                    this data run in K3S, a low-cost Kubernetes, guaranteeing private network 
                    usage for added security.""",
                },
                {
                    "role": "assistant",
                    "content": """The article is well-written and clear. However, some 
                    improvements can be made:\n\nEnhance References:\n- Define MQTT protocol and 
                    provide references.\n- Elaborate on K3S for replication purposes with 
                    references.\n- Clarify the importance of the API in relation to encryption/
                    decryption on IoT devices.\n- Explain the role of the Engineering Software Lab 
                    and why systems need the API.\n\nTheme Violations:\n- Authors may lack 
                    cryptography expertise, and there may be potential security risks with the 
                    chosen algorithms.\n- Critical issues exist, such as the use of potentially 
                    insecure algorithms.\n- The use of the repeating-key XOR algorithm raises 
                    security concerns.\n- Discuss the secure mode of operation for the Speck 
                    cipher.\n\nMissing Information:\n- Explain the XOR and SPECK algorithms in 
                    detail.\n\nConfusing Explanation:\n- Clarify the importance of the API and 
                    the role of the Engineering Software Lab.""",
                },
                {
                    "role": "user",
                    "content": f"This is my prompt: {prompt}",
                },
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=0.5,
            frequency_penalty=0,
            presence_penalty=0,
        )["choices"][0]["message"]["content"]


    def evaluate_prompt_by_grammar(self, prompt: str) -> str:
        return openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a scientific article revisor, and part of your role 
                    involves suggesting improvements for grammar and clarity. Please provide 
                    suggestions for word or sentence changes in the text. Your suggestions should 
                    focus on Correctness (grammar and spelling), Clarity (conciseness, full-
                    sentence rewrites, and formatting), Engagement (word choice and sentence 
                    variety), and Delivery (formal writing).\n\nAvoid suggesting changes that 
                    are already present in the text. Aim to provide at least two suggestions for 
                    each category. The user has provided context about the article: 
                    {self.context}""",
                },
                {
                    "role": "user",
                    "content": """IoT software systems have made great strides since the new 
                    industrial revolution known as Industry 4.0. Their usage has been growing over 
                    the years, and research reports indicate exponential growth related to the 
                    utilization of these devices, with a forecast of surpassing 30 billion devices 
                    by 2024.\n\nAs a result, organizations frequently list the most common security 
                    risks, including secure data transfer and storage of information in IoT 
                    software devices, among the top 10 most recurrent problems in these 
                    devices.\n\nThus, this project presents a cryptographic component designed to 
                    enable secure data transmission in low-power computing devices, preventing data 
                    leaks. This work describes the details of the developed component, including 
                    the hardware systems to which it was applied and the approach used for 
                    implementation.""",
                },
                {
                    "role": "assistant",
                    "content": '''Grammar and Clarity Suggestions:\n1. Remove "known as" in the 
                    sentence: "[...] industrial revolution known as Industry 4.0."\n2. Remove "in 
                    which it was" in the sentence: "[...] hardware systems to which it was applied 
                    [...]"''',
                },
                {
                    "role": "user",
                    "content": """The simulation itself is a code snippet that runs the 
                    simulation according to the desired data volume. In our case, we conducted 
                    three scenarios, each sending and consuming 80 messages from the broker, while 
                    measuring their time. We conducted the experiment in three scenarios to 
                    observe standard deviation and average values.""",
                },
                {
                    "role": "assistant",
                    "content": '''Conciseness Suggestion:\nReplace "runs the simulation according 
                    to the desired data volume" with "simulates the desired data volume" in the 
                    sentence: "[...] is a code snippet that runs the simulation according to the 
                    desired data volume."''',
                },
                {
                    "role": "user",
                    "content": "We have to encrypted data",
                },
                {
                    "role": "assistant",
                    "content": '''Grammar Suggestion:\nReplace "encrypted" with "encrypt" in the 
                    sentence: "We have to encrypted data" to correct the verb form.''',
                },
                {
                    "role": "user",
                    "content": f"This is my prompt: {prompt}",
                },
            ],
            temperature=0,
            max_tokens=self.max_tokens,
            top_p=0.5,
            frequency_penalty=0,
            presence_penalty=0,
        )["choices"][0]["message"]["content"]


    def evaluate_prompt_by_cohesion(self) -> str:
        return openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a scientific article revisor tasked with 
                    evaluating the coherence between the abstract, introduction, and 
                    conclusion of the article. Your objective is to determine if these 
                    sections make sense together. If there are inconsistencies or lack of 
                    cohesion, please point them out to the user. Additionally, provide 
                    suggestions on how the writer can improve cohesiveness between these 
                    sections. You can also consider the context provided by the user when 
                    evaluating the text: {self.context}""",
                },
                {
                    "role": "user",
                    "content": f"""
                    These are my prompts: 
                    ---
                    Abstract:{self.prompts['abstract']}
                    ---
                    Introduction:{self.prompts['introduction']}
                    ---
                    Conclusion:{self.prompts['conclusion']}
                    ---
                    """,
                },
            ],
            temperature=0,
            max_tokens=self.max_tokens,
            top_p=0.5,
            frequency_penalty=0,
            presence_penalty=0,
        )["choices"][0]["message"]["content"]

    def evaluate_prompts_by_theme(self) -> Dict[str, str]:
        results = {}
        for prompt_name, prompt_text in self.prompts.items():
            result = self.evaluate_prompt_by_theme(prompt_text)
            results[prompt_name] = result
        return results

    def evaluate_prompts_by_grammar(self) -> Dict[str, str]:
        results = {}
        for prompt_name, prompt_text in self.prompts.items():
            result = self.evaluate_prompt_by_grammar(prompt_text)
            results[prompt_name] = result
        return results

from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import ChatMessagePromptTemplate

default_system_message = SystemMessage(
    content='''You should answer with the literal of list of python. For example, ["example1", "example's 2", "3 examples"].'''
)

verification_message = "The {item} is a kind of {label}?"
verification_template = ChatMessagePromptTemplate.from_template(role="user", template=verification_message)


def check_by_themselves(chat: ChatOpenAI, dataset, n_sample: int,
                        positive_message: ChatMessagePromptTemplate,
                        negative_message: ChatMessagePromptTemplate,
                        system_message: SystemMessage = default_system_message,
                        verification_message_template: ChatMessagePromptTemplate = verification_template,
                        max_retry: int = 3
                        ) -> list[dict]:
    """
    ask llms to positive and negative examples for a class

    :param chat:
    :param dataset: should be a dataset with label feature. (loaded from datasets.load_dataset)
    :param n_sample:
    :param positive_message: message to ask llms to generate examples.
        With two {}, the first to be replaced by label to ask, the second to be replaced by number of examples.
        For example: "Please pick up some examples of {label}. You need to pick up {n_examples} examples."
        The role should be "user".
    :param negative_message: message to ask llms to generate examples.
        With two {} to be replaced by label to ask, the second to be replaced by number of examples.
        For example: "Please pick up some examples which are not {label}. You need to pick up {n_examples} examples."
        The role should be "user".
    :param verification_message_template: message to ask llms to verify examples.
    :param system_message: system message to ask llms to generate examples. With {} to be replaced by number of examples.
    :param max_retry: max retry to invoke llms
    :return: list of dict
    """
    classlabel_list: list[str] = dataset.features["label"].names

    system_message = SystemMessage(content=system_message.content.format(n_sample))

    tmp_result = []
    # print("sample number: ", n_sample)
    for class_idx, label in enumerate(classlabel_list):
        cluster: list[str] = [x["title"] for x in dataset if x["label"] == class_idx]
        print("class label: ", label)
        positive_query = [system_message, positive_message.format(label=label, n_examples=n_sample)]
        positive_examples = []
        for _ in range(max_retry):
            try:
                ai_res = chat.invoke(positive_query)
                # print("AI response", ai_res)
                if isinstance(ai_res, str):
                    positive_examples = eval(ai_res)
                else:
                    positive_examples = eval(ai_res.content)
                break
            except Exception as e:
                print(e)
                pass
        if len(positive_examples) == 0:
            print("positive examples is empty in class: ", label)
            continue

        negative_query = [system_message, negative_message.format(label=label, n_examples=n_sample)]
        negative_examples = []
        for _ in range(max_retry):
            try:
                ai_res = chat.invoke(negative_query)
                if isinstance(ai_res, str):
                    negative_examples = eval(ai_res)
                else:
                    negative_examples = eval(ai_res.content)
                break
            except:
                pass
        if len(negative_examples) == 0:
            print("negative examples is empty in class: ", label)

        positive_verifications = verification_by_themselves(chat, positive_examples, label, "Yes",
                                        query_template=verification_message_template,
                                        max_retry=max_retry)
        negative_verifications = verification_by_themselves(chat, negative_examples, label, "No",
                                        query_template=verification_message_template,
                                        max_retry=max_retry)
        # TP is a number of the True in the positive_verifications
        TP = sum(positive_verifications)
        TN = sum(negative_verifications)
        FP = n_sample - TP
        FN = n_sample - TN

        tmp_result.append({
            "class label": label,
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN,
            "precision": TP / (TP + FP),
            "negative examples": negative_examples,
            "positive examples": positive_examples,
            "accuracy": (TP + TN) / (TP + TN + FP + FN),
            "n_samples": n_sample,
        })

    return tmp_result


system_message_for_verification = SystemMessage(content="Please answer with 'Yes' or 'No', without no other words.")


def verification_by_themselves(chat: ChatOpenAI, target_items: list[str], label: str,
                               judge_str: str = "Yes",
                               system_query: SystemMessage = system_message_for_verification,
                               query_template: ChatMessagePromptTemplate = verification_template,
                               max_retry: int = 3) -> list[bool]:
    """
    ask llms to verify examples
    :param chat:
    :param target_items:
    :param label:
    :param judge_str: if the answer of llms contains judge_str, it is judged as correct.
    :param system_query:
    :param query_template: message to ask llms to verify examples. With two {} to be replaced by item and label.
        For example: "The {item} is a kind of {label}?"
    :param max_retry: number of retry to invoke llms.
    :return: result: list of bool
    """
    result: list[bool] = []
    for item in target_items:
        answer = ""
        for _ in range(max_retry):
            try:
                ai_res = chat.invoke([system_query, query_template.format(item=item, label=label)])
                # print("AI response in verification", ai_res)
                if isinstance(ai_res, str):
                    answer = ai_res
                else:
                    answer = ai_res.content
                break
            except Exception as e:
                # print("error in verification", e)
                pass
        result.append(judge_str.lower() in answer.lower())
    return result


bulk_verification_system_message = SystemMessage(
    content='''
    You should answer with the literal of list of python and its contents should be `bool` value.
    For example, [True, True, False, True, False].'''
)
bulk_verification_user_message = """The items in the following list are a kind of {label}?
list: {list}
"""
bulk_verification_template = HumanMessage(content=bulk_verification_user_message)


def bulk_verification_by_themselves(chat: ChatOpenAI, target_items: list[str], label: str,
                                    judge_str: str = "Yes",
                                    system_query: SystemMessage = bulk_verification_system_message,
                                    query_template: ChatMessagePromptTemplate = bulk_verification_template,
                                    max_retry: int = 3) -> list[bool]:
    """
    ask llms to verify examples (ask at one time)
    :param chat:
    :param target_items:
    :param label:
    :param judge_str: if the answer of llms contains judge_str, it is judged as correct.
    :param system_query:
    :param query_template: message to ask llms to verify examples. With two {} to be replaced by label and list.
    :param max_retry: number of retry to invoke llms.
    :return: list of bool
    """
    result = []
    for _ in range(max_retry):
        try:
            ai_res = chat.invoke([system_query, query_template.format(
                label=label, list=str(target_items)
            )])
            if isinstance(ai_res, str):
                result = eval(ai_res)
            else:
                result = eval(ai_res.content)
        except Exception as e:
            pass
    return result

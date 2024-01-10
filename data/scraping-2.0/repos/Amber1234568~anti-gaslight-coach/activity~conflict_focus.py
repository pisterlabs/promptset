import cohere
from cohere.classify import Example


def get_class(response: str) -> str:
    examples = [Example("I don\'t care", "confident"),
                Example("yeah yeah", "confident"),
                Example("Yeah, you wish", "confident"),
                Example("As if", "confident"),
                Example("I don\'t think you", "confident"),
                Example("Oh", "low self-esteem"),
                Example("I\'m sorry", "low self-esteem"),
                Example("I think that too", "low self-esteem"),
                Example("Sorry", "low self-esteem"),
                Example("Thank you for pointing out", "low self-esteem"),
                Example("why would say something like that",
                        "peace or not bothered"),
                Example("Huh", "peace or not bothered"),
                Example("Stop being rude", "peace or not bothered"),
                Example("You should not talk to people like that",
                        "peace or not bothered"),
                Example("That\'s rude", "peace or not bothered")]
    co = cohere.Client('mDgdZhuZohzzJnVmOnygreJzaxHyvUD4MdPcEoOq')

    response_class = co.classify(
        model='large',
        inputs=[response],
        examples=examples,
    )

    sentence = str(response_class.classifications[0])
    result1 = sentence.index('prediction: ')
    result2 = sentence.index(', confidence:')

    result_class = sentence[result1 + 13: result2 - 1]
    return result_class


def reasonable_answer(class_response: str) -> str:
    if class_response == "confident":
        return f"Your response shows you are a {class_response} person, \ngood job!"
    elif class_response == "low self-esteem":
        return f"Be confident, express your feeling if you are offended."
    else:
        return f"Good job, " \
               f"you stand your ground and \ndon't let others influence you!"


if __name__ == '__main__':
    start_sentence = "You can’t even get that answer? You’re so dumb!"
    # print("You can’t even get that answer? You’re so dumb!")
    # class_start = get_class(start_sentence)

    response = input("Enter your response: ")
    result_class = get_class(response)
    print("Your response is", result_class)
    reasonable_answer(result_class)
    #print(reasonable_answer(result_class))

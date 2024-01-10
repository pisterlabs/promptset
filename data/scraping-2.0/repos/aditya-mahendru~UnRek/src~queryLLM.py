from langchain.llms import Ollama
from yaspin import yaspin
from consistency import confidence

# from langchain.chains import ConversationChain
# from langchain.memory import ConversationBufferMemory

MODEL = "openhermes2.5-mistral:7b-fp16"

llm = Ollama(
    base_url="http://localhost:11434", model=MODEL, temperature=0.1, stop=["<|im_end|>"]
)


def singleQuestionForInfo(property, intendedTowards, isBool):
    boolQuery = "This is a yes or no question." if isBool else ""
    query = "Generate a formal worded question intended towards {} asking their {} in their bachelor's prgram.{} Generate without any greetings. Ask for single word answers. Only generate One question".format(
        intendedTowards, property, boolQuery
    )
    return llm(query)


def singleQuestionForPreference(property, isBool):
    query = (
        "Generate a formal worded question asking user their preferred {} for their masters program. Only generate the question expecting a one word answer Only generate One question".format(
            property
        )
        if not isBool
        else "Generate a formal worded question to ask if {} is an immportant factor while looking a masters program.Only generate the question expecting a yes or no answer Only generate One question".format(
            property
        )
    )

    return llm(query)


def searchQuery(property, preference):
    query = "Generate a search prompt to be used in a web search to get University data for a aspiring masters student with {} requirement and the {} preferences. Do not generate anything besides the query.".format(
        property, preference
    )

    queryOut = 'masters degree "{}" {} {}'.format(
        preference[0], property[1], property[0]
    )
    return queryOut
    # return llm(query)


def summarizeRanking(textContent, property):
    query = "Consider the following University related data from a webpage:{} From the above data generate a list of top 5 universities matching the {} criteria. Only output the rankings along with the critera for your descision.Only use the provided data.Only generate a list of universities".format(
        textContent, property
    )
    # with yaspin():
    #     confidence_score = confidence(query)
    # function_string = f"Summarizing Ranking: {confidence_score}"
    # with open("./confidence.txt", "a") as file:
    #     file.write(function_string + "\n")
    return llm(query)


def genJsonRankings(textContent, property):
    query = r"""{} 
    Convert it to a JSON list of only university names and {} requirement. Just generate the JSON without saying anything and without markdown. Only use data from the given inputs. Every Json element should be of the following format:
    {{"University":University Name here,{}:Citeria in one word}}
    """.format(
        textContent, property, property
    )

    # print(query)
    # return True
    return llm(query)


def generateIntersection(list1, list2):
    query = "Generate a python list containing the common universities in {} and {}. Generate only a python list as an output.".format(
        list1, list2
    )
    # with yaspin():
    #     confidence_score = confidence(query)
    # function_string = f"Intersecting Universities: {confidence_score}"
    # with open("./confidence.txt", "a") as file:
    #     file.write(function_string + "\n")
    return llm(query)


def checkCountry(list1, countryName):
    query = "{} From the following list remove all the universities that are not in {} country. Only use the data provided. Generate only a python list as an output. Do not generate anything else.".format(
        list1, countryName
    )
    return llm(query)


def summarizeUniData(uniName, data):
    query = "{}  {}. From the above data write a 100 word summary about {}. Do not generate anything else.".format(
        data[0], data[1], uniName
    )
    # with yaspin():
    #     confidence_score = confidence(query)
    # function_string = f"Summarizing University Data: {confidence_score}"
    # with open("./confidence.txt", "a") as file:
    #     file.write(function_string + "\n")
    return llm(query)


# list1 = [
#     "University of California - San Diego (UCSD)",
#     "Stanford University - Department of Computer Science",
#     "University of California, Los Angeles - Department of Computer Science",
#     "Massachusetts Institute of Technology (MIT)",
#     "None",
#     "IIT Delhi",
#     "IIT Madras",
#     "California Institute of Technology",
#     "Princeton University",
#     "Stanford University",
#     "Harvard University",
#     "California Institute of Technology (Caltech)",
#     "Massachusetts Institute of Technology",
#     "Marlan and Rosemary Bourns College of Engineering at UC Riverside",
#     "IIT Bombay",
#     "Massachusetts Institute of Technology - Department of Electrical Engineering and Computer Science",
#     "IIT Roorkee",
#     "IIT Kharagpur",
# ]
# list2 = [
#     "Indian Institute of Technology (IIT), Kanpur",
#     "Carnegie Mellon University",
#     "Indian Institute of Technology (IIT), Kharagpur",
#     "Indian Institute of Technology (IIT) - Kharagpur",
#     "Indian Institute of Technology (IIT) - Madras",
#     "Princeton University",
#     "Stanford University",
#     "Indian Institute of Technology (IIT) - Bombay",
#     "University of California, Berkeley",
#     "ETH Zurich",
#     "University of California - Berkeley",
#     "Indian Institute of Technology (IIT) - Kanpur",
#     "Indian Institute of Technology (IIT), Bombay",
#     "Indian Institute of Technology (IIT), Madras",
#     "Harvard University",
#     "Indian Institute of Technology (IIT), Delhi",
#     "Massachusetts Institute of Technology (MIT)",
#     "Georgia Institute of Technology",
#     "Indian Institute of Technology (IIT) - Delhi",
#     "California Institute of Technology (Caltech)",
#     "MIT",
# ]

# print(checkCountry(generateIntersection(list1, list2), "India"))

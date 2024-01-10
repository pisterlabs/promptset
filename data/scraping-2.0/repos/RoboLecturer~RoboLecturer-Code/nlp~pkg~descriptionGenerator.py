# This script produces the content class (1 per slide) and the descriptive keywords associate with the slide content #

# import packages
# import openai
from pkg.chat import chat_model

# function to create the descriptions
def genKeywords(content):
    """This function creates a list of keywords from the contents of a lecture slide
    Args:
            content - [string] the contents of a lecture slide
    Returns: 
            keywords - [list][string] array of keywords
    """
    # set query
    query = f"give me a list of 5 keyphrases associated with the following text: {content}" 
    # get response to query using selelcted model
    response = chat_model.getResponse(query)
    
    keywords = createKeywords(response)
    return keywords

# reformat response
def createKeywords(response):
    """this function reformats the davinci keyword response into a usable list
    Args:  
            response - [string] response from davinci with keywords
    Returns:
            keywords - [list][string] list containing each keyword alone
    """
    # response from davinci is seperate by new lines
    list_from_reponse = response.split("\n")
    keywords = []
    for item in list_from_reponse:
        # remove empty elements created by \n\n
        if item == "":
            continue
        else:
            # remove the enumeration from the front of each keyword
            # [3:] because list will never be longer than 5
            # e.g - 1. Water ..... 5. Atmosphere
            keywords.append(item[3:])
    return keywords        

# get title and script content
def getTitle(slide):
    """This function extracts the slide title and the script content for use in keyword creation function
    Args:   slide - [string] contents of a slide
    Returns: 
        title - [string] title of the current slide
        content - [string] content of the current slide script
    """
    # split slide content into seperate lines
    lines_of_slide = slide.split("\n")
    # title is the first line
    title = lines_of_slide[0]
    return title 

# create the description and keywords
def createDescription(slide,script,class_description):
    """ This function creates content type and keywords for a slide
    Args:
        slide - [string] content of a single slide
        script - [string] lecture script content
    Returns:
        class_description - [dict] { [string]: [[list][string]] } dictionary containing the class and corresponding keyworkds
    """
    # process text
    title = getTitle(slide)
    keywords = genKeywords(script)
    # add element to dictionary
    class_description[f"{title}"] = keywords

    return class_description

# initialise the class description dictionary with the operational classes
def initDict():
    """This function initialises the class_description dictionary with the operational keys
    Returns:
            class_descriptions - [dict] { [string]: [[list][string]] }
    """
    class_descriptions = {
        "increase speech speed": ["speed up", "faster", "quickly"],
        "decrease speech speed": ["slow down", "slower", "more slowly"],
        "increase speech volume": ["louder", "increase volume", "raise voice","speak up"],
        "decrease speech volume": ["softer", "lower volume", "quieter"],
        "go to previous slide": ["previous slide", "go back"],
        "go to next slide": ["next slide", "advance slide"],
        "go to specific slide number": ["go to slide"],
        "finished": ["yes", "all good", "thank you", "yep"],
    }
    return class_descriptions

from flask import jsonify, request, Blueprint, session
import os, re
import openai

makecypher = Blueprint("make_cypher_query", __name__, url_prefix="/")

openai.api_key = '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'

prompt = """
This information describes the schema of a Neo4J graph database, defining the properties of each node label and the directional relationships between labels.

Node labels and their properties:
• Device: No properties
• Location: locationType, Coordinates
• Presentation: presentation_id, name, aspect_ratio, style, mood, description, uploader 
• Person: Person_id, name, age, gender
• Tag: tag_id, tagName
• Asset: name, aspect_ratio, style, mood, description, uploader, Subject
• Mood: No properties
• Style: No properties
• Group: No properties
• Object: No properties

Relationships between nodes:
• Device-Location: locatedAt
• Location-Device: hasDevice
• Device-Presentation: presented
• Presentation-Device: presentedAt
• Presentation-Tag: hasTag
• Presentation-Mood: hasMood
• Presentation-Style: hasStyle
• Tag-Presentation: taggedAt
• Presentation-Asset: includes
• Asset-Presentation: includedAt
• Asset-Mood: hasMood
• Asset-Style: hasStyle
• Presentation-Person: ownedBy
• Person-Presentation: looked, purchased
• Tag-Asset: taggedAt
• Asset-Tag: hasTag
• Person-Asset: purchased, looked
• Asset-Person: ownedBy
• Asset-Object: includes
• Object-Asset: includedAt
• Mood-Presentation: featureOf
• Mood-Asset: featureOf
• Style-Presentation: featureOf
• Style-Asset: featureOf
• Location-Location: nearBy
• Tag-Tag: taggedWith
• Object-Object: supersetOf, subsetOf, showsTogetherWith
• Mood-Mood: taggedWith
• Style-Style: taggedWith
• Group-Object: proper
• Object-Group: properedBy

"""

@makecypher.route('/make_cypher_query', methods=['POST'])
def make_cypher_query():
    text = request.args.get('text', False)
    res = {}

    # match_obj = re.search(r'(\d+)대', text)
    # if match_obj:
    #     modify_text = text.replace(match_obj.group(), f"{match_obj.group(1)}살 이상 {int(match_obj.group(1)) + 9}살 이하")
    #
    # res['Modify_text'] = modify_text

    # response = openai.Completion.create(
    #     model="text-davinci-003",
    #     prompt= f"""
    #     Translate this sentence {modify_text} into Neo4j Cypher query with this information {prompt}.
    #     """,
    #     temperature=0.5,
    #     max_tokens=1000,
    #     top_p=1.0,
    #     frequency_penalty=0.0,
    #     presence_penalty=0.0
    # )
    #
    # match_query = response['choices'][0]['text'].lower()
    # match_obj = re.search(r'age: (\d+)..(\d+),', match_query)
    # if match_obj:
    #     match_query = match_query.replace(match_obj.group(), '').strip()
    #     match_query = match_query[:match_query.index('return')] + \
    #                   f"where p.age >= {match_obj.group(1)} and p.age <= {int(match_obj.group(1)) + 9}" + \
    #                   match_query[match_query.index('return') - 1:]

    # res['result_1'] = '1' #match_query

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"""
            After dividing {text} into subject, adjective, and object, remove the words 'presentation' and 'image' from the object if they appear. 
            Extract the subject, adjective, and object, and use the extracted words to create Please return only the result in the format "main subject - main adjective - main object".
            """,
        temperature=0.5,
        max_tokens=1000,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    result2 = response['choices'][0]['text']
    # pattern = r"Main Subject - Main Adjective - Main Object: (.*)"
    # result2 = re.search(pattern, result2)

    res['result_2'] = result2

    session.clear()

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"""
                Return the results of a search in GPT for this sentence {result2} that corresponds to "Subject - Adjective - Object".
                Don't explain the results, just return about 10 words to the list.
                """,
        temperature=0.5,
        max_tokens=1000,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    result3 = response['choices'][0]['text']
    res['result_3'] = result3


    return res

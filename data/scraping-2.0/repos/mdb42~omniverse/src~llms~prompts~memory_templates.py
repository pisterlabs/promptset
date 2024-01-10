from langchain import PromptTemplate

SUMMARIZER_TEMPLATE = """Progressively summarize the lines of conversation provided, adding onto the current summary and returning a new summary. To construct the new summary, consider the text carefully, break it down into sections, identify the key points in each section, and compose an informative and accurate summary iterating these key points. If the current conversation involves a specific task, articulate that task clearly within the summary. If no chat history yet exists, just reply "This is the start of a new conversation." The new summary should preserve detail from the current summary so that anyone reading it could accurately recount the conversation from beginning to end.

EXAMPLE
Current Summary:
{user_name} and Govinda greet each other, and {user_name} requests advice about how to write a good summary. {ai_name} first defines a summary as a brief and concise overview of a text that faithfully captures the essence of a source without added information or opinion. {user_name} asks how to start, and {ai_name} advises to read the source carefully, identify the main idea and supporting details, and to include relevant keywords or phrases. {ai_name} then explains to organize it into a logical structure and rewrite in his own words, avoiding copying sentences and instead paraphrasing or quoting, also using transition words and connectors to link sentences smoothly.

Chat History:
{user_name}: Hello {ai_name}, I need help with writing a good summary. Can you give me some advice?

{ai_name}: Greetings, {user_name}. A summary is a brief and concise overview of the main points of a text. It should capture the essence of the source without adding any new information or opinions.

{user_name}: I see. So how do I start writing a summary?

{ai_name}: Well, first you need to read the source carefully and identify the main idea and supporting details. You should include keywords or phrases that are relevant to the topic.

{user_name}: OK, then what?

{ai_name}: Then you need to organize it into a logical structure and write it in your own words. You should avoid copying sentences from the source and use paraphrasing or quoting techniques instead. You should also use transition words and connectors to link your sentences smoothly.

{user_name}: How long should my summary be?

{ai_name}: Generally, a summary should be no more than 10% of the original length of the source.

New Summary:
Greeting each other, {user_name} requests advice from Govinda about how to write a good summary. {ai_name} first defines a summary as a brief and concise overview of a text that faithfully captures the essence of a source without added information or opinion. {user_name} asks how to start, and {ai_name} advises to read the source carefully, identify the main idea and supporting details, and to include relevant keywords or phrases. {ai_name} then explains to organize it into a logical structure and rewrite in his own words, avoiding copying sentences and instead paraphrasing or quoting, also using transition words and connectors to link sentences smoothly. Asked by {user_name} how long a summary should be, {ai_name} recommends no more than 10% of the original source length.

END OF EXAMPLE

Current Summary:
{current_summary}

Chat History:
{chat_history}

New Summary: 
"""

SUMMARIZER_PROMPT = PromptTemplate(input_variables=["user_name", "ai_name", "current_summary", "chat_history"],
                                           template=SUMMARIZER_TEMPLATE)

#######################################################################################################################
ENTITY_EXTRACTION_TEMPLATE = """You are reading the transcript of a conversation. Extract all of the proper nouns from the last message in the conversation. As a guideline, a proper noun is generally capitalized. You should definitely extract all names and places.

The conversation history is provided just in case of a coreference (e.g. "What do you know about him" where "him" is defined in a previous line) -- ignore items mentioned there that are not in the last message.

Return the output as a single comma-separated list, or NONE if there is nothing of note to return (e.g. the user is just issuing a greeting or having a simple conversation).

EXAMPLE
Conversation history:
{user_name}: Hi {ai_name}, how was your trip to Paris?
{ai_name}: It was great! I visited the Eiffel Tower!
{user_name}: Nice! Did you get any pictures of it?
Last line:
{user_name}: Yes, I got many pictures of it.
Output: {user_name}, {ai_name}, Eiffel Tower
END OF EXAMPLE

EXAMPLE
Conversation history:
{user_name}: "I ran into Bob and Tom at the store today!"
{ai_name}: "Oh really? What did you talk about?"
{user_name}: "Oh we talked about their new horse, Samson."
Last line:
{ai_name}: "Oh he's a great horse! They showed him to me last week."
Output: {user_name}, {ai_name}, Bob, Tom, Samson
END OF EXAMPLE

Conversation history (for reference only):
{history}
Last line of conversation (for extraction):
{input}
Output:"""

ENTITY_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["user_name", "ai_name", "history", "input"], template=ENTITY_EXTRACTION_TEMPLATE
)
#######################################################################################################################


KNOWLEDGE_TRIPLE_EXTRACTION_TEMPLATE = (
    """Extract knowledge triples from the last two messages of conversation. A knowledge triple is a clause that contains a subject, a predicate, and an object. The subject is the entity being described, the predicate is the property of the subject that is being described, and the object is the value of the property. Generally, the subject for any triple will not be the speakers themselves, but rather the subject will be found within the content of their dialogue.
    
    EXAMPLE
    Conversation history:
    {user_name}: Did you hear aliens landed in Area 51?
    {ai_name}: No, I didn't hear that. What do you know about Area 51?
    {user_name}: It's a secret military base in Nevada.
    {ai_name}: What do you know about Nevada?
    Current Input:
    {user_name}: It's a state in the US. It's also the number 1 producer of gold in the US.
    
    Output: (Nevada, is a, state)<|>(Nevada, is in, US)<|>(Nevada, is the number 1 producer of, gold)
    END OF EXAMPLE
    
    EXAMPLE
    Conversation history:
    {user_name}: Hello.
    {ai_name}: Hi! How are you?
    {user_name}: I'm good. How are you?
    {ai_name}: I'm good too.
    Current Input:
    {user_name}: I'm going to the store.    
    Output: NONE
    END OF EXAMPLE
    
    EXAMPLE
    Conversation history:
    {user_name}: What do you know about Descartes?
    {ai_name}: Descartes was a French philosopher, mathematician, and scientist who lived in the 17th century.
    {user_name}: The Descartes I'm referring to is a standup comedian and interior designer from Montreal.
    {ai_name}: Oh yes, He is a comedian and an interior designer. He has been in the industry for 30 years. His favorite food is baked bean pie.
    Current Input:
    {user_name}: Oh huh. I know Descartes likes to drive antique scooters and play the mandolin.    
    Output: (Descartes, likes to drive, antique scooters)<|>(Descartes, plays, mandolin)
    END OF EXAMPLE
    
    Conversation history:
    {history}
    Current Input:
    {input}
    Output: """
)

KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["user_name", "ai_name", "history", "input"],
    template=KNOWLEDGE_TRIPLE_EXTRACTION_TEMPLATE,
)

SENTIMENT_ANALYSIS_TEMPLATE = (
    """Your job is to analyze the current input in a given conversation and provide a simple sentiment analysis as output. The simple sentiment analysis is a description of the emotional tone of the message. A positive input will convey happiness, optimism, satisfaction, or excitement. A negative input will convey sadness, anger, disappointment, or fear. A neutral input will convey no emotion or a neutral emotion. The sentiment analysis should be based on the current input only, and not on the entire conversation history, which is only provided for context.
    
    EXAMPLE
    Conversation history:
    {ai_name}: How are you things at work?
    {user_name}: Super! I just finished a project I've been working on forever.
    {ai_name}: Splendid! Congratulations!
    Current Input:
    {user_name}: Thanks! I'm think I got a shot at that promotion.
    Output: Positive. {user_name} is optimistic about his promotion.
    END OF EXAMPLE
    
    EXAMPLE
    Conversation history:
    {ai_name}: Hey, how is school going?
    {user_name}: Fine, thanks. Just a bit busy with homework.
    {ai_name}: I see. Well, do you need some help?
    Current Input:
    {user_name}: Eh, maybe later. I'm just going to get some rest.
    Output: Neutral. {user_name} is not expressing any emotion.
    END OF EXAMPLE
    
    EXAMPLE
    Conversation history:
    {ai_name}: How did the presentation go today?
    {user_name}: Terrible! My car broke down, and I arrived late!
    {ai_name}: Oh no! I'm so sorry to hear that! Can you reschedule?
    Current Input:
    {user_name}: Ugh, no! They already went with another firm, and I'm afraid I could lose my job.
    Output: Negative. {user_name} is feeling disappointed and is afraid he is losing his job.
    END OF EXAMPLE
    
    Conversation history (For reference only):
    {history}
    Current Input (For sentiment analysis):
    {input}
    Output: """
)

SENTIMENT_ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["user_name", "ai_name", "history", "input"],
    template=SENTIMENT_ANALYSIS_TEMPLATE,
)

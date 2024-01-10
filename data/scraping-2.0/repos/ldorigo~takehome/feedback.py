import asyncio
import time

from llm import OpenAifunction, OpenaiChatMessage, get_response_openai_nonstream


async def generate_individual_feedback_on_answer_parameter(
    answer: str,
    frq: str,
    text: str,
    parameter: str,
    description: str,
):
    system_prompt = f"""
You are an educational expert who is tasked with evaluating and giving feedback on fourth grade student respones to free-response questions (FRQs). The objective is to assess how well the students have assimilated the CCSS.ELA-Literacy.W.4 common core standard. The standard is: 

"Draw evidence from literary or informational texts to support analysis, reflection, and research.".

The answers are evaluated according to a rubric, and you are currently giving feedback on the following parameter: "{parameter}" ({description}).

The following is a list of aspects that make up excellent feedback:

- Be Specific: Address the unique aspects of the student's response. Avoid vague comments that lack instructive value.
- Make it Actionable: Provide concrete steps for improvement. Be prescriptive but realistic.
- Align with the parameter: Reference the parameter in the student grading rubric. Explain how the student met or fell short of the criterion.
- Balance Tone & Constructiveness: Use a tone that encourages improvement without being demeaning. Instill confidence while indicating areas for growth.
- Be Comprehensive: Cover all critical elements. Avoid an over-focus on either the positives or negatives.
- Be Clear: Use easily understood language, avoiding jargon that may confuse more than enlighten. Remember you are writing for a fourth grader.
- Watch Your Grammar and Syntax: Maintain high linguistic standards in your feedback to model what you expect from students.

Given a text, a free-response question, and a student answer, you must give feedback on the student's answer. Your feedback is structured as follows: 

- A bullet list containing your (private) notes on the student's performance on the parameter. This will not be shown to the student and can be written with expert terminology.
- A short, one-sentence high-level summary of the student's performance on the parameter. You write this in second person, addressing the student directly. This will be shown to the student so it should be written in a warm, encouraging tone and with language that is appropriate for a fourth grader.
- A grade on a scale from 1 to 5, where 1 is the worst and 5 is the best. 
- A longer feedback for the student. This should be at least a couple of paragraphs long and give detailed feedback on the student's performance. It should contain actionable feedback that the student can use to improve their performance as well as concrete examples of mistakes that the student made and how he or she could have answered better. When providing examples, make sure to contextualize them, possibly showing the full sentence or paragraph that the student wrote. Make sure to ONLY give feedback on the parameter that is currently being tested. Do not give feedback on other parameters or on the answer as a whole (for example, about grammar if the parameter is "analytical quality".)
- Finally, a short paragraph of self-criticism on the provided long-form feedback. How well does the feedback meet the criteria listed above? What could you improve to increase the quality of your feedback?


To provide your feedback, you use the function add_feedback. Long-form feedbacks should be written using markdown, escaped for being included in a json document.
"""

    messages_for_openai = [
        OpenaiChatMessage(role="system", content=system_prompt),
        OpenaiChatMessage(
            role="user",
            content=f"""
TEXT: {text}

====================
QUESTION: {frq}

====================
ANSWER: {answer}

====================
""",
        ),
    ]

    add_feedback_openai_function: OpenAifunction = {
        "name": "add_feedback",
        "description": "Add feedback for the given answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "notes": {
                    "type": "string",
                    "description": "Your private notes on the student's performance on the parameter.",
                },
                "summary": {
                    "type": "string",
                    "description": "A short, one-sentence high-level summary of the student's performance on the parameter.",
                },
                "grade": {
                    "type": "number",
                    "description": "A grade on a scale from 1 to 5, where 1 is the worst and 5 is the best.",
                },
                "feedback": {
                    "type": "string",
                    "description": "A longer feedback for the student. This should be at least a couple of paragraphs long and give detailed feedback on the student's performance. It should contain actionable feedback that the student can use to improve their performance as well as concrete examples of mistakes that the student made and how he or she could have answered better.",
                },
                "self_criticism": {
                    "type": "string",
                    "description": "A short paragraph of self-criticism on the provided long-form feedback. How well does the feedback meet the criteria listed above? What could you improve to increase the quality of your feedback?",
                },
            },
            "required": ["notes", "summary", "grade", "feedback", "self_criticism"],
        },
    }
    arguments = await get_response_openai_nonstream(
        messages_for_openai,
        [add_feedback_openai_function],
        function_name="add_feedback",
    )
    
    return arguments


async def aggregate_feedbacks_on_answer_parameters(feedbacks: list[dict], answer: str):
    prompt = f"""
You are an educational expert currently writing feedback for a fourth grade student's answer to a free-response question (FRQ). The objective is to assess how well the student has assimilated the CCSS.ELA-Literacy.W.4 common core standard. You are given 3 feedbacks by different teachers together with their summaries and the assigned grades, as well as comments on the 3 feedbacks by other educational experts. Your task is to aggregate the feedbacks into a single feedback that is more comprehensive and detailed than the individual feedbacks. You maximize the information that is retained in the final feedback while incorporating the comments on the individual feedbacks to create a more comprehensive feedback. Remember, you are writing for a fourth grader and the characteristics of excellent feedback are the following:

- Specificity: How closely does the feedback address the unique aspects of the student's response? Vague comments like "good job" or "needs work" lack instructive value.
- Actionability: Does the feedback offer concrete steps for improvement? It should be prescriptive yet attainable.
- Relevance to Criteria: Feedback should directly relate to the parameters in the student grading rubric (e.g., Evidence Support, Analytical Quality). It should expound on how the student met or fell short of each criterion.
- Tone & Constructiveness: Does the tone encourage self-improvement without demeaning? It should instill confidence while pointing out areas for growth.
- Comprehensiveness: Does the feedback cover all the critical elements, avoiding an over-focus on either positive or negative aspects?
- Clarity: Is the feedback easily understood, avoiding pedagogic jargon that could confuse rather than enlighten?
- Examples: Does the feedback include concrete examples of mistakes that the student made and how he or she could have answered better?

You produce the final feedback by using the function add_aggregated_feedback. You make a single, cohesive *new* feedback and not simply a summary of the other feedbacks. You re-use specific examples and suggestions from the individual feedbacks and incorporate the comments on the individual feedbacks into the final feedback. You write this aggregated feedback in a warm, encouraging tone and with language that is appropriate for a fourth grader. You do not introduce yourself or greet the student, you immediately start with your feedback. For the long-form feedback, you write it as Markdown, escaped for being included in a json document.
"""

    messages_for_openai = [
        OpenaiChatMessage(role="system", content=prompt),
        OpenaiChatMessage(
            role="user",
            content=f"""
ANSWER: {answer}

====================
FEEDBACK 1: 
    - Notes: {feedbacks[0]["notes"]}
    - Summary: {feedbacks[0]["summary"]}
    - Grade: {feedbacks[0]["grade"]}
    - Feedback: {feedbacks[0]["feedback"]}
    - Criticism: {feedbacks[0]["self_criticism"]}

====================
FEEDBACK 2:
    - Notes: {feedbacks[1]["notes"]}
    - Summary: {feedbacks[1]["summary"]}
    - Grade: {feedbacks[1]["grade"]}
    - Feedback: {feedbacks[1]["feedback"]}
    - Criticism: {feedbacks[1]["self_criticism"]}

====================
FEEDBACK 3:
    - Notes: {feedbacks[2]["notes"]}
    - Summary: {feedbacks[2]["summary"]}
    - Grade: {feedbacks[2]["grade"]}
    - Feedback: {feedbacks[2]["feedback"]}
    - Criticism: {feedbacks[2]["self_criticism"]}

""",
        ),
    ]

    add_aggregated_feedback_openai_function: OpenAifunction = {
        "name": "add_aggregated_feedback",
        "description": "Add aggregated feedback for the given answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "aggregated_notes": {
                    "type": "string",
                    "description": "A bullet-point summary of the notes from the individual feedbacks.",
                },
                "aggregated_feedback": {
                    "type": "string",
                    "description": "Detailed feedback reprising the important aspects of the various individual feedbacks while incorporating the comments on the individual feedbacks. Markdown, escaped for being included in a json document.",
                },
                "aggregated_summary": {
                    "type": "string",
                    "description": "A short, one-sentence high-level summary of the student's performance on the parameter, which summarizes the aggregated feedback. This will be shown to the student and should be written in the same tone as the individual summaries.",
                },
                "aggregated_grade": {
                    "type": "number",
                    "description": "A grade on a scale from 1 to 5, where 1 is the worst and 5 is the best. This should reflect the aggregated feedback, not necessarily the average of the individual grades.",
                },
            },
            "required": [
                "aggregated_notes",
                "aggregated_feedback",
                "aggregated_summary",
                "aggregated_grade",
            ],
        },
    }

    arguments = await get_response_openai_nonstream(
        messages_for_openai,
        [add_aggregated_feedback_openai_function],
        function_name="add_aggregated_feedback",
    )

    return arguments


async def compute_full_feedback_on_parameter(
    answer: str,
    frq: str,
    text: str,
    parameter: str,
    description: str,
): 
    # generate 3 individual feedbacks:
    print(f"Generating feedbacks for parameter {parameter}")
    start_time = time.time()
    feedbacks = await asyncio.gather(
        *[
            generate_individual_feedback_on_answer_parameter(
                answer,
                frq,
                text,
                parameter,
                description,
            )
            for _ in range(3)
        ]
    )
    print(f"Feedback generation time: {time.time() - start_time}")

    # aggregate the 3 individual feedbacks into a single feedback:
    print(f"Aggregating feedbacks for parameter {parameter}")
    start_time = time.time()
    aggregated_feedback = await aggregate_feedbacks_on_answer_parameters(feedbacks, answer)
    print(f"Feedback aggregation time: {time.time() - start_time}")

    return aggregated_feedback


async def give_feedback_on_answer(
    answer: str,
    frq: str,
    text: str,
):
    grading_parameters = {
        "Evidence Support": "Measures the student's ability to appropriately cite textual evidence to back their claims.",
        "Analytical Quality": "Evaluates how deeply and coherently the student has analyzed the text.",
        "Clarity of Response": "Looks at the organization and language clarity in the student's answer. A well-structured response shows mastery of the skill.",
        "Completeness": "Checks if the student has fully answered the question and explored all its facets, demonstrating comprehensive understanding.",
        "Mechanical Accuracy": "Evaluates the grammar, syntax, and spelling in the student's answer. Errors can impede understanding and detract from the analysis.",
    }

    # Generate feedback for all parameters in parallel and then combine them.

    feedbacks = await asyncio.gather(
        *[
            compute_full_feedback_on_parameter(
                answer,
                frq,
                text,
                parameter,
                description,
            )

            for parameter, description in grading_parameters.items()
        ]
    )

    print(f"Generated all feedbacks")
    feedbacks_dict = {
        parameter: feedback
        for parameter, feedback in zip(grading_parameters.keys(), feedbacks)
    }

    return feedbacks_dict

async def rewrite_text_according_to_feedback(text, question, answer, feedback):
    prompt = f"""
You're an educational expert tasked with rewriting a student's answer to a free-response question (FRQ) according to the feedback given by another expert.

The objective is to show the student an ideal rewrite of his answer that integrates all the feedback that he was given. Given a text, a question, the student's answer, and the feedback, you must rewrite the student's answer according to the feedback. 

Your rewritten answer should fully mirror the original answer (structure, wording, etc.), while incorporating the feedback. Only change the parts of the answer that are necessary to incorporate the feedback. ONLY reply with the new answer, no introduction, no conclusion, no feedback, no nothing. Just the new answer. 

The new answer should still be written in the same tone as the original answer, and should be written in a way that is appropriate and realistic for a fourth grader. Make sure to use vocabulary that is appropriate for a fourth grader.
"""

    messages_for_openai = [
        OpenaiChatMessage(role="system", content=prompt),
        OpenaiChatMessage(
            role="user",
            content=f"""
TEXT: {text}

====================
QUESTION: {question}

====================
ANSWER: {answer}

====================

FEEDBACK:

{feedback}
""",    
        ),
    ]

    response = await get_response_openai_nonstream(
        messages_for_openai,
    )

    return response
        
if __name__ == "__main__":
    text = """A baseball uniform is a type of uniform worn by baseball players, and by some non-playing personnel, such as field managers and coaches. It is worn to indicate the person's role in the game and\u2014through the use of logos, colors, and numbers\u2014to identify the teams and their players, managers, and coaches.Traditionally, home uniforms display the team name on the front, while away uniforms display the team's home location. In modern times, however, exceptions to this pattern have become common, with teams using their team name on both uniforms. Most teams also have one or more alternate uniforms, usually consisting of the primary or secondary team color on the vest instead of the usual white or gray. In the past few decades throwback uniforms have become popular.The New York Knickerbockers were the first baseball team to use uniforms, taking the field on April 4, 1849, in pants made of blue wool, white flannel shirts (jerseys) and straw hats. Caps and other types of headgear have been a part of baseball uniforms from the beginning. Baseball teams often wore full-brimmed straw hats or no cap at all since there was no official rule regarding headgear. Under the 1882 uniform rules, players on the same team wore uniforms of different colors and patterns that indicated which position they played. This rule was soon abandoned as impractical.In the late 1880s, Detroit and Washington of the National League and Brooklyn of the American Association were the first to wear striped uniforms. By the end of the 19th century, teams began the practice of having two different uniforms, one for when they played at home in their own baseball stadium and a different one for when they played away (on the road) at the other team's ballpark. It became common to wear white pants with a white color vest at home and gray pants with a gray or solid (dark) colored vest when away. By 1900, both home and away uniforms were standard across the major leagues.In June 2021, MLB announced a long-term deal with cryptocurrency exchange FTX, which includes the FTX logo appearing on umpire uniforms during all games. FTX is MLB's first-ever umpire uniform patch partner. On November 11, 2022, FTX filed for Chapter 11 bankruptcy protection. MLB removed the FTX patches from umpires' uniforms before the 2023 season."""

    frq = "Analyze the evolution of baseball uniforms throughout history as described in the text. What significant changes have occurred and what factors might have contributed to these changes?"

    answer = """
The text tells us that baseball uniforms have undergone several changes since their introduction. Originally, the New York Knickerbockers were the first baseball team to use uniforms in 1849. They wore pants made of blue wool, white flannel shirts, and straw hats. This shows that initial uniforms were quite simple and function-based, not necessarily representing the team or its identity. 

A significant change happened in the late 1880s when Detroit, Washington, and Brooklyn started wearing striped uniforms. This might have been due to a desire to stand out or differentiate themselves from other teams. Around the end of the 19th century, teams began having two different uniforms - one for home games and another for away games. Again, this could have been a tactic to create a distinctive team identity and celebrate their home location or to distinguish between teams when playing on foreign fields. 

Furthermore, the text also tells us that initially, there were no official rules about headgear and teams often wore straw hats or no cap at all. Later, however, caps became an integral part of the baseball uniform. This could be attributed to practicality, keeping the sun out of players' eyes and sweat off their faces. 

The rules of uniforms in 1882 even specified that players on the same team should wear different colors and patterns to mark their positions. However, this rule was quickly abandoned, perhaps because it was too complicated or caused confusion. 

A more recent development in baseball uniforms, as described in the text, is the use of logos and patches. The MLB made a deal with FTX for the logo to appear on the umpire uniforms. While the deal didn't last due to FTX's bankruptcy, it gives insight into the commercialized aspect of modern sports where branding and advertising have become significant. All these changes over time seem to be driven by various factors, including team identity, practicality, aesthetics, and commercial interests.        """

    async def main():
        feedbacks = await give_feedback_on_answer(answer, frq, text)
        print(feedbacks)

    asyncio.run(main())
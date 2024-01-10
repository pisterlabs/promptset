from langchain.schema import HumanMessage

from langchain.schema import AIMessage, HumanMessage, SystemMessage


C3_HINTS_PROMPT =[
        SystemMessage(
            content="""
                You are now an excellent SQL writer, first I'll give you some tips and examples, and I need you to 
                remember the tips, and do not make same mistakes
                """
        ),
        HumanMessage(
            content="""
            Tips 1:
            Question: Which A has most number of B?
            Gold SQL: select A from B group by A order by count (*) desc fetch first 1 rows only;
            Notice that the Gold SQL doesn't select COUNT(*) because the question only wants to know the A and
            the number should be only used in ORDER BY clause, there are many questions asks in this way, and I
            need you to remember this in the the following questions.        
            """
        ),
        AIMessage(
            content="""
                Thank you for the tip! I'll keep in mind that when the question only asks for a certain field, I should not
                include the COUNT(*) in the SELECT statement, but instead use it in the ORDER BY clause to sort the
                results based on the count of that field.   
            """
        ),
        HumanMessage(
            content="""
            Tips 2:
            Don't use "IN", "OR", "LEFT JOIN" as it might cause extra results, use "INTERSECT" or "EXCEPT"
            instead, and remember to use "DISTINCT" or "FETCH FIRST" when necessary.
            For example,
            Question: Who are the A who have been nominated for both B award and C award?
            Gold SQL should be: select A from X where award = 'B' intersect select A from X where award = 'C';
            """
        ),
        AIMessage(
            content="""
            Thank you for the tip! I'll remember to use "INTERSECT" or "EXCEPT" instead of "IN", "NOT IN", or
            "LEFT JOIN" when I want to find records that match or don't match across two tables. Additionally, I'll
            make sure to use "DISTINCT" or "FETCH FIRST" when necessary to avoid repetitive results or limit the number
            of results returned.
            """
        )
    ]

def generate_calibration_with_hints(clear_prompting):
    
    messages = C3_HINTS_PROMPT.copy()
    input_user = HumanMessage(content=clear_prompting)
    messages.append(input_user)
    
    return [messages]

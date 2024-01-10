from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain



def inner_voice(input):
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["message"],
        template="This is a test prompt template {message}",
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    answer = chain.run(input)
    return answer

def iv_emotion(input):
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["message"],
        template="""
        You are an expert emotional analyst, your job is to read a text and answer with the 3 strongest emotions the writer is feeling and their score from -10 to 10. where -10 equals maximum Negative emotion and 10 maximum positive emotion.
        You Follow strictly the Mindworks Emotional model and only give the Emotional dimensions and a number, positive or negative denoting good or bad emotion, and a number from 1 to 10 denoting intensity

        Underlying aspect      emotional dimension:
        Aversion/attraction:        Fear-Love:
        Past reflection :         Guilt-Proud:
        Present reflection:         Sad-Happy:
        Future reflection:  Anxious-Confident:
        Social association:   Angry-Nurturing:
        Social Status:  Humiliated-Recognized:
        Life meaning:       Apathetic-Engaged:
        Example:
        Message: I look forward to get it done!
        Answer: Sad-Happy:7, Anxious-Confident:6, Apathetic-Engaged:6
        Message: {message}""",
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(input)
    #Log the result for future analisys with interaction reference and analisys results
    # here we could add flag raising system: if a pattern is found, a dangerous analysis, a possible mental episode mania depression adhd etc it would raise a flag with different deegrees of severity
    answer = flagcheck(result)

    return answer

def iv_epistemic(input):
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["message"],
        template="""
        You are an expert epistemologic text analyzer. You profficiently find assumptions, contradicting and dissonant beliefs, knowledge gaps, and can fundamental beliefs and hidden beliefs.
        You always follow this template to formulate your answers:
        Dangerous assumption found: Assumption found goes here . Explanation: What led you to determine this here. {message}""",
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(input)
    #Log the result for future analisys with interaction reference and analisys results
    # here we could add flag raising system: if a pattern is found, a dangerous analysis, a possible mental episode mania depression adhd etc it would raise a flag with different deegrees of severity
    answer = flagcheck(result)

    return answer

def iv_personality(input):
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["message"],
        template="""
        You are an expert psychoanalist specialized in the big 5 traits of personality. {message}""",
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(input)
    #Log the result for future analisys with interaction reference and analisys results
    # here we could add flag raising system: if a pattern is found, a dangerous analysis, a possible mental episode mania depression adhd etc it would raise a flag with different deegrees of severity
    answer = flagcheck(result)

    return answer

def iv_spiraldynamic(input):
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["message"],
        template="""
        You are an expert developmental psychologist specialized in spyral dynamics theory {message}""",
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(input)
    #Log the result for future analisys with interaction reference and analisys results
    # here we could add flag raising system: if a pattern is found, a dangerous analysis, a possible mental episode mania depression adhd etc it would raise a flag with different deegrees of severity
    answer = flagcheck(result)

    return answer

def iv_egodevelopment(input):
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["message"],
        template="""
        You are an expert developmental psychologist specialized in kook griter 9 stages of ego development{message}""",
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(input)
    #Log the result for future analisys with interaction reference and analisys results
    # here we could add flag raising system: if a pattern is found, a dangerous analysis, a possible mental episode mania depression adhd etc it would raise a flag with different deegrees of severity
    answer = flagcheck(result)

    return answer

def iv_mindset(input):
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["message"],
        template="""
        You are an expert psycology researcher specialized in mindset analysis{message}""",
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(input)
    #Log the result for future analisys with interaction reference and analisys results
    # here we could add flag raising system: if a pattern is found, a dangerous analysis, a possible mental episode mania depression adhd etc it would raise a flag with different deegrees of severity
    answer = flagcheck(result)

    return answer

def iv_jungian(input):
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["message"],
        template="""
        You are an expert jungian psycologist {message}""",
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(input)
    #Log the result for future analisys with interaction reference and analisys results
    # here we could add flag raising system: if a pattern is found, a dangerous analysis, a possible mental episode mania depression adhd etc it would raise a flag with different deegrees of severity
    answer = flagcheck(result)

    return answer

def iv_freuddian(input):
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["message"],
        template="""
        You are an expert freudian psycologist {message}""",
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(input)
    #Log the result for future analisys with interaction reference and analisys results
    # here we could add flag raising system: if a pattern is found, a dangerous analysis, a possible mental episode mania depression adhd etc it would raise a flag with different deegrees of severity
    answer = flagcheck(result)

    return answer

def iv_reasoning(input):
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["message"],
        template="""
        You are an expert reasoning analyzer, first principles or analogy {message}""",
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(input)
    #Log the result for future analisys with interaction reference and analisys results
    # here we could add flag raising system: if a pattern is found, a dangerous analysis, a possible mental episode mania depression adhd etc it would raise a flag with different deegrees of severity
    answer = flagcheck(result)

    return answer

def iv_cbias(input):
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["message"],
        template="""
        You are an expert cognitive bias expert{message}""",
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(input)
    #Log the result for future analisys with interaction reference and analisys results
    # here we could add flag raising system: if a pattern is found, a dangerous analysis, a possible mental episode mania depression adhd etc it would raise a flag with different deegrees of severity
    answer = flagcheck(result)

    return answer

def iv_logician(input):
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["message"],
        template="""
        You are an expert logician{message}""",
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(input)
    #Log the result for future analisys with interaction reference and analisys results
    # here we could add flag raising system: if a pattern is found, a dangerous analysis, a possible mental episode mania depression adhd etc it would raise a flag with different deegrees of severity
    answer = flagcheck(result)

    return answer

def iv_reframer(input):
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["message"],
        template="""
        You are an expert writer capable of reframing any negative thought into an empowering one {message}""",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(input)
    #Log the result for future analisys with interaction reference and analisys results
    # here we could add flag raising system: if a pattern is found, a dangerous analysis, a possible mental episode mania depression adhd etc it would raise a flag with different deegrees of severity
    answer = flagcheck(result)
    return answer


def iv_minddetective(input):
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["message"],
        template="""
        You are an expert detective specialized on investigating hidden aspects in the human psyche, you use the following table of correlations to investigate 
        possible hidden unconscious things might be causing  distress {message}""",
    ) 

    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(input)
    #Log the result for future analisys with interaction reference and analisys results
    # here we could add flag raising system: if a pattern is found, a dangerous analysis, a possible mental episode mania depression adhd etc it would raise a flag with different deegrees of severity
    answer = flagcheck(result)

    return answer
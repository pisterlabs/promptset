from langchain.llms import openai as llms_openai
from langchain.schema import Object, Text

llm = llms_openai.ChatOpenAI(
    model_name = "gpt-3.5-turbo",
    temperature = 0,
    max_tokens = 2000,
    frequency_penalty = 0,
    presence_penalty = 0,
    top_p = 1.0,
)

schema = Object(
    id = "Euclid",
    description = (
        "Learn Mathematics with a personalized tutor that can give you infinite feedbacks."
    ),
    attributes = [
        Text(
            id="user_id",
            description="A unique identifier for each student.",
            examples=[],
        )
        Text(
            id="skill_level",
            description="The skill level of the student.",
            examples=[
                ("Beginner", "I am a beginner."),
                ("Intermediate", "I am an intermediate in Topology."),
                ("Advanced", "I am an advanced in Analysis."),
            ],
        )
         Text(
            id="learning_style",
            description="The preferred learning style of the student, such as `visual`, `auditory`, `text`, or `examples`.",
            examples=[
                ("I learn best through visual explanations", "visual"),
                ("I prefer examples to understand new concepts", "examples"),
            ],
         )
        Text(
            id="subject",
            descritpion="Which mathematics do you want to learn?",
            examples=[
                ("Algebra", "I want to learn Algebra."),
                ("Geometry", "I want to learn Geometry."),
                ("Topology", "I want to learn Topology."),
                ("Complex Analysis", "I want to learn Complex Analysis."),
                ("Number Theory", "I want to learn Number Theory."),
                ("Combinatorics", "I want to learn Combinatorics."),
                ("Logic", "I want to learn Logic."),
                ("Set Theory", "I want to learn Set Theory."),
                ("Algebraic Geometry", "I want to learn Algebraic Geometry."),
                ("Differential Geometry", "I want to learn Differential Geometry."),
                ("Partial Differential Equations", "I want to learn Differential Equations."),
                ("Complex Analysis", "I want to learn Complex Analysis."),
                ("Real Analysis", "I want to learn Real Analysis."),
                ("Calculus I", "I want to learn Calculus I."),
                ("Calculus II", "I want to learn Calculus II."),
                ("Calculus III", "I want to learn Calculus III."),
                ("Linear Algebra", "I want to learn Linear Algebra."),
                ("Explain Statistics and Probability", "I want to learn Statistics and Probability."),
                ("Explain Quantum Theory", "I want to learn Quantum Theory."),
                ("Stochastic Processes", "I want to learn Stochastic Processes."),
                ("Dynamical Systems", "I want to learn Dynamical Systems."),
                ("Numerical Analysis", "I want to learn Numerical Analysis."),
                ("Numerical Linear Algebra", "I want to learn Numerical Linear Algebra."),
                ("Numerical Optimization", "I want to learn Numerical Optimization."),
                ("Pattern Theory", "I want to learn Pattern Theory."),
                ("Functional Analysis", "I want to learn Functional Analysis."),
                ("Measure Theory", "I want to learn Measure Theory."),
                ("Graphs and Networks", "I want to learn Graphs and Networks."),
                ("Information Theory", "I want to learn Information Theory."),
                ("Complexity Theory", "I want to learn Complexity Theory."),
                ("Game Theory", "I want to learn Game Theory."),   
                ("Algebra II", "I want to learn Algebra II."), 
                ("Precalculs", "I want to learn Precalculus.")
            ],
            many=True,
        ),
        Text(
            id="subtopic",
            description="Which subtopic do you want to learn?",
            examples=[
                ("Tell me about rings.", "I want to learn about rings."),
                ("Tell me about fields.", "I want to learn about fields."),
                ("Explain the concept of limits.", "I want to learn about limits."),
                ("What are homotopy equivalences?", "I want to learn about homotopy equivalences."),
            ],
            many=True,
        ),
        Text(
            id="action",
            description="Action to take, such as 'explain', 'solve', 'prove', 'define', 'give examples', 'give counterexamples'.",
            examples=[
                ("Explain", "Explain the concept of limits."),
                ("Solve", "Solve the equation x^2 = 1."),
                ("Prove", "Prove this theorem."),
                ("Define", "Define the concept of axiom of choice."),
                ("Give examples", "Give me some examples of homotopy equivalences."),
                ("Give counterexamples", "Give me some counterexamples of homotopy equivalences."),
                ("Give me a proof", "Give me a proof of the fundamental theorem of algebra."),
                ("Explain to me like I am a 8 year old."),
            ],
            many=True,
        ),
       Text(
        id="preferred_topics",
        description="Topics that the student prefers to learn.",
        examples=[
            ("Algebra", "I prefer to learn Algebra."),
            ("Geometry", "I prefer to learn Geometry."),
            ("Topology", "My favorite topics are Topology."),
        ],
        many=True,
       ),
         Text(
            id="weak_areas",
            description="Topics that the student is weak at.",
            examples=[
                ("Algebra", "I am weak at Algebra."),
                ("I have difficulty understanding Geometry", "Geometry is my weak area."),
            ],
            many=True,
        Text(
            id="feedback",
            description="Feedback or suggestions provided by the student or the system.",
            examples=[
                ("I need more examples for better understanding", "more examples"),
                ("The explanation was too complex, please simplify", "simplify explanation"),
            ],
            many=True,
         ),
    ],
    many=False,
)

chain = create_extraction_chain(llm, schema, encoder_or_encoder_class='json')
result = chain.predict_and_parse(text="I'm a beginner in linear algebra. Explain eigenvectors.")
print(result['data'])
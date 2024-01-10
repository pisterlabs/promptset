from inner_dialog.inner_dialog import Supervisor, end_token
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
)


def test_supervisor_init():
    question = "things to consider before starting a business"
    supervisor = Supervisor(
        question=question,
        system_message=SystemMessage(
            content="You are a experience supervisor.",
        ),
        model=ChatOpenAI(temperature=0.2),
    )
    assert supervisor.question == question
    assert supervisor.note == ""
    assert len(supervisor.subtopics) == 0
    assert supervisor.num_subtopics == 3
    assert supervisor.current_subtopic == ""


def test_supervisor_conclude():
    question = "things to consider before starting a business"
    dialog = [
        "What are some potential risks and challenges that entrepreneurs commonly face when starting a business, and how can they mitigate or overcome them?",
        """Starting a business can be an exciting venture, but it is important for entrepreneurs to be aware of potential risks and challenges that they may encounter. Here are some key risks and challenges to consider, along with strategies to mitigate or overcome them:
1. Financial Risks: Insufficient capital, cash flow problems, and unexpected expenses can pose significant challenges. Entrepreneurs should conduct thorough financial planning, secure adequate funding, and maintain a contingency fund to mitigate these risks.
2. Market Risks: Entrepreneurs need to assess market demand and competition before launching a business. Conducting market research, identifying target customers, and developing a unique value proposition can help mitigate market risks.
3. Operational Risks: Inefficient processes, supply chain disruptions, and operational inefficiencies can impact business performance. Entrepreneurs should establish robust operational systems, implement quality control measures, and have contingency plans in place to mitigate operational risks.
    """,
    ]
    note = """
- Market risks: Conducting market research, identifying target customers, and developing a unique value proposition can help mitigate market risks.
- Operational risks: Establishing robust operational systems, implementing quality control measures, and having contingency plans in place can mitigate operational risks.
    """
    supervisor = Supervisor(
        question=question,
        system_message=SystemMessage(
            content="You are a experience supervisor.",
        ),
        model=ChatOpenAI(temperature=0.2),
    )
    supervisor.next_subtopic()
    supervisor.note = note  # Insert initial note.
    supervisor.conclude(dialog)
    print(f"Supervisor final note: {supervisor.note}")


def test_supervisor_next_subtopics():
    question = "things to consider before starting a business"
    supervisor = Supervisor(
        question=question,
        system_message=SystemMessage(
            content="You are a experience supervisor.",
        ),
        model=ChatOpenAI(temperature=0.2),
    )
    while supervisor.current_subtopic != end_token:
        supervisor.next_subtopic()
        print(f"Supervisor first sub topic: {supervisor.current_subtopic}")


def test_supervisor_integrate():
    question = "things to consider before starting a business"
    keypoints = """
- Financial risks: Insufficient capital, cash flow problems, and unexpected expenses can be mitigated through thorough financial planning, securing adequate funding, and maintaining a contingency fund.
- Market risks: Conducting market research, identifying target customers, and developing a unique value proposition can help mitigate market risks.
- Legal and regulatory risks: Staying updated on laws, seeking legal advice, and ensuring proper licenses and permits are obtained can mitigate legal and regulatory risks.
    """
    note = """
- Market risks: Conducting market research, identifying target customers, and developing a unique value proposition can help mitigate market risks.
- Operational risks: Establishing robust operational systems, implementing quality control measures, and having contingency plans in place can mitigate operational risks.
    """
    supervisor = Supervisor(
        question=question,
        system_message=SystemMessage(
            content="You are a experience supervisor.",
        ),
        model=ChatOpenAI(temperature=0.2),
    )
    supervisor.next_subtopic()  # Decide subtopic.
    print(f"Supervisor sub topic: {supervisor.current_subtopic}")
    print(f"Supervisor keypoints: {keypoints}")
    supervisor.note = note  # Insert some note
    print(f"Supervisor initial note: {supervisor.note}")
    keypoints = supervisor.integrate(keypoints=keypoints)
    print(f"Supervisor final note: {supervisor.note}")


def test_supervisor_extract_keypoints():
    question = "things to consider before starting a business"
    dialog = [
        "What are some potential risks and challenges that entrepreneurs commonly face when starting a business, and how can they mitigate or overcome them?",
        """Starting a business can be an exciting venture, but it is important for entrepreneurs to be aware of potential risks and challenges that they may encounter. Here are some key risks and challenges to consider, along with strategies to mitigate or overcome them:
1. Financial Risks: Insufficient capital, cash flow problems, and unexpected expenses can pose significant challenges. Entrepreneurs should conduct thorough financial planning, secure adequate funding, and maintain a contingency fund to mitigate these risks.
2. Market Risks: Entrepreneurs need to assess market demand and competition before launching a business. Conducting market research, identifying target customers, and developing a unique value proposition can help mitigate market risks.
3. Operational Risks: Inefficient processes, supply chain disruptions, and operational inefficiencies can impact business performance. Entrepreneurs should establish robust operational systems, implement quality control measures, and have contingency plans in place to mitigate operational risks.
    """,
    ]
    supervisor = Supervisor(
        question=question,
        system_message=SystemMessage(
            content="You are a experience supervisor.",
        ),
        model=ChatOpenAI(temperature=0.2),
    )
    supervisor.next_subtopic()  # Decide subtopic.
    print(f"Supervisor sub topic: {supervisor.current_subtopic}")
    keypoints = supervisor.extract_keypoints(dialog=dialog)
    print(f"Supervisor keypoints: {keypoints}")


if __name__ == "__main__":
    # test_supervisor_extract_keypoints()
    # test_supervisor_integrate()
    # test_supervisor_next_subtopics()
    # test_supervisor_init()
    test_supervisor_conclude()

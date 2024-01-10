from typing import List, Tuple, Optional
import random
from langchain.llms.openai import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.schema.messages import AIMessage

import src.utils.config as config
from src.twenty_questions_game import TwentyQuestionsGame


def game_loop(game: TwentyQuestionsGame) -> None:
    """
    Executes the main game loop for a given TwentyQuestionsGame instance.
    """
    init_message = "Let's play 20 Questions! Think of an object, and I will try to guess it. You can only answer 'Yes' or 'No'."
    print("AI: ", init_message)
    game.memory.chat_memory.add_ai_message(init_message)

    game_over = False

    while not game_over:
        user_input = input("Your answer (Yes/No): ")
        response, game_over, question_count = process_game_turn(game, user_input, None)

        print(f"AI: ", response)
        print(question_count)

        if game_over:
            print(
                f"AI successfully guessed the object correctly in {question_count} questions!"
            )
        elif question_count >= 20:
            print("Game over! The AI failed to guess the object in 20 questions.")
            break


def bulk_test_game(
    game_instance: TwentyQuestionsGame, concepts: List[str], num_games: int = 10
) -> List[Tuple[str, int]]:
    """
    Simulates multiple rounds of the 20 Questions game for testing purposes.

    Args:
        game_instance: Instance of TwentyQuestionsGame for running the simulation.
        concepts: List of concepts for the AI user to think of.
        num_games: Number of games to simulate.

    Returns:
        Dictionary with detailed results and a performance score.
    """
    results = []
    for _ in range(num_games):
        concept = random.choice(concepts)
        result = simulate_game(game_instance, concept)
        results.append(result)

    # Calculate performance metrics
    success_count = sum(1 for _, outcome, _ in results if outcome == "Success")
    average_questions = (
        sum(question_count for _, _, question_count in results) / num_games
    )
    performance_score = success_count / num_games  # Ratio of successful games

    return {
        "detailed_results": results,
        "success_count": success_count,
        "average_questions": average_questions,
        "performance_score": performance_score,  # Higher is better
    }


def process_game_turn(
    game: TwentyQuestionsGame, user_input: str, concept: Optional[str]
) -> Tuple[str, bool, int]:
    """
    Processes a single turn in the 20 Questions game.

    Args:
        game: Instance of TwentyQuestionsGame for running the game.
        user_input: User (or AI) input for the game turn.

    Returns:
        A tuple containing the AI response, a flag indicating if the game is over, and the current question count.
    """
    game.memory.chat_memory.add_user_message(user_input)
    response = game.run(user_input)

    game_over = (
        response.strip().lower().startswith("hooray")
        or concept.lower() in response.lower()
    )

    ai_messages = [
        msg for msg in game.memory.chat_memory.messages if isinstance(msg, AIMessage)
    ]
    question_count = len(ai_messages) - 1

    return response, game_over, question_count


def simulate_game(game_instance: TwentyQuestionsGame, concept: str) -> Tuple[str, int]:
    """
    Simulates a single round of the 20 Questions game using an AI user.

    Args:
        game_instance: Instance of TwentyQuestionsGame for running the game.
        llm_chain: Instance of LLMChain to generate AI user responses.
        concept: The concept the AI user is thinking of.

    Returns:
        Tuple with the result of the game ('Success' or 'Failure') and number of questions asked.
    """
    config.load_env_variables()
    _, deployment_name, _, _, _ = config.get_api_credentials(False)
    game_instance.reset_game()

    print("\nHuman: The concept is ...", concept)
    init_message = "Let's play 20 Questions! Think of an object, and I will try to guess it. You can only answer 'Yes' or 'No'."
    print("AI: ", init_message)
    game_instance.memory.chat_memory.add_ai_message(init_message)

    question_count = 0
    user_input = "Yes"  # First input is always 'Yes'

    while question_count < 20:
        response, game_over, question_count = process_game_turn(
            game_instance, user_input, concept
        )
        print("\nHuman: ", user_input)
        print("\nAI: ", response)

        if game_over:
            return concept, "Success", question_count

        # Get the next user input
        latest_question = game_instance.get_latest_question()
        user_input = ai_user_response(
            llm=AzureChatOpenAI(azure_deployment=deployment_name, temperature=0),
            concept=concept,
            question=latest_question,
        )

    return concept, "Failure", question_count


def ai_user_response(llm: AzureOpenAI, concept: str, question: str) -> str:
    """
    Generates a binary response using an AzureChatOpenAI LLMChain based on the concept and the question.

    Args:
        llm_chain: An instance of LLMChain to generate responses.
        concept: The concept or object the AI user is thinking of.
        question: The question asked by the guessing AI.

    Returns:
        'Yes' or 'No' based on the AI's response.
    """
    # Construct the prompt for the AI model
    prompt = f"""
    You are playing the '20 Questions' game with another player. Your role is to answer 'Yes' or 'No' to questions based on a given concept or object.

    ## Concept/Object:
    The concept/object for this session is identified as {concept}.
    ## Rules for Answering Questions:
    Direct Relevance: If the binary question ({question}) asked by the player is directly related to the {concept}, respond truthfully based on the nature of the {concept}.
    - Answer 'YES' if the {question} correctly pertains to the {concept}.
    - Answer 'NO' if the {question} does not pertain to the {concept}.
    Answer:"""
    # Get the AI's response
    ai_response = llm.predict(prompt)

    # Process and return the AI's response
    return "Yes" if ai_response.strip().lower() == "yes" else "No"

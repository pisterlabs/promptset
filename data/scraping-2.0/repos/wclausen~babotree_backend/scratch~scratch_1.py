import openai
import tiktoken

from app import babotree_utils
from app.database import get_direct_db
from app.models import Highlight, ContentEmbedding, HighlightSource

# openai.api_key = babotree_utils.get_secret('OPENAI_API_KEY')
openai_client = openai.OpenAI(
    api_key=babotree_utils.get_secret('TOGETHER_API_KEY'),
    base_url="https://api.together.xyz/v1",
)
def get_llm_response(highlights, prompt):
    openai_question_function_tools = [{
        "type": "function",
        "function": {
            "name": "create_question_answer_pair",
            "description": "Creates a question answer pair",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question string",
                    },
                    "answer": {
                        "type": "string",
                        "description": "The correct answer to the question",
                    },
                },
                "required": ["question", "answer"]
            },
        }
    },
    ]
    messages = [
        {
            "role": "system",
            "content": "You are an expert educator, a master of helping students traverse Bloom's taxonomy and understand subjects on a deep level.",
        },
        {
            "role": "user",
            "content": prompt + "\n---\n" + "\n".join(
                [highlight.text for highlight in highlights]) + "\n---\nBe very concise.",
        },
    ]
    print("Fetching questions from openai")
    print(messages[1]['content'])
    tiktoken_encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    text_tokens = tiktoken_encoding.encode(messages[1]['content'])
    print(f"Used this many tokens in this request: {len(text_tokens)}")
    response = openai_client.chat.completions.create(
        model='mistralai/Mixtral-8x7B-Instruct-v0.1',
        messages=messages,
        temperature=1.0,
        max_tokens=1000,
        top_p=1.0,
        frequency_penalty=.25,
        presence_penalty=.45,
    )
    print("--- OPENAI RESPONSE ---")
    print(response.choices[0].message.content)
    print(f"-- Used this many tokens: {response.usage.total_tokens} --")
    print("--- HERE'S A SHORTER RESPONSE ---")
    # add the response message content to the prompt and tell the model to be more concise
    messages.append({
        "role": "assistant",
        "content": response.choices[0].message.content,
    })
    messages.append({
        "role": "user",
        "content": "Be more concise in your answer.",
    })
    response = openai_client.chat.completions.create(
        model='mistralai/Mixtral-8x7B-Instruct-v0.1',
        messages=messages,
        temperature=.5,
        max_tokens=600,
        top_p=1.0,
        frequency_penalty=.6,
        presence_penalty=.75,
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content


def scratch():
    db = get_direct_db()
    docker_highlights = db.query(Highlight).filter(Highlight.source_id.in_(['23c2b96b-ff28-4a62-9d9b-b4fc7563cec8'])).all()
    print()
    # for docker_highlight in docker_highlights:
    #     docker_highlight_embedding = db.query(ContentEmbedding).filter(ContentEmbedding.source_id == docker_highlight.id, ContentEmbedding.source_type == 'HIGHLIGHT_TEXT').first()
    #     # print(docker_highlight_embedding)
    #     # print()
    #     closest_highlights = db.query(Highlight).join(ContentEmbedding, ContentEmbedding.source_id == Highlight.id).filter(ContentEmbedding.source_type == 'HIGHLIGHT_TEXT', Highlight.id != docker_highlight.id).order_by(ContentEmbedding.embedding.cosine_distance(docker_highlight_embedding.embedding)).limit(3).all()
    #     get_llm_response([docker_highlight] + closest_highlights, "Consider the following excerpts from an article about CSS. Formulate a question/answer pair from these excepts that tests the readers knowledge. The question should test an atomic unit of information, avoid questions with multiple parts or answers, and the question should not be a True/False question:")
    #     print("-- HIGHLIGHTS THAT CREATED THIS QUESTION --")
    #     print("\n".join([x.text for x in [docker_highlight] + closest_highlights]))
    #     print('----')
    # now get an llm response for a question that prompts the user for an open-ended question that tests their knowledge across all of the highlights from the source
    # get_llm_response(docker_highlights, "Consider the following excerpts from an article about CSS. Formulate a question/answer pair from these excepts that tests the reader's knowledge. The question should be open-ended and encourage the reader to combine mulitple information from multiple excerpts. Ideally, the question prompts the reader to use the knowledge in a real-world situation.")
    # now let's try keyword extraction
    # keywords = get_llm_response(docker_highlights, "Please extract a list of keywords from the following text. Return the list as comma-separated values.")
    # for keyword in keywords.split(','):
    #     highlights_mentioning_keyword = [highlight for highlight in docker_highlights if keyword.strip().lower() in highlight.text.lower()][:5]
    #     print(f"--- HIGHLIGHTS MENTIONING {keyword.strip()} ---")
    #     print("\n".join([highlight.text for highlight in highlights_mentioning_keyword]))
    #     messages = [{
    #         "role": "system",
    #         "content": "You are a helpful assistant with an expert knowledge in how to define terms."
    #     },
    #     {
    #         "role": "user",
    #         "content": f"Here are some excerpts from an article that contain the keyword \"{keyword.strip()}\":\n{[highlight.text for highlight in highlights_mentioning_keyword]}"
    #     },
    #     {
    #         "role": "assistant",
    #         "content": f"Ok, I'm ready."
    #     },
    #     {
    #         "role": "user",
    #         "content": f"Please define \"{keyword.strip()}\" in this context in a single sentence."
    #     }]
    #     response = openai_client.chat.completions.create(
    #         model='mistralai/Mixtral-8x7B-Instruct-v0.1',
    #         messages=messages,
    #         temperature=1.0,
    #         max_tokens=1000,
    #         top_p=1.0,
    #         frequency_penalty=.25,
    #         presence_penalty=.45,
    #     )
    #     print("--- OPENAI RESPONSE ---")
    #     print(response.choices[0].message.content)
    #     more_concise_count = 0
    #     while len(response.choices[0].message.content) > 140 and more_concise_count < 5:
    #         print("...response was more than 140 characters...making more concise")
    #         print("--- MORE CONCISE ---")
    #         messages.append({
    #             "role": "assistant",
    #             "content": response.choices[0].message.content,
    #         })
    #         messages.append({
    #             "role": "user",
    #             "content": "Be more concise, remove words like \"in this context\", just provide the definition in a single sentence.",
    #         })
    #         response = openai_client.chat.completions.create(
    #             model='mistralai/Mixtral-8x7B-Instruct-v0.1',
    #             messages=messages,
    #             temperature=1.0,
    #             max_tokens=1000,
    #             top_p=1.0,
    #             frequency_penalty=.25,
    #             presence_penalty=.45,
    #         )
    #         print(response.choices[0].message.content)
    #         more_concise_count += 1
    # now take a look at the embeddings to determine what are the titles of the sources most similar to this source
    relevant_source_id = '23c2b96b-ff28-4a62-9d9b-b4fc7563cec8'
    relevant_source_full_highlights_embedding = db.query(ContentEmbedding).filter(ContentEmbedding.source_type == 'FULL_HIGHLIGHTS_FROM_SOURCE', ContentEmbedding.source_id == relevant_source_id).first()
    most_similar_source_embeddings = db.query(ContentEmbedding).filter(ContentEmbedding.source_type == 'FULL_HIGHLIGHTS_FROM_SOURCE', ContentEmbedding.source_id != relevant_source_id).order_by(ContentEmbedding.embedding.cosine_distance(relevant_source_full_highlights_embedding.embedding)).limit(5).all()
    most_similar_source_ids = [source_embedding.source_id for source_embedding in most_similar_source_embeddings]
    most_similar_sources = db.query(HighlightSource).filter(HighlightSource.id.in_(most_similar_source_ids)).all()

    # now let's do this for the 10 most recent sources
    most_recent_sources = db.query(HighlightSource).order_by(HighlightSource.created_at.desc()).limit(10).all()
    for recent_source in most_recent_sources:
        recent_source_id = recent_source.id
        print("--- MOST SIMILAR SOURCES TO RECENT SOURCE ---")
        print(recent_source.title)
        print("-----")
        recent_source_full_highlights_embedding = db.query(ContentEmbedding).filter(ContentEmbedding.source_type == 'FULL_HIGHLIGHTS_FROM_SOURCE', ContentEmbedding.source_id == recent_source_id).first()
        most_similar_source_embeddings = db.query(ContentEmbedding).filter(ContentEmbedding.source_type == 'FULL_HIGHLIGHTS_FROM_SOURCE', ContentEmbedding.source_id != recent_source_id).order_by(ContentEmbedding.embedding.cosine_distance(recent_source_full_highlights_embedding.embedding)).limit(5).all()
        most_similar_source_ids = [source_embedding.source_id for source_embedding in most_similar_source_embeddings]
        most_similar_sources = db.query(HighlightSource).filter(HighlightSource.id.in_(most_similar_source_ids)).all()
        print("\n".join([source.title for source in most_similar_sources]))
        print("---------------------")


    print("--- MOST SIMILAR SOURCES ---")
    print("\n".join([source.title for source in most_similar_sources]))

def make_outline_from_highlights():
    db = get_direct_db()
    docker_highlights = db.query(Highlight).filter(Highlight.source_id.in_(['0a16c1a1-33b3-4777-8d2a-59347d1a985a'])).all()
    db.close()
    print()
    messages = [{
        "role": "system",
        "content": "You are a helpful assistant with an expert knowledge in how to create study outlines on various topics."
    },
        {
            "role": "user",
            "content": "Here are some excerpts from an article about Docker:\n" + "\n".join(
                [highlight.text for highlight in docker_highlights])
        },
        {
            "role": "assistant",
            "content": "Ok, I'm ready."
        },
        {
            "role": "user",
            "content": "Please create an outline in markdown syntax about the key concepts in the excepts."
        }]
    response = openai_client.chat.completions.create(
        model='mistralai/Mixtral-8x7B-Instruct-v0.1',
        messages=messages,
        temperature=1.0,
        max_tokens=1000,
        top_p=1.0,
        frequency_penalty=.25,
        presence_penalty=.45,
    )
    print("--- OPENAI RESPONSE ---")
    print(response.choices[0].message.content)

def main():
    # scratch()
    make_outline_from_highlights()

if __name__ == '__main__':
    main()
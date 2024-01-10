# RAG for Arxiv abstracts 2021

from datasets import load_dataset
from tqdm import tqdm

from ragen.args import build_argument_parser
from ragen.client import OpenAIClient, PgClient, generate_prompt
from ragen.file import Chunk


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    data = "gfissore/arxiv-abstracts-2021"

    chat_client = OpenAIClient(args.model, args.api_key, args.api_base)
    emb_client = OpenAIClient(args.emb_model, args.emb_api_key, args.emb_api_base)
    db_client = PgClient(
        args.db_host, args.db_user, args.db_password, args.db_port, args.emb_dim
    )

    datasets = load_dataset(data, split="train", streaming=True)
    for paper in tqdm(datasets):
        text = paper.get("abstract").strip()
        chunk = Chunk(
            filename=paper.get("title").strip(),
            index=0,
            text=text,
            emb=emb_client.embeddings(text),
            tags=paper.get("categories")[0].split(" "),
        )
        db_client.insert_chunk(chunk=chunk)
    db_client.indexing()

    while True:
        try:
            user_input = input("Enter your question: ")
        except KeyboardInterrupt:
            print("\nBye!")
            break
        request = Chunk(
            filename="", index=0, text=user_input, emb=emb_client.embeddings(user_input)
        )
        user_context = db_client.retrieve_similar_chunk(request, args.top_k)
        chat_client.chat(
            generate_prompt(user_context, request),
        )


if __name__ == "__main__":
    main()

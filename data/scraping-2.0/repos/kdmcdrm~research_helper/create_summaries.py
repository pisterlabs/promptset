from pathlib import Path
from dotenv import load_dotenv
from langchain.document_loaders import PyMuPDFLoader
from agents import OpenAIResearchAgent
import logging
import os
from tqdm import tqdm
from summary import summarize_paper

logger = logging.getLogger(__name__)
_ = load_dotenv()
if "OPENAI_API_KEY" not in os.environ:
    raise EnvironmentError("Environment must define the OPENAI_API_KEY.")
MODEL_NAME = "gpt-3.5-turbo"
AGENT = OpenAIResearchAgent(MODEL_NAME, os.environ["OPENAI_API_KEY"])


def create_paper_summaries(papers_dir: str) -> dict[str, str]:
    """
    Creates a set of summaries of papers in the directory, in a /summaries
    subfolder. Only .pdf is currently supported.
    Args:
        papers_dir: Path to the folder of .pdfs
    Returns:
        summaries: A list of text summaries of the paper content
    """
    logger.info("---- Create Summaries ----")
    papers_dir = Path(papers_dir)
    paper_paths = [str(x) for x in papers_dir.glob("*.pdf")]
    if len(paper_paths) == 0:
        raise ValueError("No .pdfs found in {str(papers_dir)}, please add papers for summarization.")

    logger.info(f"Running summarization for {len(paper_paths)} papers in {str(papers_dir)}")
    sum_dir = papers_dir / "summaries"
    sum_dir.mkdir(exist_ok=True)

    summaries = {}
    for paper_path in tqdm(paper_paths, "Summarizing Papers"):
        paper_name = Path(paper_path).stem
        sum_path = sum_dir / (paper_name + "_summary.md")

        # Only generate summary if it doesn't already exist
        if not sum_path.exists():
            loader = PyMuPDFLoader(str(paper_path))
            summary = summarize_paper(loader.load(), AGENT, "reduce")
            # Save summary to disk
            logger.debug(f"Saving summary to {sum_path}")
            with open(sum_path, "w", encoding="utf8") as fh:
                fh.write(summary)
        else:
            with open(sum_path, "r") as fh:
                summary = fh.read()

        # Store to dictionary for later reference
        summaries[paper_name] = summary

    return summaries


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_paper_summaries("./papers")

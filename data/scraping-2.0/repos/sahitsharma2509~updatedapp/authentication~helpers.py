from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.llms import OpenAI



llm = OpenAI(temperature=0)


from django.apps import apps

def get_knowledge_summary(knowledgebase):
    try:
        # Retrieve knowledge documents associated with the knowledgebase
        knowledge_documents = knowledgebase.documents.all()
        print(f"Number of documents: {knowledge_documents.count()}")

        # If there's only one document, just return its content as summary
        if knowledge_documents.count() == 1:
            summary_content = knowledge_documents.first().content
            print("Single document summary content:", summary_content)

        # If there's more than one document, concatenate their content and summarize it
        else:
            combined_content = " ".join([doc.content for doc in knowledge_documents])
            print("Type 1:",type(combined_content))
            print("Combined content:", combined_content)

            summary_chain = load_summarize_chain(llm, chain_type="map_reduce")

            document = Document(page_content=combined_content,lookup_index=0)
            print("Document:", document)
            print("Document type:", type(document))
            summary_content = summary_chain.run(document)

            print("Summary content:", summary_content)
            print("Summary content type:", type(summary_content))

    except Exception as e:
        print(f"Error while generating summary: {str(e)}")
        summary_content = "Error occurred while generating summary."

    return summary_content



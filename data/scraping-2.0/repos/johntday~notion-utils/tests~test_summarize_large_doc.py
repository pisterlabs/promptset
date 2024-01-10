from langchain.document_loaders import WebBaseLoader
from notion_utils.summarize_large_doc import summarize, summarize_pdf

# SUMMARIZE WEB PAGE
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()

summary_post = summarize(docs)
print(summary_post)


# SUMMARIZE PDF
pdf_url = "https://file.notion.so/f/s/072dc982-d142-41df-abaa-4fe779f6573e/sap-commerce-cloud_2211_intelligent-selling-services.pdf?id=ba4e99c4-c22d-4f94-9bfd-1c8191a28c98&table=block&spaceId=b1d1c56c-4fa2-4928-802e-2030e1e91d56&expirationTimestamp=1697745600000&signature=v3cn10R4CNTzmNzJW7lvblQ95FgBFjFPzv53q2RrWZg&downloadName=sap-commerce-cloud_2211_intelligent-selling-services.pdf"
summary_pdf = summarize_pdf(pdf_url)
print(summary_pdf)

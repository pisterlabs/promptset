from langchain.document_loaders import (
    TextLoader,
    DirectoryLoader,
    CSVLoader,
    BSHTMLLoader,
    JSONLoader,
    UnstructuredMarkdownLoader,
)

# -------------------------------------------
# -------------------------------------------
# Basics
# -------------------------------------------
# -------------------------------------------

# ---------------------
# TEXT file - one file
# ---------------------
loader = TextLoader("./source/forrest-gump.txt")

result = loader.load()
print(result)


# ---------------------
# Directory - all files
# ---------------------
loader = DirectoryLoader("./source", glob="**/*.txt")

result = loader.load()
print(result)

# ---------------------
# CSV
# ---------------------
loader = CSVLoader(file_path="./source/TSLA.csv")

result = loader.load()
print(result)


# ---------------------
# HTML
# ---------------------
loader = BSHTMLLoader("./source/website.html")

result = loader.load()
print(result)

# ---------------------
# JSON
# ---------------------
loader = JSONLoader(file_path="./source/data.json", jq_schema=".content")

result = loader.load()
print(result)

# ---------------------
# Markdown
# ---------------------
loader = UnstructuredMarkdownLoader("./source/readme.md")

result = loader.load()
print(result)

# ---------------------
# Pdf
# ---------------------
# loader = PyPDFLoader("./source/book.pdf")

# result = loader.load()
# print(result)


# -------------------------------------------
# -------------------------------------------
# Integrations
# -------------------------------------------
# -------------------------------------------


# ---------------------
# Microsoft Excel
# ---------------------


# ---------------------
# Azure Blob storage cointainer
# ---------------------
# loader = AzureBlobStorageContainerLoader(conn_str="<conn_str>", container="<container>")

# ---------------------
# Azure Blob storage file
# ---------------------
# loader = AzureBlobStorageFileLoader(
#     conn_str="<connection string>",
#     container="<container name>",
#     blob_name="<blob name>",
# )


# ---------------------
# Copy-paste = NEW DOCUMENT
# ---------------------
# text = "..... put the text you copy pasted here......"
# metadata = {"source": "internet", "date": "Friday"}
# doc = Document(page_content=text, metadata=metadata)

# print(doc)


# ---------------------
# Email (via files = .eml or .msg)
# ---------------------

# loader = UnstructuredEmailLoader("example_data/fake-email.eml")
# or
# loader = OutlookMessageLoader("example_data/fake-email.msg")

# ---------------------
# Git
# ---------------------
# Load codebase
# loader = GitLoader(
#     clone_url="https://github.com/hwchase17/langchain",
#     repo_path="./example_data/test_repo2/",
#     branch="master",
# )

# ---------------------
# GitHub
# ---------------------
# Load issues and PRs
# loader = GitHubIssuesLoader(
#     repo="hwchase17/langchain",
#     access_token=ACCESS_TOKEN,  # delete/comment out this argument if you've set the access token as an env var.
#     creator="UmerHA",
# )

# ---------------------
# Google Drive (via API)
# ---------------------
# loader = GoogleDriveLoader(
#     folder_id="1yucgL9WGgWZdM1TOuKkeghlPizuzMYb5",
#     # Optional: configure whether to recursively fetch files from subfolders. Defaults to False.
#     recursive=False,
# )

# ---------------------
# Hugging Face dataset
# ---------------------
# dataset_name = "imdb"
# page_content_column = "text"
# loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)


# ---------------------
# Telegram (via files = JSON)
# ---------------------
# loader = TelegramChatFileLoader("example_data/telegram.json")

# ---------------------
# Twitter (via API)
# ---------------------
# loader = TwitterTweetLoader.from_bearer_token(
#     oauth2_bearer_token="YOUR BEARER TOKEN",
#     twitter_users=["elonmusk"],
#     number_tweets=50,  # Default value is 100
# )

# Or load from access token and consumer keys
# loader = TwitterTweetLoader.from_secrets(
#     access_token='YOUR ACCESS TOKEN',
#     access_token_secret='YOUR ACCESS TOKEN SECRET',
#     consumer_key='YOUR CONSUMER KEY',
#     consumer_secret='YOUR CONSUMER SECRET',
#     twitter_users=['elonmusk'],
#     number_tweets=50,
# )


# ---------------------
# WhatsApp (via files = .txt)
# ---------------------
# loader = WhatsAppChatLoader("example_data/whatsapp_chat.txt")

# ---------------------
# YouTube (via API)
# ---------------------
# loader = YoutubeLoader.from_youtube_url(
#     "https://www.youtube.com/watch?v=QsYGlZkevEg",
#     add_video_info=True,
#     language=["en", "id"],
#     translation="en",
# )

# Google Drive loader


# The LangChain Google Drive Loader efficiently imports data from Google Drive by using the GoogleDriveLoader class. It can fetch data from a list of Google Docs document IDs or a single folder ID.

# Prepare necessary credentials and tokens:

# By default, the GoogleDriveLoader searches for the credentials.json file in ~/.credentials/credentials.json. Use the credentials_file keyword argument to modify this path.
# The token.json file follows the same principle and will be created automatically upon the loader's first use.
# To set up the credentials_file, follow these steps:

# Create a new Google Cloud Platform project or use an existing one by visiting the Google Cloud Console. Ensure that billing is enabled for your project.
# Enable the Google Drive API by navigating to its dashboard in the Google Cloud Console and clicking "Enable."
# Create a service account by going to the Service Accounts page in the Google Cloud Console. Follow the prompts to set up a new service account.
# Assign necessary roles to the service account, such as "Google Drive API - Drive File Access" and "Google Drive API - Drive Metadata Read/Write Access," depending on your needs.
# After creating the service account, access the "Actions" menu next to it, select "Manage keys," click "Add Key," and choose "JSON" as the key type. This generates a JSON key file and downloads it to your computer, which serves as your credentials_file.
# Retrieve the folder or document ID from the URL:

# Folder: https://drive.google.com/drive/u/0/folders/{folder_id}
# Document: https://docs.google.com/document/d/{document_id}/edit



from langchain.document_loaders import GoogleDriveLoader


loader = GoogleDriveLoader(
    folder_id="your_folder_id",
    recursive=False # # Optional: Fetch files from subfolders recursively. Defaults to False.
)

docs = loader.load()


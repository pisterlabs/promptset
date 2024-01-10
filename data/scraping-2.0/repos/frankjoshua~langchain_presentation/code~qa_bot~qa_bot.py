from langchain.document_loaders import BSHTMLLoader

loader = BSHTMLLoader("example_data/fake-content.html")
data = loader.load()
data

async def onMessage(message, callback):
    await callback(f"You asked {message}")
    await callback("Recieved")
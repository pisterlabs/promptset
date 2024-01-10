from langchain.embeddings import OpenAIEmbeddings

# requires `pip install openai`
embeddings_model = OpenAIEmbeddings()

# Embed list of texts
embeddings = embeddings_model.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!"
    ]
)
print(len(embeddings))
"""
5
"""

# length of 1536
print(len(embeddings[0]))
print(len(embeddings[1]))
"""
[... 0.022085255981691917, 0.025720015991018582, 0.008734743689027272, -0.006709843137048811, -0.022764415491192392, -0.00257671800269355, 0.010677894145694868, 0.0001446357869742665, -0.02568228625240111, -0.010438930752548039, -0.002831402818756228, -0.012992066737283132, 0.0015925658455746433, -0.021569597594135712, 0.011853846242120273, 0.015771589535625893, 0.006238204640524732, 0.02429881221167677, 0.014086268272736402, -0.024575506274763608, -0.021129402409104603, 0.007653119664435697, 0.006021250727232698, -0.02475158583889211, -0.012853719705739713, 0.018048030525951612, -0.0018441062839218978, -0.008445472115078739, -0.006885921304193508, 0.00240850043059146, 0.00827568270336489, -0.008030431020448483, -0.004181860777053302, 0.0010344603379206113, 0.007552503768493557, 0.01879007479579295, 0.008451761336170855, -0.014249769394680672, -0.03264995904888929, 0.004728961544779937, -0.0020343339179553677, -0.024927663540375542, -0.006565207350074544, -0.014765427782236877 ...]
"""

# Use Document Loader
from langchain.document_loaders.csv_loader import CSVLoader
loader = CSVLoader(file_path='./csv_sample.csv')
data = loader.load()
print(data)
embeddings = embeddings_model.embed_documents(
    [
        text.page_content for text in data
    ]
)
print(len(embeddings))
"""
9
"""

# Embed single query
# Embed a single piece of text for the purpose of comparing to other embedded pieces of texts.
embedded_query = embeddings_model.embed_query("What was the name mentioned in the conversation?")
print(embedded_query[:5])
"""
[0.005354681365594307, -0.0005715346531097274, 0.03887590993433691, -0.0029596003572924623, -0.00896628532870428]
"""
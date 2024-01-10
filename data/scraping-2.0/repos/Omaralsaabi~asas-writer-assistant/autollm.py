from autollm import AutoQueryEngine, read_files_as_documents
import openai

# Set OpenAI API key
openai.api_key = 'API_KEY'

# Read files as documents
documents = read_files_as_documents('data')

# Create query engine
query_engine = AutoQueryEngine.from_parameters(
    documents=documents,
    llm_params={"model": "gpt-3.5-turbo"}

    )

# Query
query = 'ما هو الحديث الخامس في الأربعون النووية؟'

# Get the answer
answer = query_engine.query(query)

# Print the answer
print(answer.response)


# Output:
"""
عَنْ أم المؤمنين أم عبد الله عَائِشَةَ ﵂ قَالَتْ: قَالَ رَسُولُ اللَّهِ: «مَنْ أَحْدَثَ فِي أَمْرِنَا هَذَا مَا لَيْسَ مِنْهُ فَهُوَ رَدٌّ» (١) رَوَاهُ الْبُخَارِيُّ، وَمُسْلِمٌ.
وَفِي رِوَايَةٍ لِمُسْلِمٍ: «مَنْ عَمِلَ عَمَلًا لَيْسَ عَلَيْهِ أَمْرُنَا فَهُوَ رَدٌّ».
"""
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap
from langserve import RemoteRunnable

openai = RemoteRunnable("http://localhost:8000/openai/")
joke_chain = RemoteRunnable("http://localhost:8000/joke/")

print(joke_chain.invoke({"topic": "parrots"}))
"""
content="Sure, here's a joke for you:\n\nWhy don't scientists trust parrots?\n\nBecause they always give them quack advice!"
"""

prompt = ChatPromptTemplate.from_messages(
    [("system", "Tell me a long story about {topic}")]
)

# Can define custom chains
chain = prompt | RunnableMap({
    "openai": openai
})

print(chain.batch([{ "topic": "parrots" }]))
"""
[{'openai': AIMessage(content='Once upon a time, in a lush tropical rainforest, there lived a colorful and vibrant community of parrots. These parrots were renowned for their intelligence, beauty, and lively personalities. They were the guardians of the forest, spreading joy and laughter with their melodious songs and playful behavior.\n\nAt the heart of the parrot community was a wise and respected elder named Ollie. Ollie had a rich plumage of emerald green feathers, with touches of bright red and blue. His eyes twinkled with an ancient wisdom that had been passed down through generations.\n\nOne day, as the parrots gathered around Ollie, he shared a tale from long ago. It was a story about an ancient treasure hidden deep within the forest. This treasure, known as the "Jewel of the Rainforest," was said to possess magical powers that could bring harmony and balance to all living creatures.\n\nThe parrots were intrigued by the tale and decided to embark on a quest to find the Jewel of the Rainforest. They believed that by harnessing its powers, they could protect their home from any harm that might come its way.\n\nLed by Ollie, the parrots set off on their adventure, flying through the dense foliage, and singing songs of courage and determination. Along the way, they encountered various challenges, but their unity and unwavering spirit kept them going.\n\nAs they ventured deeper into the forest, they faced treacherous rivers, towering trees, and hidden predators. However, their sharp minds and agile wings helped them overcome every obstacle. They supported one another, sharing their knowledge and strength.\n\nDuring their journey, the parrots discovered the incredible diversity of the rainforest. They marveled at the exotic flowers, cascading waterfalls, and the symphony of sounds that filled the air. They also encountered other creatures, like mischievous monkeys swinging from the trees and wise old turtles slowly making their way across the forest floor.\n\nOne day, after months of relentless searching, the parrots stumbled upon a hidden cave. Inside, they found an ethereal glow emanating from a magnificent jewel. It was the Jewel of the Rainforest, shimmering with every color imaginable. The parrots knew they had found what they were looking for.\n\nAs they gathered around the jewel, a magical energy enveloped them. They felt a sense of peace and enlightenment wash over them, as if the forest itself had embraced them. From that moment on, the parrots became the true guardians of the rainforest, using the Jewel\'s powers to protect and preserve their home.\n\nWith the Jewel\'s magic, the parrots brought harmony to the ecosystem, ensuring that all creatures lived in balance and peace. They sang their songs of unity, healing the wounds of the forest and spreading happiness throughout.\n\nWord of the parrots\' noble efforts spread far and wide, attracting visitors from across the world who wanted to witness the wonders of the rainforest. People marveled at the parrots\' intelligence, their ability to mimic human speech, and their vibrant colors.\n\nThe parrots became ambassadors of the rainforest, teaching humans the importance of conservation and the need to protect the Earth\'s natural wonders. Their enchanting presence brought joy to all who encountered them, reminding everyone of the beauty and magic that resided within nature.\n\nAnd so, the parrots lived on, their legacy carried through the generations. They continued to guard the rainforest, ensuring its prosperity and safeguarding the Jewel of the Rainforest. Their story became a legend, passed down through time, reminding everyone of the power of unity, wisdom, and the magic that lies within the hearts of parrots.')}]
"""
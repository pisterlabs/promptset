import openai

import os
from dotenv import load_dotenv

from story_development.setting import Setting

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

conditioning_info = "target audience is a 5 year old boy.The story should help the reader overcome a fear of butterflies."
premise = '"The Butterfly Keeper": In a small village, a young boy named Juan loves catching and collecting butterflies. However, when he learns about the importance of butterflies in the ecosystem, he must reconcile his love of catching them with the need to protect them.'
outline = """Expanded Story Premise:

Characters:
- Juan, a curious and adventurous 5-year-old boy who loves catching butterflies.
- Abuela, Juan's grandmother, who is wise and caring. She is concerned about the butterflies' welfare and wants to teach Juan about their importance.
- The Butterfly Keeper, an expert on butterflies who lives outside the village. She is empathetic and patient and teaches Juan about the importance of butterflies in the ecosystem.

Setting:
- A small village surrounded by lush green forests and meadows.
- A butterfly garden filled with colorful flowers and plants, where Juan spends most of his time catching butterflies.
- The Butterfly Keeper's home, a cottage in the woods, surrounded by a butterfly sanctuary.

Themes:
- Caring for nature and its creatures.
- Empathy and understanding.
- Overcoming fears.

Outline:

Exposition:
- Juan loves catching and collecting butterflies in his butterfly garden. He doesn't understand why his grandmother, Abuela, is concerned about harming the butterflies.
- Abuela tells Juan about the importance of butterflies in the ecosystem, and how they help pollinate plants and flowers. She shows him how to care for the butterflies, and they release them back into the garden.

Rising Action:
- Juan spots a rare butterfly in his garden, which he wants to catch. Despite Abuela's warning, he chases after it but ends up getting hurt and scaring the butterfly away.
- Feeling guilty, Juan decides to learn more about how to care for butterflies. He asks Abuela for help, and she suggests that they visit the Butterfly Keeper to learn from an expert.

Climax:
- At the Butterfly Keeper's home, Juan learns about the different types of butterflies and how they contribute to the ecosystem. He also helps the Butterfly Keeper care for the butterflies and releases them back into their sanctuary.
- However, when Juan encounters a large butterfly, he becomes scared and runs away, leaving the butterfly in danger. The Butterfly Keeper tells Juan that having a fear of butterflies is okay, but he must learn to empathize with them and respect their place in nature.

Falling Action:
- Juan realizes his fear of butterflies stems from not knowing enough about them. He apologizes to the Butterfly Keeper and asks to learn more.
- The Butterfly Keeper teaches Juan how to gently hold and care for the butterflies, and Juan gains a newfound appreciation and understanding for these beautiful creatures.

Resolution:
- Juan returns home and shows Abuela all that he has learned while caring for the butterflies. He promises to protect and respect them, and never harm them again.
- The story ends with Juan and Abuela sitting in the butterfly garden, watching the beautiful creatures flutter around them, feeling grateful for all that they have learned."""
setting = Setting(conditioning_info, premise, outline)
setting.score(verbose=True, n=3)
recommendations = setting.make_recommendations(1, verbose=True)
[print(recommendation) for recommendation in recommendations]

import openai

class Summarizer:
    def __init__(self, original_txt, openai_key, language):
        self.original_txt = original_txt
        openai.api_key = openai_key
        self.language = language

    def start(self):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Summarize this paragraph & give the output in {self.language} Language: '{self.original_txt}'",
            max_tokens=100,  # Adjust the number of tokens as per your needs
            temperature=0.5,  # Adjust the temperature as per your preference
            stop=None,
            n=1
        )
        summarized_txt = response.choices[0].text.strip()
        return summarized_txt

if __name__ == "__main__":
    openai_key = "sk-Ksl0bXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXhxgBNZNHvpOrqhrXygo"

    original_text = """
    Once upon a time in a quaint little village nestled amidst rolling hills, there lived a young girl named Lily. She had an insatiable curiosity and an unwavering passion for adventure. Every day, Lily would explore the enchanting forests that surrounded her village, seeking hidden treasures and magical encounters.

One sunny morning, as Lily ventured deeper into the woods, she stumbled upon a mystical clearing bathed in golden sunlight. In the center stood an ancient oak tree, its gnarled branches reaching toward the heavens. Curiosity piqued, Lily approached the tree and noticed a small, shimmering key hanging from one of its branches.

Intrigued by the mysterious key, Lily wondered what it unlocked. Determined to uncover its secret, she embarked on a thrilling quest. Following a faint trail of clues, she ventured through treacherous caves, over glistening rivers, and across towering bridges. Along the way, she encountered talking animals, friendly fairies, and wise old sages who shared their wisdom and offered guidance.

Finally, after overcoming countless challenges, Lily arrived at a hidden door nestled deep within a forgotten ruin. With bated breath, she inserted the key into the lock. The door creaked open, revealing a dazzling realm brimming with breathtaking beauty and wonder.

Lily had discovered the realm of dreams, a place where imagination and reality intertwined. She marveled at the surreal landscapes, vibrant colors, and fantastical creatures that roamed freely. Every step she took filled her heart with joy and her mind with boundless inspiration.

In this magical realm, Lily realized the power of her dreams. She vowed to cherish her creativity and nurture her adventurous spirit. With newfound confidence, she returned to her village, sharing tales of her extraordinary journey and inspiring others to embrace their own dreams.

From that day forward, Lily became known as the village's beloved storyteller. Her stories transported people to realms of magic and wonder, igniting their imaginations and fueling their own dreams.

And so, the young girl who once sought adventure in the woods became the catalyst for countless dreams, reminding everyone that within their hearts, the power to create their own extraordinary stories resided.
    """

    language = "hindi"
    summarize = Summarizer(original_text, openai_key, language)
    summarized_txt = summarize.start()
    print(summarized_txt)

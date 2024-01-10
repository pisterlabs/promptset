import os
import random
import openai
from twilio.rest import Client
import modal
from sqlitedict import SqliteDict

stub = modal.Stub("magical")
data = modal.NetworkFileSystem.persisted("magical-db2")

image = modal.Image.debian_slim().pip_install("sqlitedict", "openai", "twilio")


def create_magic():
    openai.api_key = os.environ["OPENAI_API_KEY"]
    prompt = """
This is a document of 1000 creative, unique, and amazing 1-line magical realism stories written by a master writer. Here they begin. They are all in a similar style and format, but each has a unique twist.

1. A fortune teller turns over a tarot card with a milkshake on it. 'You will be killed by a Belgian soldier,' he says to you.
2. A prince owns an ice sculpture which depicts every humiliating defeat that has ever happened.
3. A tired woman falls into a lagoon filled with sexual fantasies. Nobody notices.
4. A busboy has a terrible headache. He discovers there is a peach inside his frontal lobe.
5. A sailor is being chased by a kitten made of skin.
6. An archangel hears of a cemetery inside a tropical island, and wants to visit it.
7. An office manager possesses a unique skill: He can sense the presence of vampires.
8. An HR manager gets married to a shark inside a silver cemetery.
9. An alchemist discovers that endometriosis does not exist, and is secretly pleased.
10. A schoolmistress steals breakfast and hides it inside a crystal galaxy.
11. There is a library in Barcelona where you can borrow opium poppies instead of books.
12. A Persian novelist discovers a pair of spectacles which let him see every act of treachery happening around the world.
13. A depressed maharajah falls in love with a textbook.
14. A porn star switches bodies with a hazel tree.
15. A beautiful woman with grey eyes is hiding in an opera house. She is thinking about hexagons. A witch is checking Facebook behind her.
16. A midwife invents a new method of divination based on the movement of turtles.
17. Apple Inc is researching how to generate feelings of bliss.
18. You are walking through a ruined city. You can hear the faint sound of an oboe.
19. A swan is whispering in the ear of Taylor Swift. It says: "It will never work."
20. A sorceress declares war on Britney Spears.
21. A deeply depressed woman spends eleven months writing a book about rock and roll.
22. A Canadian opera singer hears an orchestra that sounds like queer theory.
23. There is a storm in Tokyo. It rains knives.
24. A Turkish literary critic spends her whole life writing an epic poem about icicles.
25. A fox and a baroness are getting a divorce.
26. An Ethiopian book describes an amethyst that gives you power over metaphysics.
27. By thinking of an iceberg, a 99-year-old viceroy is able to destroy the universe.
28. An Anatolian politician goes for a walk in a forest and discovers a chrysanthemum made of climate change.
29. A necromancer swallows a jade city.
30. You find out that you are a congresswoman trapped in a haunted panopticon.
31. A pharmacist sees an albatross with Bruegel's Fall of Icarus painted on its wings.
32. A eucalyptus tree has the power to transport you to a cinema in Seattle.
33. A library is haunted by the ghost of a bank manager. He was murdered by a giant squid.
34. Every astronomer in Shanghai is grateful.
35. A skirt causes anyone who wears it to turn into a stomach.
36. A senator finds a map which shows the location of every palindrome in Kyoto.
37. An Iraqi president owns a cathedral that is filled with higher education.
38. A zucchini as beautiful as the Milky Way grows in a Tibetan garden. A field marshal plots to steal it.
39. A pianist finds a telescope that lets him talk to the Enlightenment.
40. A meerkat is traveling from Canada to a pine forest.
41. A woolly mammoth is eating an albatross.
42. A girl is having sexual intercourse with consciousness.
43. A salamander made of polar bears is born in Iceland.
44. Candy apples, shopping malls, and a prison. That is all I have to say.
45. A stone-hearted anthropologist falls in love with despair.
46. An admiral and a geisha play a game in which the contestants must strangle epistemology.
47. A travel agent in Oregon flies to Dar es Salaam on her lunch break.
48. In Vancouver is a poet who has a crucifix instead of an ear.
49. An ancient Spartan fisherman appears in a concert hall in California. He is breathing heavily.
50. An Austrian queen becomes lost in a glass house of mirrors. When she emerges, she realizes she is a software developer from Baltimore.
51. An Algerian TV station airs a show about pizzas that lasts for 265 million years.
52. A video game designer writes a poem about every act of sexual intercourse that has taken place in New Zealand.
53. A politician sees a duchess with tortoise shells instead of eyes.
54. On New Year's Eve, a sentence appears above a Spanish pyramid. It reads: "I hope you like radishes."
55. A governess falls pregnant with winter.
56.
""".strip()

    cmpl = openai.Completion.create(
        engine="davinci-002",
        prompt=prompt,
        temperature=1,
        max_tokens=100,
        stop=["\n", "57."],
        best_of=3,
        n=1
    )

    return cmpl.choices[0].text.strip() # type: ignore


@stub.function(
    network_file_systems={"/data": data},
    image=image,
    secret=modal.Secret.from_dotenv()
)
def send_text(text: str):
    client = Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])

    db = SqliteDict("/data/magical.db", autocommit=True)
    db["text"] = db.get("text", []) + [text]

    num = len(db["text"])

    numbers = os.environ["MAGICAL_NUMBERS"].split(",")

    for number in numbers:
        message = client.messages.create(
            to=number,
            from_="+18333081599",
            body=f"{num}. {text}")

        print("sent text: ", text, " to ", numbers, " at ", message.sid)

@stub.function(
    schedule=modal.Cron("13 15 * * *"),
    image=image,
    secret=modal.Secret.from_dotenv()
)
def send_magic(dry_run: bool = False):
    print("Making magic...")
    text = create_magic()
    print("Got text: ", text)
    if not dry_run:
        send_text.call(text)


if __name__ == "__main__":
    # test magic
    with stub.run():
        send_magic.call(dry_run=True)

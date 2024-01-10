from pathlib import Path
from openai import OpenAI

client = OpenAI()

parts = [
    """
"Superhero for a Day"

I. Introduction

One morning in the suburb of Pinegrove, Evan Reed sprang up from his bed, feeling a spark of energy buzzing through his veins, electrifying him from head to toe. His body felt different, charged, potent, and then something miraculous began to unfold— He flourished his fingers and found himself shooting webs across his room, much like his superhero idol, Spider-Man!

"Whoa!" he exclaimed, wide-eyed and breathless, his heart pounding like a drum. In the mirror, he noticed his ordinary reflection interspersed with flickers of a superhero's kicky suit. Questions filled his mind. Why did this happen? How long would it last? What could he do with the powers? But most importantly, could he be a superhero for a day, just like Spider-Man?

Overwhelmed with excitement and a dash of bewilderment, he decided to test out his newfound powers. Here was a chance to live his dream, to swap the comic books' vibrant still-images with living, breathing, adrenaline-surging action.

II. Act One: Discovering Powers

The discovery phase was a montage of trail and error, interspersed with wild laughter and occasional groans. Shooting webs? Check. Climbing walls? Absolutely! Heightened senses? Most certainly! He could hear the neighbor’s dog barking, smell the aroma of freshly baked cookies from the bakery around the block, and see clearly across to the next street.

As Evan experimented with his powers, his room turned into a chaotic battleground of spewed webs and flying toys. His beloved teddy bear ended up dangling from the ceiling fan, and even Spider-Man wall decals gate-crashed into each other. He glanced around at the hilarious mess he'd made in his room, laughing heartily at the sight of his teddy rotating on the fan, its beady eyes rolling dizzyingly. After rescuing his teddy from its unplanned amusement ride, Evan carried on with his exploration of powers, growing more adept with each attempt.

A day in the life of a superhero was unfolding, and Evan's thrilling adventure was just beginning!
""",
    """
III. Act Two: First Task as a Superhero

With his newfound abilities, Evan was eager to take on his first task as a superhero. Suddenly, something was brought to his attention. The once bustling local park now appeared lifeless and neglected. Its once vibrant jungle gyms, blooming flowerbeds, and welcoming picnic tables had now become rusty, withered, and battered from lack of maintenance. He decided that restoring the park's lost cheer would be the perfect mission to accomplish.

With immense enthusiasm, serenaded by the sound of chalk against paper as he chalked out his plan, Evan quickly fell into a series of multitasks. While wielding his homework in one hand like an indomitable young scholar, his other was busy sketching out strategies to restore the park on his trusty blue tablet, all while keeping his secret identity under wraps.

There were superheroes, and then there was Evan. An eight-year-old boy, juggling school chores while planning restoration of a park, coming to grips with his secret identity - the charm and humor were undeniable!

IV. Act Three: A Hero's Challenges

Evan soon found out that being a superhero wasn't all flying through the skies and defeating evil. It was hard work! As he snuck out to the park, donning a camouflage cap, he launched his solo mission. After an epic fail attempt at cleaning up the park single-handedly - involving a trash bin, a webline, and an unfortunate tumble - Evan realized he had underestimated the task.

He attempted to use his superpowers to pull weeds out of the garden patches. Instead, he yanked the park's oldest tree out by its roots, almost revealing his secret to the startled passerby. With an awkward laugh and a hasty excuse, he managed to wave off their suspicions.

Through his misadventures, he learned an important lesson; superheroes were not just about power - it required a lot of responsibility and hard work. Tough as it was, Evan found himself ridiculously excited for what was to come next in his superhero saga.
""",
    """
V. Climax: Learning the True Meaning of Heroism

Exhausted and a little frustrated, Evan sat on the park bench, his new superhero suit feeling heavier on his shoulders. He found himself wondering if being a superhero was really worth all the trouble. As he sighed, looking at the park's half-cleaned state, he met Mr. Gilbert, an elderly man who was known as a park regular.

Hearing stories about the park's heydays from Mr. Gilbert, where children used to run around laughing, families picnicking under the shady trees, and friends playing catch, something sparked within Evan. The park wasn't just a space; it was a legacy of shared joy and community.

With a soft smile, he realised that his superpower was not just his Spider-Man like abilities. It was his will to make a difference, his wish to fill the park with blooming flowers of joy again. Bringing happiness to others, and to build a sense of community, was just as heroic. It was then that Evan understood the true essence of heroism.

VI. Resolution: Becoming the Neighborhood Hero

With newfound determination, Evan decided to solve the park problem, not with his superpowers, but with unity. He approached his friends, school, and neighborhood, seeking their help to restore the park. And soon enough, the community rallied together, breaking the inertia of inaction. The sight was heartwarming - children picking up litter, adults painting the withered benches, and everyone planting new flowers.

With every patch of green cleared, every flower that bloomed, the joy in the community grew. A once neglected park transformed into a beaming symbol of unity and shared enthusiasm. Evan, watching this as he sat on one of the newly-painted benches, felt a sense of fulfillment wash over him. He had done it! He was indeed the neighbourhood's superhero, but not of the sort he had imagined, a much better one.
""",
]

speech_file_path = Path(__file__).parent / "part2.mp3"
response = client.audio.speech.create(
    model="tts-1",
    voice="onyx",
    input=parts[2],
)

response.stream_to_file(speech_file_path)

# script to generate a list of random words
import json
import os
import random
from copy import deepcopy
from typing import List
import re

import openai

from semantiq.embeddings import load_default_embeddings, cosine_similarity, embedding

WORDS = '''
cat, dog, house, car, book, tree, sun, moon, star, lamp, chair, table, pen, paper, pencil, computer, 
phone, flower, butterfly, bird, fish, rock, river, ocean, mountain, road, cloud, grass, shoe, hat, 
shirt, dress, pants, sock, glove, wallet, key, clock, spoon, fork, knife, plate, cup, bowl, towel, 
soap, toothbrush, toothpaste, mirror, window, door, bed, pillow, blanket, picture, music, movie, 
game, puzzle, paint, brush, camera, microphone, guitar, piano, drum, soap, milk, bread, egg, cheese, 
apple, orange, banana, strawberry, lemon, lime, grape, cherry, cucumber, tomato, potato, carrot, lettuce, 
onion, garlic, rice, pasta, pizza, burger, fries, ice cream, chocolate, candy, cookie, cake, sugar, salt, pepper,
innovation, discovery, adventure, laughter, kindness, creativity, harmony, tranquility, wisdom, ambition, freedom, 
courage, integrity, passion, loyalty, honesty, resilience, compassion, generosity, optimism, empathy, tranquility, 
discipline, dedication, sincerity, spontaneity, serenity, perseverance, patience, curiosity, gratitude, humility, 
ambition, endurance, inspiration, perspective, integrity, determination, enthusiasm, courage, diversity, justice, 
peace, vibrancy, faith, devotion, dignity, elegance, empathy, forgiveness, gratitude, hope, humility, joy, love, 
motivation, optimism, pride, respect, trust, unity, wisdom, zest, balance, beauty, focus, growth, satisfaction, 
success, change, discipline, gratitude, curiosity, determination, peace, tranquility, passion, victory, versatility, 
vibrant, wonder, accomplishment, adaptability, assertiveness, authenticity, bravery, calmness, charity, 
cheerfulness, commitment, compassion, confidence, cooperation, creativity, credibility, dynamism, efficiency, 
empathy, flexibility, happiness, harmony, hope, independence, inspiration, integrity, joy, kindness, love, 
loyalty, motivation, patience, persistence, positivity, resilience, respect, responsibility, simplicity, sincerity, 
spontaneity, strength, trust, understanding, unity, wisdom, zeal, balance, innovation, respect, motivation, optimism, 
gratitude, determination, humility, discipline, satisfaction, confidence, patience, courage, success, passion, 
perseverance, focus, laughter, generosity, curiosity, ambition, tranquility, joy, creativity, harmony, wisdom, 
freedom, integrity, loyalty, resilience, kindness, empathy, spontaneity, serenity, dignity, elegance, pride, 
trust, accomplishment, adaptability, assertiveness, authenticity, bravery, calmness, charity, cheerfulness, 
commitment, cooperation, dynamism, efficiency, flexibility, happiness, hope, independence, inspiration, love, 
persistence, positivity, responsibility, simplicity, strength, understanding, unity, zeal,
table, chair, computer, book, pen, camera, phone, clock, lamp, window, door, carpet, bed, blanket, pillow, mirror, 
television, radio, fridge, stove, oven, toaster, kettle, mug, plate, spoon, fork, knife, bowl, glass, bottle, 
bag, backpack, purse, wallet, key, watch, bracelet, necklace, ring, shoe, sock, hat, coat, shirt, pants, dress, 
skirt, sweater, glove, scarf, belt, umbrella, bicycle, car, truck, bus, boat, airplane, train, dog, cat, bird, 
fish, horse, cow, sheep, pig, chicken, rabbit, lion, tiger, bear, elephant, monkey, giraffe, zebra, kangaroo, 
dolphin, rose, tulip, daisy, sunflower, tree, bush, grass, leaf, fruit, vegetable, apple, banana, orange, strawberry, 
blueberry, pineapple, grape, cherry, tomato, cucumber, carrot, potato, onion, bread, cheese, butter, egg, milk, juice, 
water, tea, coffee, sugar, salt, pepper, rice, pasta, chocolate, cake, cookie, ice cream, pizza, sandwich, hamburger, 
soup, salad, music, art, dance, sports, soccer, basketball, baseball, tennis, golf, swimming, movie, game, puzzle, 
book, newspaper, magazine, photograph, painting, drawing, sculpture, map, ticket, money, coin, dollar, heart, star, 
moon, sun, rainbow, cloud, rain, snow, wind, fire, river, lake, ocean, mountain, forest, desert, island, city, town, 
village, street, park, garden, school, hospital, store, restaurant, hotel, library, museum, zoo, farm, beach, bridge, 
tower, castle, book, dog, cat, flower, phone, house, tree, chair, bike, mountain, river, cloud, bed, pen, computer, 
moon, sun, star, dress, camera, necklace, song, movie, game, coffee, tea, ocean, painting, hat, bookshelf, car, bus, 
school, elephant, pizza, sandwich, guitar, violin, key, door, window, jewelry, wristwatch, butterfly, rainbow, cupcake, 
umbrella, fire, snow, suitcase, heart, rose, diamond, kitchen, garden, forest, candle, blanket, pineapple, strawberry, 
laptop, bread, mirror, soap, rain, spoon, beach, kite, museum, zoo, park, apple, baby, castle, bread, watch, photo, 
refrigerator, sculpture, pillow, map, ring, candy, perfume, airplane, volcano, hammock, lipstick, turtle, socks, 
shark, bracelet, dragon, spider, robot, dinosaur, ghost, cowboy, pirate, vampire, detective, astronaut, mermaid, 
superhero, princess, alien, wizard, zombie, knight, fairy, soldier, ninja, scar, angel, cowboy, tiger, lion, eagle, 
owl, parrot, dolphin, whale, kangaroo, bear, wolf, squirrel, rabbit, fox, giraffe, chameleon, panda, cheetah, zebra, 
monkey, raccoon, leopard, peacock, flamingo, swan, pigeon, dove, sparrow, duck, turkey, chicken, goldfish, penguin, 
octopus, lobster, jellyfish, bee, snail, ladybug, ant, cricket, grasshopper, earthworm, turtle, snake, frog, alligator, 
crocodile, mosquito, bat, hedgehog, beaver, horse, donkey, deer, buffalo, hamster, guinea pig, mouse, rat, bat, skunk, 
otter, seal, walrus, platypus, raccoon, porcupine, koala, armadillo, hummingbird, cardinal, bluebird, toucan, 
woodpecker, ostrich, canary, budgie, falcon, hawk, dodo, rooster, pheasant,
adventure, mystery, creativity, passion, universe, dream, courage, harmony, wisdom, paradise, freedom, masterpiece, 
destiny, fascination, magic, miracle, curiosity, sensation, rebellion, illusion, romance, treasure, eternity, 
nostalgia, pleasure, fantasy, infinity, serendipity, euphoria, legend,
sunset, galaxy, melody, velvet, laughter, echo, whisper, illusion, silhouette, labyrinth, oasis, kaleidoscope, 
euphoria, infinity, paradox, serendipity, hurricane, labyrinth, nostalgia, mirage, twilight, chimera, solstice, 
autumn, symphony, horizon, passion, daydream, silhouette, wildfire, mountain, sunrise, river, laughter, galaxy, 
forest, rainbow, melody, dream, adventure, mystery, passion, freedom, harmony, butterfly, wildfire, echo, infinity, 
journey, twilight, whisper, horizon, solitude, miracle, poetry, masterpiece, eclipse, heartbeat, comet, silhouette,
moon, bicycle, rainbow, dragon, ocean, city, garden, volcano, mirror, elephant, castle, forest, robot, violin, pirate, 
ghost, universe, tornado, diamond, eagle, pizza, warrior, jellyfish, skyscraper, galaxy, river, book, cactus, 
butterfly, spaceship, waterfall, dinosaur, snowflake, wizard, zebra, ballet, chocolate, sphinx, treasure, festival, 
compass, mermaid, sunflower, labyrinth, island, dream, mammoth, kangaroo, carnival, sunrise, honey, statue, gypsy, 
desert, fairy, astronaut, labyrinth, compass, phoenix, avalanche, meadow, comet, ninja, hurricane, glacier, waterfall, 
rainbow, lighthouse, crystal, dolphin, rhinoceros, cyborg, chocolate, skyscraper, diamond, rose, snowflake, daisy, 
raccoon, parrot, sunflower, tarantula, tornado, cactus, unicorn, mammoth, warrior, dragon, garden, forest, castle, 
ocean, universe, ghost, pirate, violin, robot, city, bicycle, moon, dolphin, zebra, avalanche, comet
'''


def get_all_words() -> List[str]:
    # words deduplicated and split using a regexp
    return list(set(re.split(r'\s*,\s*', WORDS.strip())))


SIZE_PER_GROUP = 3


def random_puzzle():
    # sample 8 words without replacement
    words = get_all_words()
    random.shuffle(words)
    return {
        'groupPos': sorted(words[:SIZE_PER_GROUP], key=lambda x: len(x)),
        'groupNeg': sorted(words[SIZE_PER_GROUP:SIZE_PER_GROUP*2], key=lambda x: len(x)),
    }


def find_closest_words(puzzle):
    # print which of the word from group one is closest to which of the words of group2
    group_neg = puzzle['groupNeg']
    group_pos = puzzle['groupPos']
    most_similar = None
    max_similarity = 0
    for w1 in group_neg:
        for w2 in group_pos:
            sim = cosine_similarity(embedding(w1), embedding(w2))
            # print(f'{w1} {w2} {sim}')
            if sim > max_similarity:
                max_similarity = sim
                most_similar = (w1, w2)
    return most_similar, max_similarity


def main():
    random.seed(0)
    i = 0
    while i < 10000:
        puzzle = random_puzzle()
        puzzle['id'] = i
        most_similar, max_similarity = find_closest_words(puzzle)
        if max_similarity > 0.82:
            print(f'Skipping due to similarity {max_similarity} ({most_similar})')
        else:
            print(f'Writing puzzle #{i} {puzzle}')
            with open(os.path.join(os.path.dirname(__file__), f'../puzzles/{i}.json'), 'w') as f:
                json.dump(puzzle, f)
            i += 1


if __name__ == '__main__':
    main()

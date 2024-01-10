import random 

# from openai playground:
# Generate a list of English greeting templates where the person to be greeted
# is represented by the letter X. Example: "hello, X!"
#
# 1. Hi, X!
# 2. Hey there, X!
# 3. Greetings, X!
# 4. Howdy, X!
# 5. Good day, X!
# 6. Hiya, X!
# 7. Salutations, X!
# 8. What's up, X?
# 9. Yo, X!
# 10. Good morning/afternoon/evening, X!

choices = [
    "Hi, X!",
    "Hey there, X!",
    "Greetings, X!",
    "Howdy, X!",
    "Good day, X!",
    "Hiya, X!",
    "Salutations, X!",
    "What's up, X?",
    "Yo, X!",
    " Good morning/afternoon/evening, X!",
]


# this is the function we're going to export
def greet(whom):
    return random.choice(choices).replace("X", whom)


# just to get a 2nd function to export
def shuffle():
    random.shuffle(choices)


# test it on the command line
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: [python] greets.py <whom>")
        sys.exit(1)

    whom = sys.argv[1]
    print(greet(whom))

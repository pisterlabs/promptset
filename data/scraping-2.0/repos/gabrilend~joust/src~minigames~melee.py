import openai
import random

openai.api_key = "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"
#engine_type = "text-ada-001"
engine_type = "text-davinci-003"

class Melee:
    
    def __init__(self, town="", reward="", description=""):
        if reward == "":
            possible_rewards=[random.randint(500, 1500),
                              random.randint(500, 1500),
                              random.randint(500, 1500),
                              random.randint(500, 1500),
                              0]
            self.reward = random.choice(possible_rewards)
        self.description = description
        if self.description == "":
                self.description = "There isn't a duel happening now."
        self.town = town

    def __str__(self):
        
        my_string = f"There is a grand melee in the town of {self.town}." + \
                 f" The reward is {self.reward} gold pieces."
        if self.reward == 0:
            my_string += " The combatants are just playing for fun!"

        return my_string

    def test_melee(self, A, B):
        print("This is a test of the melee functionality.")
        print("This is the description of the melee:")
        print(self)
        print("This is a test run of the melee with no arguments:")
        self.main_melee(A, B)

    def main_melee(self, A, B):
        
        #FIXME - add name generation capabilites to "character" script
        #        make it genericized and ask for a class, then filter
        #        based on the value you pass it and pass it different
        #        parameters depending on class
        
        #FIXME
        #prompt = self.__setup_melee(A, B)

        print(f"You are at the melee. {A.name} versus {B.name}")
        
        while(True):
            self.user_input = self.generate_die_rolls()


            if self.roll_to_hit(self.user_input[0], B.ac):
                self.resolve_damage(self.user_input[1], B)
                A_missed = False
            else:
                A_missed = True
            if self.roll_to_hit(self.user_input[2], A.ac):
                self.resolve_damage(self.user_input[3], A)
                B_missed = False
            else:
                B_missed = True

            if A_missed and B_missed:
                print(self.describe_stalemate())
            else:
                print(f"{A.name} rolled a {self.user_input[0]} to hit and " + \
                      f"{self.user_input[1]} for damage.")
                print(f"{B.name} rolled a {self.user_input[2]} to hit and " + \
                      f"{self.user_input[3]} for damage.\n")
                print(f"{A.name} has an AC of {A.ac}")
                print(f"{B.name} has an AC of {B.ac}\n")
                print(f"{A.name} has {A.health} hitpoints.")
                print(f"{B.name} has {B.health} hitpoints.")

                self.describe_characters(A, B)
            
            input("< PUSH ENTER TO CONTINUE >")

            if A.health <= 0 or B.health <= 0:
                result = self.end_melee(A.health, B.health)
                if result == "a":
                    print(f"{A.name} won the melee!")
                elif result == "b":
                    print(f"{B.name} won the melee!")
                break
            else:
                continue

    def __setup_melee(self, A, B):
        #FIXME - add some description of the melee
        print("The melee is grand and honorable. Who will succeed on this day?")


    def generate_die_rolls(self):
        attack_roll = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, \
                       12, 13, 14, 15, 16, 17, 18, 19, 20]
        damage_roll = [1, 2, 3, 4, 5, 6]

        user_attack = random.choice(attack_roll)
        user_damage = random.choice(damage_roll)
        comp_attack = random.choice(attack_roll)
        comp_damage = random.choice(damage_roll)

        return (user_attack, user_damage, comp_attack, comp_damage)

    def roll_to_hit(self, roll, ac):

        if roll > ac:
            return True
        else:
            return False

    def resolve_damage(self, damage, character):
        
        character.health -= damage

    def describe_stalemate(self):
        
        #FIXME
        prompt = "Two combatants are fighting in a medieval duel. Armored and" \
                 + " bearing great two handed weapons, they fight for honor " \
                 + "and valor. The two catch their breath, as they search for" \
                 + " an opening in their opponent's defences. For now, there " \
                 + "is a stalemate. Describe the two combatants as they circle"\
                 + " one another:\n\n"


        response = openai.Completion.create(engine=engine_type, \
                                            prompt=prompt, \
                                            max_tokens = 256, \
                                            temperature = 1)
        self.status = response.choices[0].text.strip()
        return self.status

    def describe_characters(self, A, B):

        prompt = self.__generate_descriptions(A, B)
        print(f"Here's how {A.name} is feeling:\n")
        print(prompt[0])
        print(f"Here's how {B.name} is feeling:\n")
        print(prompt[1])
        
    def __generate_descriptions(self, A, B):
        
        a_prompt = A.generate_status()
        b_prompt = B.generate_status()

        return (a_prompt, b_prompt)

    def end_melee(self, A_health, B_health):
        print("The melee is finished. Both combatants bow their heads and " + \
              "relinquish the thrill of valor.")
        if A_health > B_health:
            return "a"
        elif B_health > A_health:
            return "b"
        else:
            return "draw"


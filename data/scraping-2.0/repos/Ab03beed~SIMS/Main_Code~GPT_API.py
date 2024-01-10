import os
import openai

class GPT_API:
    #default constructor
    def __init__(self):
        # Set OpenAI API key through system varibales
        openai.api_key = os.getenv("GPT_API_KEY")

        # Lists of possible transcriptions for each box to handle misinterpretations
        # or different pronunciations/accent variations
        self.box_1_variants = [ 
            "box one", "box 1", "box won", "barks one", "fox one", "books one",  
            "boks one", "bok one", "boxen one", "box on", "bax one", "bux one", 
            "boks ett", "bok ett", "box ett", "bex one", "bogs one", "bog one", 
            "buck one", "buck ett", "bokk one", "bokks one", "boxen ett", "box on ett", 
            "boks on", "boxen on", "bok on", "boxen won", "boxen ett", "bokks won",  
            "bokks on", "bokks ett", "bex won", "bex on", "bex ett", "volkswagen" ,"1"
        ] 
        self.box_2_variants = [ 
            "box 2", "books to","box two", "box 2", "box to", "box too", "barks two", "fox 2"," fox two", "books two", 
            "boks two", "bok two", "boxen two", "box tu", "bax two", "bux two", 
            "boks två", "bok två", "box två", "bex two", "bogs two", "bog two", 
            "buck two", "buck två", "bokk two", "bokks two", "boxen två", "box tu två", 
            "boks tu", "boxen tu", "bok tu", "boxen too", "boxen två", "bokks too",  
            "bokks tu", "bokks två", "bex too", "bex tu", "bex två" ,"2"
        ] 
        self.box_3_variants = [ 
            "box three", "box 3", "barks three", "fox three", "books three", 
            "boks three","books 3", "bok three", "boxen three", "box tree", "bax three", "bux three", 
            "boks tre", "bok tre", "box tre", "bex three", "bogs three", "bog three", 
            "buck three", "buck tre", "bokk three", "bokks three", "boxen tre", "box tree tre", 
            "boks tree", "boxen tree", "bok tree", "boxen trey", "boxen tre", "bokks tree",  
            "bokks trey", "bokks tre", "bex tree", "bex trey", "bex tre", "3" 
        ] 
        self.box_4_variants = [
            "books four","books 4","box 4.","box 4.","box 4","box for", "bucks for","box four","fox four", "box before","box floor","pox for", "blocks for",
            "barks for","box for the","books for","box before the", "boss for","boat's for","box for the win",
            "bucks for the","boxing for","box before four","box for you","backs for","boxing four","box of four",
            "box or four","boxing for the","box it for","bucks for the win","boxed for","books for the","box it before",
            "box or the four", "4"
        ]


    def _gptCall(self, which_box):
        gpt_call = gpt_call = openai.ChatCompletion.create( 
            model="gpt-4", 
            temperature=0,
            messages=[ 
        {"role": "system", "content": "You are an experienced robot operations coder that will help the user to code a collaborative robot."}, 
        {"role": "user", "content": f"""
        Imagine we are working with a collaborative robot with the task of moving four boxes from a "grabbing table" to a "release table".  
        The four boxes is called BOX_1, BOX_2 and BOX_3 and BOX_4. 
        
        The coordinate (XYZ) to grab boxes: BOX_1(90,-220,245), BOX_2(90,-400,245), BOX_3(-90,-400,245), BOX_4(-90,-220,245).  .  
         
        The the cordinate (XYZ) to release boxes: BOX_1(90, 400, 245), BOX_2(90, 220, 245), BOX_3(-90, 220, 245), BOX_4(-90, 400, 245).
        
       When going to and from grab and release positions, the robot arm should avoid collision with other boxes by first visiting these coordinates:
        collision avoidance coordinates when grabbing:BOX_1(90,-220,465), BOX_2(90,-400,465), BOX_3(-90,-400,465), BOX_4(-90,-220,465)
        collision avoidance coordinates when releasing: BOX_1(90, 400, 465), BOX_2(90, 220, 465), BOX_3(-90, 220, 465), BOX_4(-90, 400, 465)

        The home position (XYZ) for the robot arm is: (270,0,504).
         
        The program should always start and end with the robot arm going to its home position.
         
        *The functions you can use are: 
            -go_to_location(X,Y,Z): Moves robot arm end effector to a location specified by XYZ coordinates. Returns nothing. 
            -grab(): Robot end effector grabs box. Returns nothing. 
            -release(): Robot arm end effector releases box. 
        
        Please have the robot move {which_box} from its pick-up position to its release-position. Return the order in how functions are used, without any comments. 
        Like this:
        1. function() 
        2. function() 
        .
     
        """}

            ]
        ) 
            
        return gpt_call

    def ask(self, task):
        
        # Check which box the user referred to in their voice command
        if any(variant in task for variant in self.box_1_variants): 
            which_box = "BOX_1" 
        elif any(variant in task for variant in self.box_2_variants): 
            which_box = "BOX_2" 
        elif any(variant in task for variant in self.box_3_variants): 
            which_box = "BOX_3" 
        elif any(variant in task for variant in self.box_4_variants): 
            which_box = "BOX_4" 
        else: 
            print("Invalid box name in voice command.") 
            exit()

        gpt_call = self._gptCall(which_box)


        return gpt_call['choices'][0]['message']['content']


    
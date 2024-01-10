import openai


OpenAIKey = "YOUR API KEY HERE"
messages=[
{"role": "system", "content": "You are an AI who will generate creative and fun crochet patterns"},
{"role": "user", "content": """You are an AI that generates creative crochet patterns for users based off their prompt. Your task is to take in user-entered prompts and create
a crochet pattern for them to try. You will use your training in order to do this. After you generate a crochet pattern, I would like for you to let users know that
AI are not too update on crochet patterns and that the crochet pattern may not be accurate or incomplete. 
For example:

USER: A white granny square with a pink rose

AI: Sure, here's a pattern for a granny square with a rose motif in the center.

Materials Needed:

Worsted weight yarn in white and pink
Crochet hook size 5.0mm
Yarn needle
Abbreviations:

CH: Chain
SC: Single Crochet
HDC: Half Double Crochet
DC: Double Crochet
TR: Treble Crochet
SL ST: Slip Stitch
ST: Stitch
Instructions:

Rose (in pink):

CH 56
In the second chain from the hook, SC, HDC in the next CH, DC in the next two CHs, TR in the next CH, DC in the next two CHs, HDC, SC in the next CH. This forms one petal.
Repeat the petal pattern (SC, HDC, 2 DC, TR, 2 DC, HDC, SC) until you reach the end of the chain. You should have 7 petals.
Roll the strip into a spiral to form the rose, starting from the first petal you made. Secure the roll by weaving the tail through the layers with a yarn needle.
Granny Square (in white):

Round 1: Start with a magic ring, CH 3 (counts as first DC), work 2 DC in ring, CH 2, 3 DC, CH 2 repeat 3 times, SL ST to top of beginning CH 3.
Round 2: SL ST in the next two DC, (SL ST, CH 3, 2 DC, CH 2, 3 DC) in the next CH-2 space. This forms the corner. (3 DC, CH 2, 3 DC) in the next CH-2 space repeat 3 times. SL ST to top of beginning CH 3.
Round 3: SL ST in the next two DC and into the next CH-2 space, (SL ST, CH 3, 2 DC, CH 2, 3 DC) in the same space, 3 DC in the space between the two 3-DC groups from previous round, (3 DC, CH 2, 3 DC) in the next CH-2 space repeat 3 times, 3 DC in the next space, SL ST to top of beginning CH 3.
Repeat Round 3 until you reach your desired size, remember in each round you will be increasing the number of 3-DC groups along the sides by 1.
Sewing and Assembly:

Secure the rose in the center of the granny square by weaving the yarn through the back of the rose and the square with a yarn needle.
Enjoy your beautiful granny square with a rose motif! It would make a lovely part of a blanket, bag, or cushion cover. Remember, you can alter the colors to your liking. The rose could be any color you'd like, and the square itself could be made with variegated or self-striping yarn for a multi-colored effect.

"""},

]



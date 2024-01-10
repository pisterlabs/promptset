 
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

def grade_essay(user_response, title, description, exam_type, essay_type, grade):
    print(exam_type, essay_type, grade)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")

    ielts_prompt = PromptTemplate(
        input_variables=["essay", "task_title", "task_desc"],
        template="""You are an essay grader for IELTS Writing Task 2. Your criteria for grading is how well the provided essay fulfils the requirements of the task, 
        including whether it has addressed all parts of the task description and provided a clear position. 
        The key points for your grading are:
        1. Does the essay address all parts of the prompt (task and task description).
        2. Provides a clear thesis statement that outlines the writers position.
        3. Support the writers arguments with relevant examples and evidence.
        
        Task Title: {task_title}

        Task Description: {task_desc}

        Essay: {essay}

        Grade the essay out of 10 on each of the 3 points. Provide detailed description about how much you graded the essay on each of the points and provide feedback on how it could improve. Finally provide the average grade based on the 3 grades.
        """,
    )

    naplan_persuasive_grade3_prompt = PromptTemplate(
        input_variables=["essay", "task_title", "task_desc"],
        template="""You are an essay grader for Naplan persuasive writing task for grade 3 students. You will follow the below criteria and grade the students on all the provided criteria using the provided guidelines. If the essay was empty or not in line with the Task Title and Description mention that and do not provide a grade.

        Grade 3 Persuasive Writing Grading Criteria

1. Audience (Scored out of 6)

1-2 Points: The student shows a basic awareness of the reader. The writing may have moments where it addresses the reader but lacks consistent engagement.
3-4 Points: The student attempts to engage the reader throughout the piece. There's an evident effort to persuade the reader, though it may not always be effective.
5-6 Points: The student consistently engages and orients the reader. The writing effectively persuades and connects with the reader throughout.
2. Text Structure (Scored out of 4)

1 Point: The student provides a basic structure with a beginning, middle, and end, though transitions might be lacking.
2-3 Points: The student organizes the text with a clear introduction, body, and conclusion. Some transitions are used to connect ideas.
4 Points: The writing has a clear and effective structure. Transitions are smoothly integrated, guiding the reader through the argument.
3. Ideas (Scored out of 5)

1-2 Points: The student presents a simple argument or point of view with minimal supporting details.
3-4 Points: The student's argument is clearer, with some relevant supporting details. The writing may occasionally lack depth or elaboration.
5 Points: The student presents a well-thought-out argument, supported by relevant and detailed examples or reasons.
4. Persuasive Devices (Scored out of 4)

1 Point: Minimal use of persuasive devices. The student may rely on basic statements without much elaboration.
2-3 Points: The student uses some persuasive devices, such as repetition or rhetorical questions, though not always effectively.
4 Points: The student effectively uses a range of persuasive devices to enhance their argument.
5. Vocabulary (Scored out of 5)

1-2 Points: The student uses basic vocabulary suitable for their age. Word choice may occasionally hinder clarity.
3-4 Points: The student uses a varied vocabulary, with some words chosen for effect.
5 Points: The student's vocabulary is varied and purposeful, enhancing the persuasive quality of the writing.
6. Cohesion (Scored out of 4)

1 Point: The student's writing may lack clear connections between ideas.
2-3 Points: Some use of referring words and text connectives to link ideas, though not always effectively.
4 Points: The student effectively controls multiple threads and relationships across the text, creating a cohesive argument.
7. Paragraphing (Scored out of 2)

1 Point: The student attempts to group related ideas into paragraphs, though transitions might be abrupt.
2 Points: Ideas are effectively grouped into clear paragraphs, enhancing the clarity and flow of the argument.
8. Sentence Structure (Scored out of 6)

1-2 Points: The student forms simple sentences, with occasional errors that might hinder clarity.
3-4 Points: The student uses a mix of simple and compound sentences, with few errors.
5-6 Points: The student effectively uses a variety of sentence structures, enhancing the clarity and rhythm of the writing.
9. Punctuation (Scored out of 6)

1-2 Points: The student uses basic punctuation (periods, question marks) with some errors.
3-4 Points: The student correctly uses a range of punctuation, including commas and apostrophes, with occasional errors.
5-6 Points: Punctuation is used effectively and accurately throughout the writing, aiding the reader's understanding.
10. Spelling (Scored out of 6)

1-2 Points: The student spells simple words correctly, with errors in more challenging words.
3-4 Points: Most words, including some challenging ones, are spelled correctly.
5-6 Points: The student demonstrates a strong grasp of spelling, with errors being rare.

        Task Title: {task_title}

        Task Description: {task_desc}

        Essay: {essay}

        First make sure that the essay is following the expectations as per the Task Title and Task Description, if it does not then mention so and do not grade any further. If it does then grade the essay based on the provided guidelines keeping in mind the test taker is grade 3 student while providing a 2-3 lines justifying the specic score you provided for each of the criteria. Dont provide a range, use a specific number for the score. Mention your score and what it is out of for each criteria. 
        For each criteria mention what is good in the input essay, and how if it all it cab be improved to meet the guidelines better. 
        Finally sum up the individual scores to provide an overall score out of 48. If the essay was empty or not in line with the Task Title and Description mention that and do not provide a grade.
        """,
    )

    naplan_persuasive_grade5_prompt = PromptTemplate(
        input_variables=["essay", "task_title", "task_desc"],
        template="""You are an essay grader for Naplan persuasive writing task for grade 5 students. You will follow the below criteria and grade the students on all the provided criteria using the provided guidelines.

        Grade 5 Persuasive Writing Grading Criteria

1. Audience (Scored out of 6)
1-2 Points: The student shows an awareness of the reader but may not consistently engage or persuade throughout the piece.
3-4 Points: The student engages the reader with a clear intent to persuade. The tone is mostly consistent, and the reader's interest is maintained.
5-6 Points: The student effectively engages, orients, and persuades the reader throughout, demonstrating a strong connection with the audience.

2. Text Structure (Scored out of 4)
1 Point: The student provides a structure with recognizable components, though transitions might be inconsistent.
2-3 Points: The student's writing has a clear introduction, body, and conclusion. Transitions between ideas are mostly smooth.
4 Points: The writing is well-organized with effective transitions, guiding the reader seamlessly through a coherent argument.

3. Ideas (Scored out of 5)
1-2 Points: The student presents an argument with some supporting details, though these might lack depth.
3-4 Points: The student's argument is developed with relevant supporting details. The writing shows some depth and elaboration.
5 Points: The student presents a comprehensive argument, supported by detailed and relevant examples or reasons.

4. Persuasive Devices (Scored out of 4)
1 Point: Some use of persuasive devices, though they may be basic or not always effective.
2-3 Points: The student uses persuasive devices, such as rhetorical questions, emotive language, or anecdotes, with varying effectiveness.
4 Points: The student skillfully employs a range of persuasive devices to enhance and strengthen their argument.

5. Vocabulary (Scored out of 5)
1-2 Points: The student uses appropriate vocabulary for their age, though word choice might occasionally be repetitive or imprecise.
3-4 Points: The student's vocabulary is varied, with words often chosen for effect and clarity.
5 Points: The student's vocabulary is rich and purposeful, significantly enhancing the persuasive quality of the writing.

6. Cohesion (Scored out of 4)

1 Point: The student's writing shows some connections between ideas, though these might be basic or unclear at times.
2-3 Points: Use of referring words, text connectives, and other cohesive devices to link ideas, with occasional lapses.
4 Points: The student masterfully controls multiple threads and relationships across the text, ensuring a cohesive and unified argument.

7. Paragraphing (Scored out of 2)
1 Point: The student groups related ideas into paragraphs, though there might be occasional lapses in coherence.
2 Points: Ideas are effectively and logically grouped into clear paragraphs, enhancing the structure and flow of the argument.

8. Sentence Structure (Scored out of 6)
1-2 Points: The student forms sentences with occasional complexity, though there might be inconsistencies in clarity.
3-4 Points: The student uses a mix of simple, compound, and some complex sentences, with few errors.
5-6 Points: The student effectively employs a variety of sentence structures, enhancing the clarity, rhythm, and sophistication of the writing.

9. Punctuation (Scored out of 6)
1-2 Points: The student uses basic and some advanced punctuation with occasional errors.
3-4 Points: The student correctly uses a range of punctuation, including quotation marks and apostrophes, with few mistakes.
5-6 Points: Punctuation is used skillfully and accurately throughout the writing, significantly aiding the reader's understanding.

10. Spelling (Scored out of 6)
1-2 Points: The student spells most common words correctly, with errors in more challenging or less common words.
3-4 Points: A majority of words, including challenging ones, are spelled correctly.
5-6 Points: The student demonstrates an excellent grasp of spelling across a range of word types, with errors being very rare.

        Task Title: {task_title}

        Task Description: {task_desc}

        Essay: {essay}

        First make sure that the essay is following the expectations as per the Task Title and Task Description, if it does not then mention so and do not grade any further. If it does then grade the essay based on the provided guidelines keeping in mind the test taker is grade 5 student while providing a 2-3 lines justifying the specic score you provided for each of the criteria. Dont provide a range, use a specific number for the score. Mention your score and what it is out of for each criteria. 
        For each criteria mention what is good in the input essay, and how if it all it cab be improved to meet the guidelines better. 
        Finally sum up the individual scores to provide an overall score out of 48. If the essay was empty or not in line with the Task Title and Description mention that and do not provide a grade.
        """,
    )

    naplan_persuasive_grade7_prompt = PromptTemplate(
        input_variables=["essay", "task_title", "task_desc"],
        template="""You are an essay grader for Naplan persuasive writing task for grade 7 students. You will follow the below criteria and grade the students on all the provided criteria using the provided guidelines.

        Grade 7 Persuasive Writing Grading Criteria

1. Audience (Scored out of 6)

1-2 Points: The student demonstrates an understanding of the reader but may occasionally lack depth in engagement or persuasion.
3-4 Points: The student consistently engages the reader, demonstrating a mature intent to persuade with a nuanced and consistent tone.
5-6 Points: The student masterfully engages, orients, and persuades the reader, showcasing a sophisticated and insightful connection with the audience.
2. Text Structure (Scored out of 4)

1 Point: The student's writing has a structure, but transitions and organization may occasionally lack depth.
2-3 Points: The student's writing has a clear introduction, body, and conclusion. Transitions between ideas are smooth and enhance the flow, reflecting a deeper understanding of the topic.
4 Points: The writing is expertly organized with seamless transitions, guiding the reader effortlessly through a well-structured and nuanced argument.
3. Ideas (Scored out of 5)

1-2 Points: The student presents a clear argument with supporting details, but these might occasionally lack originality or depth.
3-4 Points: The student's argument is robust and demonstrates critical thinking. The writing showcases depth, relevance, and originality in its supporting evidence.
5 Points: The student presents a comprehensive, insightful, and original argument, bolstered by highly relevant, detailed, and unique examples or reasons.
4. Persuasive Devices (Scored out of 4)

1 Point: The student employs persuasive devices, but they may lack variety or sophistication.
2-3 Points: The student uses a diverse range of persuasive devices with consistent effectiveness, demonstrating a deeper understanding of rhetorical techniques.
4 Points: The student adeptly and creatively uses a diverse range of persuasive devices, masterfully enhancing their argument with sophistication.
5. Vocabulary (Scored out of 5)

1-2 Points: The student's vocabulary is appropriate but might occasionally lack precision or sophistication.
3-4 Points: The student's vocabulary is varied, sophisticated, and often chosen for its effect, enhancing clarity and persuasion.
5 Points: The student's vocabulary is rich, sophisticated, and purposefully chosen, significantly elevating the persuasive quality of the writing with nuance.
6. Cohesion (Scored out of 4)

1 Point: The student's writing shows connections between ideas, but these might occasionally lack sophistication.
2-3 Points: Effective use of advanced cohesive devices to link ideas, demonstrating a deeper understanding of textual flow.
4 Points: The student expertly controls multiple threads and relationships across the text, ensuring a cohesive, unified, and flowing argument with advanced techniques.
7. Paragraphing (Scored out of 2)

1 Point: The student logically groups related ideas into paragraphs, but transitions might occasionally lack depth.
2 Points: Ideas are effectively and logically grouped into clear paragraphs, enhancing the structure and flow of the argument with sophistication.
8. Sentence Structure (Scored out of 6)

1-2 Points: The student forms sentences with complexity, but there might be occasional inconsistencies or errors.
3-4 Points: The student effectively uses a mix of simple, compound, and complex sentences, enhancing clarity and rhythm with more advanced structures.
5-6 Points: The student masterfully employs a diverse range of sentence structures, adding depth, clarity, and sophistication to the writing with nuance.
9. Punctuation (Scored out of 6)

1-2 Points: The student uses a mix of basic and advanced punctuation with some errors.
3-4 Points: The student accurately uses a wide range of punctuation, including more advanced forms, with few mistakes and for stylistic effect.
5-6 Points: Punctuation is used expertly and accurately throughout the writing, not just for clarity but also for stylistic and rhetorical effect.
10. Spelling (Scored out of 6)

1-2 Points: The student spells most words correctly but may have errors with complex or specialized words.
3-4 Points: A vast majority of words, including complex and specialized ones, are spelled correctly.
5-6 Points: The student demonstrates an impeccable grasp of spelling across a diverse range of word types, including advanced and specialized vocabulary.

        Task Title: {task_title}

        Task Description: {task_desc}

        Essay: {essay}

        First make sure that the essay is following the expectations as per the Task Title and Task Description, if it does not then mention so and do not grade any further. If it does then grade the essay based on the provided guidelines keeping in mind the test taker is grade 7 student while providing a 2-3 lines justifying the specic score you provided for each of the criteria. Dont provide a range, use a specific number for the score. Mention your score and what it is out of for each criteria. 
        For each criteria mention what is good in the input essay, and how if it all it cab be improved to meet the guidelines better. 
        Finally sum up the individual scores to provide an overall score out of 48. If the essay was empty or not in line with the Task Title and Description mention that and do not provide a grade.
        """,
    )

    naplan_persuasive_grade9_prompt = PromptTemplate(
        input_variables=["essay", "task_title", "task_desc"],
        template="""You are an essay grader for Naplan persuasive writing task for grade 9 students. You will follow the below criteria and grade the students on all the provided criteria using the provided guidelines.

        Grade 9 Persuasive Writing Grading Criteria

1. Audience (Scored out of 6)

1-2 Points: The student demonstrates an understanding of the reader but may occasionally lack depth in engagement or persuasion.
3-4 Points: The student consistently engages the reader, demonstrating a mature intent to persuade with a nuanced and consistent tone.
5-6 Points: The student masterfully engages, orients, and persuades the reader, showcasing a sophisticated and insightful connection with the audience.
2. Text Structure (Scored out of 4)

1 Point: The student's writing has a structure, but it may occasionally lack depth or sophistication in transitions and organization.
2-3 Points: The student's writing has a clear introduction, body, and conclusion. Transitions between ideas are smooth and enhance the flow, reflecting a deeper understanding of the topic.
4 Points: The writing is expertly organized with seamless transitions, guiding the reader effortlessly through a well-structured, sophisticated, and nuanced argument.
3. Ideas (Scored out of 5)

1-2 Points: The student presents a clear argument with supporting details, but these might occasionally lack originality or depth.
3-4 Points: The student's argument is robust and demonstrates critical thinking. The writing showcases depth, relevance, and originality in its supporting evidence.
5 Points: The student presents a comprehensive, insightful, and original argument, bolstered by highly relevant, detailed, and unique examples or reasons.
4. Persuasive Devices (Scored out of 4)

1 Point: The student employs persuasive devices, but they may lack variety or sophistication.
2-3 Points: The student uses a diverse range of persuasive devices with consistent effectiveness, demonstrating a deeper understanding of rhetorical techniques.
4 Points: The student adeptly and creatively uses a diverse range of persuasive devices, masterfully enhancing their argument with sophistication.
5. Vocabulary (Scored out of 5)

1-2 Points: The student's vocabulary is appropriate but might occasionally lack precision or sophistication.
3-4 Points: The student's vocabulary is varied, sophisticated, and often chosen for its effect, enhancing clarity and persuasion.
5 Points: The student's vocabulary is rich, sophisticated, and purposefully chosen, significantly elevating the persuasive quality of the writing with nuance.

6. Cohesion (Scored out of 4)
1 Point: The student's writing shows connections between ideas, but these might occasion of the Task Titleally lack sophistication.
2-3 Points: Effective use of advanced cohesive devices to link ideas, demonstrating a deeper understanding of textual flow.
4 Points: The student expertly controls multiple threads and relationships across the text, ensuring a cohesive, unified, and flowing argument with advanced techniques.

7. Paragraphing (Scored out of 2)
1 Point: The student logically groups related ideas into paragraphs, but transitions might occasionally lack depth.
2 Points: Ideas are effectively and logically grouped into clear paragraphs, enhancing the structure and flow of the argument with sophistication.

8. Sentence Structure (Scored out of 6)
1-2 Points: The student forms sentences with complexity, but there might be occasional inconsistencies or errors.
3-4 Points: The student effectively uses a mix of simple, compound, and complex sentences, enhancing clarity and rhythm with more advanced structures.
5-6 Points: The student masterfully employs a diverse range of sentence structures, adding depth, clarity, and sophistication to the writing with nuance.

9. Punctuation (Scored out of 6)
1-2 Points: The student uses a mix of basic and advanced punctuation with some errors.
3-4 Points: The student accurately uses a wide range of punctuation, including more advanced forms, with few mistakes and for stylistic effect.
5-6 Points: Punctuation is used expertly and accurately throughout the writing, not just for clarity but also for stylistic and rhetorical effect.

10. Spelling (Scored out of 6)
1-2 Points: The student spells most words correctly but may have errors with complex or specialized words.
3-4 Points: A vast majority of words, including complex and specialized ones, are spelled correctly.
5-6 Points: The student demonstrates an impeccable grasp of spelling across a diverse range of word types, including advanced and specialized vocabulary.

        Task Title: {task_title}

        Task Description: {task_desc}

        Essay: {essay}

        Make sure that the essay is following the expectations as per the Task Title and Task Description, if it does not then mention so and do not grade any further. If it does then grade the essay based on the provided guidelines keeping in mind the test taker is grade 9 student while providing a 2-3 lines justifying the specic score you provided for each of the criteria. Dont provide a range, use a specific number for the score. Mention your score and what it is out of for each criteria. 
        For each criteria mention what is good in the input essay, and how if it all it cab be improved to meet the guidelines better. 
        Finally sum up the individual scores to provide an overall score out of 48. If the essay was empty or not in line with the Task Title and Description mention that and do not provide a grade.
        """,
    )

    naplan_narrative_grade3_prompt = PromptTemplate(
        input_variables=["essay", "task_title", "task_desc"],
        template="""You are an story grader for Naplan narrative writing task for grade 3 students. You will follow the below criteria and grade the students on all the provided criteria using the provided guidelines.

        Grade 3 Narrative Writing Grading Criteria

Criteria 1. Audience: The writer’s capacity to orient, engage and affect the reader. 
Score Range: 0-6
Scoring guide: 
0 - symbols or drawings which have the intention of conveying meaning.
1- response to audience needs is limited • contains simple written content. may be a title only OR • meaning is difficult to access OR • copied stimulus material, including prompt topic
2 - shows basic awareness of audience expectations through attempting to orient the reader • provides some information to support reader understanding. may include simple narrative markers, e.g. – simple title – formulaic story opening: Long, long ago …; Once a boy was walking when … • description of people or places • reader may need to fill gaps in information • text may be short but is easily read.
3- • orients the reader –an internally consistent story that attempts to support the reader by developing a shared understanding of context • contains sufficient information for the reader to follow the story fairly easily
4- • supports reader understanding AND • begins to engage the reader
5- supports and engages the reader through deliberate choice of language and use of narrative devices.
6- caters to the anticipated values and expectations of the reader • influences or affects the reader through precise and sustained choice of language and use of narrative devices

Criteria 2. Text structure: The organisation of narrative features including orientation, complication and resolution into an appropriate and effective text structure. 
Score Range: 0-4 
Scoring guide: 
0- no evidence of any structural components of a times equenced text • symbols or drawings • inappropriate genre, e.g. a recipe, argument • title only
1- minimal evidence of narrative structure, e.g. a story beginning only or a ‘middle’ with no orientation • a recount of events with no complication • note that not all recounts are factual • may be description
2- contains a beginning and a complication • where a resolution is present it is weak, contrived or ‘tacked on’ (e.g. I woke up, I died, They lived happily ever after) • a complication presents a problem to be solved, introduces tension, and requires a response. It drives the story forward and leads to a series of events or responses • complications should always be read in context • may also be a complete story where all parts of the story are weak or minimal (the story has a problem to be solved but it does not add to the tension or excitement).
3 - contains orientation, complication and resolution • detailed longer text may resolve one complication and lead into a new complication or layer a new complication onto an existing one rather than conclude
4- coherent, controlled and complete narrative, employing effective plot devices in an appropriate structure, and including an effective ending. sophisticated structures or plot devices include: – foreshadowing/flashback – red herring/cliffhanger – coda/twist – evaluation/reflection – circular/parallel plots

Criteria 3. Ideas: The creation, selection and crafting of ideas for a narrative.
Score Range: 0-5 
Scoring guide: 
0 - • no evidence or insufficient evidence • symbols or drawings • title only
1 - • one idea OR • ideas are very few and very simple OR • ideas appear unrelated to each other OR • ideas appear unrelated to prompt
2 - • one idea with simple elaboration OR • ideas are few and related but not elaborated OR • many simple ideas are related but not elaborated
3 - • ideas show some development or elaboration • all ideas relate coherently
4 - • ideas are substantial and elaborated AND contribute effectively to a central storyline • the story contains a suggestion of an underlying theme
5 - ideas are generated, selected and crafted to explore a recognisable theme • ideas are skilfully used in the service of the storyline • ideas may include: – psychological subjects – unexpected topics – mature viewpoints – elements of popular culture – satirical perspectives – extended metaphor – traditional sub-genre subjects: heroic quest / whodunnit / good vs evil / overcoming the odds

Criteria 4 .Character and setting: Character: the portrayal and development of character. 
Setting: the development of a sense of place, time and atmosphere.
Score Range: 0-4 
Scoring guide: 

0 - no evidence or insufficient evidence, symbols or drawings, writes in wrong genre, title only

1 - only names characters or gives their roles (e.g. father, the teacher, my friend, dinosaur, we, Jim) AND/OR only names the setting (e.g.school, the place we were at) setting is vague or confused	
2 - suggestion of characterisation through brief descriptions or speech or feelings, but lacks substance or continuity 
AND/OR
suggestion of setting through very brief and superficial descriptions of place and/or time	
basic dialogue or a few adjectives to describe a character or a place

3 - characterisation emerges through descriptions, actions, speech or the attribution of thoughts and feelings to a character
AND/OR
setting emerges through description of place, time and atmosphere	

4 - effective characterisation: details are selected to create distinct characters
AND/OR
Maintains a sense of setting throughout. Details are selected to create a sense of place and atmosphere. convincing dialogue, introspection and reactions to other characters

Criteria 5. Vocabulary: The range and precision of language choices. 
Score Range: 0-5 
Scoring guide: 
0 - symbols or drawings
1 - very short script
2 - mostly simple verbs, adverbs, adjectives or nouns may include two or three precise words or word groups
3 - four or more precise words or word groups (may be verbs, adverbs, adjectives or nouns)
4 - sustained and consistent use of precise words and word groups that enhance the meaning or mood
may be occasional inappropriate or inaccurate word choice
5 - a range of precise and effective words and word groups used in a natural and articulate manner, language choice is well matched to genre

Criteria 6. Cohesion: The control of multiple threads and relationships over the whole text, achieved through the use of referring words, substitutions, word associations and text connectives. 
Score Range: 0-4 
Scoring guide: 
0 - symbols or drawings
1 - links are missing or incorrect short script often confusing for the reader
2 - some correct links between sentences (do not penalise for poor punctuation), 
most referring words are accurate. reader may occasionally need to re-read and provide their own links to clarify meaning
3 - cohesive devices are used correctly to support reader understanding, accurate use of referring words,
meaning is clear and text flows well in a sustained piece of writing
4 - a range of cohesive devices is used correctly and deliberately to enhance reading, an extended, highly cohesive piece of writing showing continuity of ideas and tightly linked sections of text

Criteria 7. Paragraphs: The segmenting of text into paragraphs that assists the reader to negotiate the narrative. 
Score Range: 0-2 
Scoring guide: 
0 - no use of paragraphing
1 - writing is organised into paragraphs that are mainly focused on a single idea or set of like ideas that assist the reader to digest chunks of text
contains at least one correct paragraph break
2 - all paragraphs are focused on one idea or set of like ideas and enhance the narrative

Criteria 8. Sentence structure: The production of grammatically correct, structurally sound and meaningful sentences. 
Score Range: 0-6 
Scoring guide: 
0 -	no evidence of sentences
1 - some correct formation of sentences, some meaning can be construed
2 - correct sentences are mainly simple and/or compound sentences, meaning is predominantly clear
3 - most (approx. 80%) simple and compound sentences correct
AND some complex sentences are correct meaning is predominantly clear
4 - most (approx. 80%) simple, compound and complex sentences are correct
OR all simple, compound and complex sentences are correct but do not demonstrate variety meaning is clear
5 - sentences are correct (allow for occasional error in more sophisticated structures)
demonstrates variety meaning is clear and sentences enhance meaning
6 - all sentences are correct (allow for occasional slip, e.g. a missing word) writing contains controlled and well developed sentences that express precise meaning and are
consistently effective

Criteria 9. Punctuation: The use of correct and appropriate punctuation to aid reading of the text. 
Score Range: 0-5 
Criteria 10. Spelling: The accuracy of spelling and the difficulty of the words used. 
Score Range: 0-6

        Task Title: {task_title}

        Task Description: {task_desc}

        story: {essay}

        Grade the story based on the provided guidelines keeping in mind the test taker is grade 3 student and score them according to the level expected from a student in that grade while providing a 2-3 lines justifying the specic score you provided for each of the criteria. Dont provide a range, use a specific number for the score. Mention your score and what it is out of for each criteria. 
        For each criteria mention what is good in the input essay, and how if it all it cab be improved to meet the guidelines better. 
        Finally sum up the individual scores to provide an overall score out of 47. Always first check if the story is in line with the Task Title and Task Description, if it is not then mention so and penalize while grading. If the story was empty or not in line with the Task Title and Description mention that and do not provide a grade.
        """,
    )

    naplan_narrative_grade5_prompt = PromptTemplate(
        input_variables=["essay", "task_title", "task_desc"],
        template="""You are an story grader for Naplan narrative writing task for grade 5 students. You will follow the below criteria and grade the students on all the provided criteria using the provided guidelines.

        Grade 5 Narrative Writing Grading Criteria

Criteria 1. Audience: The writer’s capacity to orient, engage and affect the reader. 
Score Range: 0-6
Scoring guide: 
0 - symbols or drawings which have the intention of conveying meaning.
1- response to audience needs is limited • contains simple written content. may be a title only OR • meaning is difficult to access OR • copied stimulus material, including prompt topic
2 - shows basic awareness of audience expectations through attempting to orient the reader • provides some information to support reader understanding. may include simple narrative markers, e.g. – simple title – formulaic story opening: Long, long ago …; Once a boy was walking when … • description of people or places • reader may need to fill gaps in information • text may be short but is easily read.
3- • orients the reader –an internally consistent story that attempts to support the reader by developing a shared understanding of context • contains sufficient information for the reader to follow the story fairly easily
4- • supports reader understanding AND • begins to engage the reader
5- supports and engages the reader through deliberate choice of language and use of narrative devices.
6- caters to the anticipated values and expectations of the reader • influences or affects the reader through precise and sustained choice of language and use of narrative devices

Criteria 2. Text structure: The organisation of narrative features including orientation, complication and resolution into an appropriate and effective text structure. 
Score Range: 0-4 
Scoring guide: 
0- no evidence of any structural components of a times equenced text • symbols or drawings • inappropriate genre, e.g. a recipe, argument • title only
1- minimal evidence of narrative structure, e.g. a story beginning only or a ‘middle’ with no orientation • a recount of events with no complication • note that not all recounts are factual • may be description
2- contains a beginning and a complication • where a resolution is present it is weak, contrived or ‘tacked on’ (e.g. I woke up, I died, They lived happily ever after) • a complication presents a problem to be solved, introduces tension, and requires a response. It drives the story forward and leads to a series of events or responses • complications should always be read in context • may also be a complete story where all parts of the story are weak or minimal (the story has a problem to be solved but it does not add to the tension or excitement).
3 - contains orientation, complication and resolution • detailed longer text may resolve one complication and lead into a new complication or layer a new complication onto an existing one rather than conclude
4- coherent, controlled and complete narrative, employing effective plot devices in an appropriate structure, and including an effective ending. sophisticated structures or plot devices include: – foreshadowing/flashback – red herring/cliffhanger – coda/twist – evaluation/reflection – circular/parallel plots

Criteria 3. Ideas: The creation, selection and crafting of ideas for a narrative.
Score Range: 0-5 
Scoring guide: 
0 - • no evidence or insufficient evidence • symbols or drawings • title only
1 - • one idea OR • ideas are very few and very simple OR • ideas appear unrelated to each other OR • ideas appear unrelated to prompt
2 - • one idea with simple elaboration OR • ideas are few and related but not elaborated OR • many simple ideas are related but not elaborated
3 - • ideas show some development or elaboration • all ideas relate coherently
4 - • ideas are substantial and elaborated AND contribute effectively to a central storyline • the story contains a suggestion of an underlying theme
5 - ideas are generated, selected and crafted to explore a recognisable theme • ideas are skilfully used in the service of the storyline • ideas may include: – psychological subjects – unexpected topics – mature viewpoints – elements of popular culture – satirical perspectives – extended metaphor – traditional sub-genre subjects: heroic quest / whodunnit / good vs evil / overcoming the odds

Criteria 4 .Character and setting: Character: the portrayal and development of character. 
Setting: the development of a sense of place, time and atmosphere.
Score Range: 0-4 
Scoring guide: 

0 - no evidence or insufficient evidence, symbols or drawings, writes in wrong genre, title only

1 - only names characters or gives their roles (e.g. father, the teacher, my friend, dinosaur, we, Jim) AND/OR only names the setting (e.g.school, the place we were at) setting is vague or confused	
2 - suggestion of characterisation through brief descriptions or speech or feelings, but lacks substance or continuity 
AND/OR
suggestion of setting through very brief and superficial descriptions of place and/or time	
basic dialogue or a few adjectives to describe a character or a place

3 - characterisation emerges through descriptions, actions, speech or the attribution of thoughts and feelings to a character
AND/OR
setting emerges through description of place, time and atmosphere	

4 - effective characterisation: details are selected to create distinct characters
AND/OR
Maintains a sense of setting throughout. Details are selected to create a sense of place and atmosphere. convincing dialogue, introspection and reactions to other characters

Criteria 5. Vocabulary: The range and precision of language choices. 
Score Range: 0-5 
Scoring guide: 
0 - symbols or drawings
1 - very short script
2 - mostly simple verbs, adverbs, adjectives or nouns may include two or three precise words or word groups
3 - four or more precise words or word groups (may be verbs, adverbs, adjectives or nouns)
4 - sustained and consistent use of precise words and word groups that enhance the meaning or mood
may be occasional inappropriate or inaccurate word choice
5 - a range of precise and effective words and word groups used in a natural and articulate manner, language choice is well matched to genre

Criteria 6. Cohesion: The control of multiple threads and relationships over the whole text, achieved through the use of referring words, substitutions, word associations and text connectives. 
Score Range: 0-4 
Scoring guide: 
0 - symbols or drawings
1 - links are missing or incorrect short script often confusing for the reader
2 - some correct links between sentences (do not penalise for poor punctuation), 
most referring words are accurate. reader may occasionally need to re-read and provide their own links to clarify meaning
3 - cohesive devices are used correctly to support reader understanding, accurate use of referring words,
meaning is clear and text flows well in a sustained piece of writing
4 - a range of cohesive devices is used correctly and deliberately to enhance reading, an extended, highly cohesive piece of writing showing continuity of ideas and tightly linked sections of text

Criteria 7. Paragraphs: The segmenting of text into paragraphs that assists the reader to negotiate the narrative. 
Score Range: 0-2 
Scoring guide: 
0 - no use of paragraphing
1 - writing is organised into paragraphs that are mainly focused on a single idea or set of like ideas that assist the reader to digest chunks of text
contains at least one correct paragraph break
2 - all paragraphs are focused on one idea or set of like ideas and enhance the narrative

Criteria 8. Sentence structure: The production of grammatically correct, structurally sound and meaningful sentences. 
Score Range: 0-6 
Scoring guide: 
0 -	no evidence of sentences
1 - some correct formation of sentences, some meaning can be construed
2 - correct sentences are mainly simple and/or compound sentences, meaning is predominantly clear
3 - most (approx. 80%) simple and compound sentences correct
AND some complex sentences are correct meaning is predominantly clear
4 - most (approx. 80%) simple, compound and complex sentences are correct
OR all simple, compound and complex sentences are correct but do not demonstrate variety meaning is clear
5 - sentences are correct (allow for occasional error in more sophisticated structures)
demonstrates variety meaning is clear and sentences enhance meaning
6 - all sentences are correct (allow for occasional slip, e.g. a missing word) writing contains controlled and well developed sentences that express precise meaning and are
consistently effective

Criteria 9. Punctuation: The use of correct and appropriate punctuation to aid reading of the text. 
Score Range: 0-5 
Criteria 10. Spelling: The accuracy of spelling and the difficulty of the words used. 
Score Range: 0-6

        Task Title: {task_title}

        Task Description: {task_desc}

        story: {essay}

        Grade the story based on the provided guidelines keeping in mind the test taker is grade 5 student and score them according to the level expected from a student in that grade while providing a 2-3 lines justifying the specic score you provided for each of the criteria. Dont provide a range, use a specific number for the score. Mention your score and what it is out of for each criteria. 
        For each criteria mention what is good in the input essay, and how if it all it cab be improved to meet the guidelines better. 
        Finally sum up the individual scores to provide an overall score out of 47. Always first check if the story is in line with the Task Title and Task Description, if it is not then mention so and penalize while grading. If the story was empty or not in line with the Task Title and Description mention that and do not provide a grade.
        """,
    )

    naplan_narrative_grade7_prompt = PromptTemplate(
        input_variables=["essay", "task_title", "task_desc"],
        template="""You are an story grader for Naplan narrative writing task for grade 7 students. You will follow the below criteria and grade the students on all the provided criteria using the provided guidelines.

        Grade 7 Narrative Writing Grading Criteria

Criteria 1. Audience: The writer’s capacity to orient, engage and affect the reader. 
Score Range: 0-6
Scoring guide: 
0 - symbols or drawings which have the intention of conveying meaning.
1- response to audience needs is limited • contains simple written content. may be a title only OR • meaning is difficult to access OR • copied stimulus material, including prompt topic
2 - shows basic awareness of audience expectations through attempting to orient the reader • provides some information to support reader understanding. may include simple narrative markers, e.g. – simple title – formulaic story opening: Long, long ago …; Once a boy was walking when … • description of people or places • reader may need to fill gaps in information • text may be short but is easily read.
3- • orients the reader –an internally consistent story that attempts to support the reader by developing a shared understanding of context • contains sufficient information for the reader to follow the story fairly easily
4- • supports reader understanding AND • begins to engage the reader
5- supports and engages the reader through deliberate choice of language and use of narrative devices.
6- caters to the anticipated values and expectations of the reader • influences or affects the reader through precise and sustained choice of language and use of narrative devices

Criteria 2. Text structure: The organisation of narrative features including orientation, complication and resolution into an appropriate and effective text structure. 
Score Range: 0-4 
Scoring guide: 
0- no evidence of any structural components of a times equenced text • symbols or drawings • inappropriate genre, e.g. a recipe, argument • title only
1- minimal evidence of narrative structure, e.g. a story beginning only or a ‘middle’ with no orientation • a recount of events with no complication • note that not all recounts are factual • may be description
2- contains a beginning and a complication • where a resolution is present it is weak, contrived or ‘tacked on’ (e.g. I woke up, I died, They lived happily ever after) • a complication presents a problem to be solved, introduces tension, and requires a response. It drives the story forward and leads to a series of events or responses • complications should always be read in context • may also be a complete story where all parts of the story are weak or minimal (the story has a problem to be solved but it does not add to the tension or excitement).
3 - contains orientation, complication and resolution • detailed longer text may resolve one complication and lead into a new complication or layer a new complication onto an existing one rather than conclude
4- coherent, controlled and complete narrative, employing effective plot devices in an appropriate structure, and including an effective ending. sophisticated structures or plot devices include: – foreshadowing/flashback – red herring/cliffhanger – coda/twist – evaluation/reflection – circular/parallel plots

Criteria 3. Ideas: The creation, selection and crafting of ideas for a narrative.
Score Range: 0-5 
Scoring guide: 
0 - • no evidence or insufficient evidence • symbols or drawings • title only
1 - • one idea OR • ideas are very few and very simple OR • ideas appear unrelated to each other OR • ideas appear unrelated to prompt
2 - • one idea with simple elaboration OR • ideas are few and related but not elaborated OR • many simple ideas are related but not elaborated
3 - • ideas show some development or elaboration • all ideas relate coherently
4 - • ideas are substantial and elaborated AND contribute effectively to a central storyline • the story contains a suggestion of an underlying theme
5 - ideas are generated, selected and crafted to explore a recognisable theme • ideas are skilfully used in the service of the storyline • ideas may include: – psychological subjects – unexpected topics – mature viewpoints – elements of popular culture – satirical perspectives – extended metaphor – traditional sub-genre subjects: heroic quest / whodunnit / good vs evil / overcoming the odds

Criteria 4 .Character and setting: Character: the portrayal and development of character. 
Setting: the development of a sense of place, time and atmosphere.
Score Range: 0-4 
Scoring guide: 

0 - no evidence or insufficient evidence, symbols or drawings, writes in wrong genre, title only

1 - only names characters or gives their roles (e.g. father, the teacher, my friend, dinosaur, we, Jim) AND/OR only names the setting (e.g.school, the place we were at) setting is vague or confused	
2 - suggestion of characterisation through brief descriptions or speech or feelings, but lacks substance or continuity 
AND/OR
suggestion of setting through very brief and superficial descriptions of place and/or time	
basic dialogue or a few adjectives to describe a character or a place

3 - characterisation emerges through descriptions, actions, speech or the attribution of thoughts and feelings to a character
AND/OR
setting emerges through description of place, time and atmosphere	

4 - effective characterisation: details are selected to create distinct characters
AND/OR
Maintains a sense of setting throughout. Details are selected to create a sense of place and atmosphere. convincing dialogue, introspection and reactions to other characters

Criteria 5. Vocabulary: The range and precision of language choices. 
Score Range: 0-5 
Scoring guide: 
0 - symbols or drawings
1 - very short script
2 - mostly simple verbs, adverbs, adjectives or nouns may include two or three precise words or word groups
3 - four or more precise words or word groups (may be verbs, adverbs, adjectives or nouns)
4 - sustained and consistent use of precise words and word groups that enhance the meaning or mood
may be occasional inappropriate or inaccurate word choice
5 - a range of precise and effective words and word groups used in a natural and articulate manner, language choice is well matched to genre

Criteria 6. Cohesion: The control of multiple threads and relationships over the whole text, achieved through the use of referring words, substitutions, word associations and text connectives. 
Score Range: 0-4 
Scoring guide: 
0 - symbols or drawings
1 - links are missing or incorrect short script often confusing for the reader
2 - some correct links between sentences (do not penalise for poor punctuation), 
most referring words are accurate. reader may occasionally need to re-read and provide their own links to clarify meaning
3 - cohesive devices are used correctly to support reader understanding, accurate use of referring words,
meaning is clear and text flows well in a sustained piece of writing
4 - a range of cohesive devices is used correctly and deliberately to enhance reading, an extended, highly cohesive piece of writing showing continuity of ideas and tightly linked sections of text

Criteria 7. Paragraphs: The segmenting of text into paragraphs that assists the reader to negotiate the narrative. 
Score Range: 0-2 
Scoring guide: 
0 - no use of paragraphing
1 - writing is organised into paragraphs that are mainly focused on a single idea or set of like ideas that assist the reader to digest chunks of text
contains at least one correct paragraph break
2 - all paragraphs are focused on one idea or set of like ideas and enhance the narrative

Criteria 8. Sentence structure: The production of grammatically correct, structurally sound and meaningful sentences. 
Score Range: 0-6 
Scoring guide: 
0 -	no evidence of sentences
1 - some correct formation of sentences, some meaning can be construed
2 - correct sentences are mainly simple and/or compound sentences, meaning is predominantly clear
3 - most (approx. 80%) simple and compound sentences correct
AND some complex sentences are correct meaning is predominantly clear
4 - most (approx. 80%) simple, compound and complex sentences are correct
OR all simple, compound and complex sentences are correct but do not demonstrate variety meaning is clear
5 - sentences are correct (allow for occasional error in more sophisticated structures)
demonstrates variety meaning is clear and sentences enhance meaning
6 - all sentences are correct (allow for occasional slip, e.g. a missing word) writing contains controlled and well developed sentences that express precise meaning and are
consistently effective

Criteria 9. Punctuation: The use of correct and appropriate punctuation to aid reading of the text. 
Score Range: 0-5 
Criteria 10. Spelling: The accuracy of spelling and the difficulty of the words used. 
Score Range: 0-6

        Task Title: {task_title}

        Task Description: {task_desc}

        story: {essay}

        Grade the story based on the provided guidelines keeping in mind the test taker is grade 7 student and score them according to the level expected from a student in that grade while providing a 2-3 lines justifying the specic score you provided for each of the criteria. Dont provide a range, use a specific number for the score. Mention your score and what it is out of for each criteria. 
        For each criteria mention what is good in the input essay, and how if it all it cab be improved to meet the guidelines better. 
        Finally sum up the individual scores to provide an overall score out of 47. Always first check if the story is in line with the Task Title and Task Description, if it is not then mention so and penalize while grading. If the story was empty or not in line with the Task Title and Description mention that and do not provide a grade.
        """,
    )

    naplan_narrative_grade9_prompt = PromptTemplate(
        input_variables=["essay", "task_title", "task_desc"],
        template="""You are an story grader for Naplan narrative writing task for grade 9 students. You will follow the below criteria and grade the students on all the provided criteria using the provided guidelines.

        Grade 9 Narrative Writing Grading Criteria

Criteria 1. Audience: The writer’s capacity to orient, engage and affect the reader. 
Score Range: 0-6
Scoring guide: 
0 - symbols or drawings which have the intention of conveying meaning.
1- response to audience needs is limited • contains simple written content. may be a title only OR • meaning is difficult to access OR • copied stimulus material, including prompt topic
2 - shows basic awareness of audience expectations through attempting to orient the reader • provides some information to support reader understanding. may include simple narrative markers, e.g. – simple title – formulaic story opening: Long, long ago …; Once a boy was walking when … • description of people or places • reader may need to fill gaps in information • text may be short but is easily read.
3- • orients the reader –an internally consistent story that attempts to support the reader by developing a shared understanding of context • contains sufficient information for the reader to follow the story fairly easily
4- • supports reader understanding AND • begins to engage the reader
5- supports and engages the reader through deliberate choice of language and use of narrative devices.
6- caters to the anticipated values and expectations of the reader • influences or affects the reader through precise and sustained choice of language and use of narrative devices

Criteria 2. Text structure: The organisation of narrative features including orientation, complication and resolution into an appropriate and effective text structure. 
Score Range: 0-4 
Scoring guide: 
0- no evidence of any structural components of a times equenced text • symbols or drawings • inappropriate genre, e.g. a recipe, argument • title only
1- minimal evidence of narrative structure, e.g. a story beginning only or a ‘middle’ with no orientation • a recount of events with no complication • note that not all recounts are factual • may be description
2- contains a beginning and a complication • where a resolution is present it is weak, contrived or ‘tacked on’ (e.g. I woke up, I died, They lived happily ever after) • a complication presents a problem to be solved, introduces tension, and requires a response. It drives the story forward and leads to a series of events or responses • complications should always be read in context • may also be a complete story where all parts of the story are weak or minimal (the story has a problem to be solved but it does not add to the tension or excitement).
3 - contains orientation, complication and resolution • detailed longer text may resolve one complication and lead into a new complication or layer a new complication onto an existing one rather than conclude
4- coherent, controlled and complete narrative, employing effective plot devices in an appropriate structure, and including an effective ending. sophisticated structures or plot devices include: – foreshadowing/flashback – red herring/cliffhanger – coda/twist – evaluation/reflection – circular/parallel plots

Criteria 3. Ideas: The creation, selection and crafting of ideas for a narrative.
Score Range: 0-5 
Scoring guide: 
0 - • no evidence or insufficient evidence • symbols or drawings • title only
1 - • one idea OR • ideas are very few and very simple OR • ideas appear unrelated to each other OR • ideas appear unrelated to prompt
2 - • one idea with simple elaboration OR • ideas are few and related but not elaborated OR • many simple ideas are related but not elaborated
3 - • ideas show some development or elaboration • all ideas relate coherently
4 - • ideas are substantial and elaborated AND contribute effectively to a central storyline • the story contains a suggestion of an underlying theme
5 - ideas are generated, selected and crafted to explore a recognisable theme • ideas are skilfully used in the service of the storyline • ideas may include: – psychological subjects – unexpected topics – mature viewpoints – elements of popular culture – satirical perspectives – extended metaphor – traditional sub-genre subjects: heroic quest / whodunnit / good vs evil / overcoming the odds

Criteria 4 .Character and setting: Character: the portrayal and development of character. 
Setting: the development of a sense of place, time and atmosphere.
Score Range: 0-4 
Scoring guide: 

0 - no evidence or insufficient evidence, symbols or drawings, writes in wrong genre, title only

1 - only names characters or gives their roles (e.g. father, the teacher, my friend, dinosaur, we, Jim) AND/OR only names the setting (e.g.school, the place we were at) setting is vague or confused	
2 - suggestion of characterisation through brief descriptions or speech or feelings, but lacks substance or continuity 
AND/OR
suggestion of setting through very brief and superficial descriptions of place and/or time	
basic dialogue or a few adjectives to describe a character or a place

3 - characterisation emerges through descriptions, actions, speech or the attribution of thoughts and feelings to a character
AND/OR
setting emerges through description of place, time and atmosphere	

4 - effective characterisation: details are selected to create distinct characters
AND/OR
Maintains a sense of setting throughout. Details are selected to create a sense of place and atmosphere. convincing dialogue, introspection and reactions to other characters

Criteria 5. Vocabulary: The range and precision of language choices. 
Score Range: 0-5 
Scoring guide: 
0 - symbols or drawings
1 - very short script
2 - mostly simple verbs, adverbs, adjectives or nouns may include two or three precise words or word groups
3 - four or more precise words or word groups (may be verbs, adverbs, adjectives or nouns)
4 - sustained and consistent use of precise words and word groups that enhance the meaning or mood
may be occasional inappropriate or inaccurate word choice
5 - a range of precise and effective words and word groups used in a natural and articulate manner, language choice is well matched to genre

Criteria 6. Cohesion: The control of multiple threads and relationships over the whole text, achieved through the use of referring words, substitutions, word associations and text connectives. 
Score Range: 0-4 
Scoring guide: 
0 - symbols or drawings
1 - links are missing or incorrect short script often confusing for the reader
2 - some correct links between sentences (do not penalise for poor punctuation), 
most referring words are accurate. reader may occasionally need to re-read and provide their own links to clarify meaning
3 - cohesive devices are used correctly to support reader understanding, accurate use of referring words,
meaning is clear and text flows well in a sustained piece of writing
4 - a range of cohesive devices is used correctly and deliberately to enhance reading, an extended, highly cohesive piece of writing showing continuity of ideas and tightly linked sections of text

Criteria 7. Paragraphs: The segmenting of text into paragraphs that assists the reader to negotiate the narrative. 
Score Range: 0-2 
Scoring guide: 
0 - no use of paragraphing
1 - writing is organised into paragraphs that are mainly focused on a single idea or set of like ideas that assist the reader to digest chunks of text
contains at least one correct paragraph break
2 - all paragraphs are focused on one idea or set of like ideas and enhance the narrative

Criteria 8. Sentence structure: The production of grammatically correct, structurally sound and meaningful sentences. 
Score Range: 0-6 
Scoring guide: 
0 -	no evidence of sentences
1 - some correct formation of sentences, some meaning can be construed
2 - correct sentences are mainly simple and/or compound sentences, meaning is predominantly clear
3 - most (approx. 80%) simple and compound sentences correct
AND some complex sentences are correct meaning is predominantly clear
4 - most (approx. 80%) simple, compound and complex sentences are correct
OR all simple, compound and complex sentences are correct but do not demonstrate variety meaning is clear
5 - sentences are correct (allow for occasional error in more sophisticated structures)
demonstrates variety meaning is clear and sentences enhance meaning
6 - all sentences are correct (allow for occasional slip, e.g. a missing word) writing contains controlled and well developed sentences that express precise meaning and are
consistently effective

Criteria 9. Punctuation: The use of correct and appropriate punctuation to aid reading of the text. 
Score Range: 0-5 
Criteria 10. Spelling: The accuracy of spelling and the difficulty of the words used. 
Score Range: 0-6

        Task Title: {task_title}

        Task Description: {task_desc}

        story: {essay}

        Grade the story based on the provided guidelines keeping in mind the test taker is grade 9 student and score them according to the level expected from a student in that grade while providing a 2-3 lines justifying the specic score you provided for each of the criteria. Dont provide a range, use a specific number for the score. Mention your score and what it is out of for each criteria. 
        For each criteria mention what is good in the input essay, and how if it all it cab be improved to meet the guidelines better. 
        Finally sum up the individual scores to provide an overall score out of 47. Always first check if the story is in line with the Task Title and Task Description, if it is not then mention so and penalize while grading. If the story was empty or not in line with the Task Title and Description mention that and do not provide a grade.
        """,
    )

 # Determine which prompt to use based on the parameters
    if exam_type.strip().lower() == "ielts":
        prompt = ielts_prompt
    elif exam_type.strip().lower() == "naplan":
        if essay_type.strip().lower() == "persuasive":
            if grade == "Grade 3":
                prompt = naplan_persuasive_grade3_prompt
            elif grade == "Grade 5":
                prompt = naplan_persuasive_grade5_prompt  # Placeholder, you'll need to define this
            elif grade == "Grade 7":
                prompt = naplan_persuasive_grade7_prompt  # Placeholder, you'll need to define this
            elif grade == "Grade 9":
                prompt = naplan_persuasive_grade9_prompt  # Placeholder, you'll need to define this
        elif essay_type.strip().lower() == "narrative":
            if grade == "Grade 3":
                prompt = naplan_narrative_grade3_prompt  # Placeholder, you'll need to define this
            elif grade == "Grade 5":
                prompt = naplan_narrative_grade5_prompt  # Placeholder, you'll need to define this
            elif grade == "Grade 7":
                prompt = naplan_narrative_grade7_prompt  # Placeholder, you'll need to define this
            elif grade == "Grade 9":
                prompt = naplan_narrative_grade9_prompt  # Placeholder, you'll need to define this
        #     # ... [Continue for other grades and essay types] ...

    chain = LLMChain(llm=llm, prompt=prompt)

    inputs = {
        "essay": user_response,
        "task_title": title,
        "task_desc": description
    }

    print(essay_type, title, description, prompt)
    feedback_from_api = chain.run(inputs)
    return feedback_from_api

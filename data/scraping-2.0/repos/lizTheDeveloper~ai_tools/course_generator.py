import os
import openai
from prompt import natural_language_prompt, discerning_natural_language_prompt, code_example_prompt
from file_handling import save_to_file
from parsers import parse_table_text, parse_data_to_table

def generate_course(topic,number_of_weeks,hours_per_week,target_job):
    ## create a new directory for the course
    course_name = topic.replace(" ","_")
    ## check to see if it already exists
    course_directory = "./courses/" + course_name.replace(" ", "_") + "_for_" + target_job.replace(" ","_")
    if os.path.exists(course_directory):
        print("course already exists")
        ## don't try to create it (it already exists)
    else:
        os.mkdir("./courses/" + course_name)
    ## check to see if the course has a objectives.md file
    if os.path.exists(course_directory + "/objectives.md"):
        print("objectives.md already exists")
        ## load the objectives.md file
        objectives_file = open(course_directory + "/objectives.md", "r")
        objectives = objectives_file.read()
        print("Found objectives.md")
        print(objectives)
        accepted_objectives = "y"
    else:
        print("No objectives found. Creating objectives.md and generating probable objectives")
        accepted_objectives = "n"

    while accepted_objectives == "n":
        objectives = objectives_from_course_generator(topic,target_job,number_of_weeks,hours_per_week)
        print("Generated objectives...")
        print(objectives)
        accepted_objectives = input("Accept these objectives? (y/n)")
        if accepted_objectives == "y":
            save_to_file(course_directory + "/objectives.md",objectives)
            break
    accepted_course_outline = "n"
    ## check to see if there is a course_outline.md file
    if os.path.exists(course_directory + "/course_outline.md"):
        print("course_outline.md already exists")
        ## load the course_outline.md file
        course_outline_file = open(course_directory + "/course_outline.md", "r")
        course_outline = course_outline_file.read()
        print(course_outline)
        accepted_course_outline = "y"
    else:
        accepted_course_outline = "n"
        while accepted_course_outline == "n":
            course_outline = course_from_topic_generator(topic,objectives,number_of_weeks,hours_per_week)
            print("Generated Course Outline...")
            print(course_outline)
            accepted_course_outline = input("Accept this course outline? (y/n)")
            if accepted_course_outline == "y":
                save_to_file(course_directory + "/course_outline.md",course_outline)
                break
    course_materials_outline = weekly_outline_to_table_parser(course_outline)

    print(course_materials_outline)
    ## generate a module overview for each week by grouping objectives by week and creating a module overview for each week, and by grouping the course materials outline by week
    for module in course_materials_outline:
        ## check to see that a module does not already exist
        if module.get("topic") is not None:
            module_directory = course_directory + "/" + module.get("topic")
            subtopic = module.get("subtopic","overview")
            topic = module.get("topic")
            module_overview_path = module_directory + "/" + module.get("subtopic") + ".md"
            if os.path.exists(module_directory):
                print("module already exists")
                ## don't try to create it (it already exists)
                continue_creating_module = input(f"Module already exists. Create module overview for { topic } { subtopic }? (y/n)")
            else:
                os.mkdir(module_directory)
                continue_creating_module = input(f'Create module overview for { topic } { subtopic }? (y/n)')
            if continue_creating_module == "y":
                module_overview = module_overview_from_topic_and_subtopic(course_name,module.get("topic"),module.get("subtopic"))
                accepted_response = input(f"Accept this module overview? (y/n)")
                if accepted_response == "y":
                    print("Saving module overview to file", module_overview_path)
                    save_to_file(module_overview_path,module_overview)
                while accepted_response == "n":
                    module_overview = module_overview_from_topic_and_subtopic(course_name, module.get("subtopic"), module.get("subtopic"))
                    accepted_response = input(f"Accept this module overview? (y/n)")
                    if accepted_response == "y":
                        print("saving to file", module_overview_path)
                        save_to_file(module_overview_path,module_overview)
                        break
            else:
                ## read the module_overview from the file
                module_overview_file = open(module_overview_path, "r")
                module_overview = module_overview_file.read()
            learning_objectives = extract_learning_objectives(module_overview)
            print(learning_objectives)
            for objective in learning_objectives:
                print(objective)

                continue_generating_artifacts = input("Continue generating artifacts for this objective? (y/n)")
                if continue_generating_artifacts == "y":
                    artifact = generate_artifacts_for_objective(objective, course_name, target_job, module_directory)
                    print(artifact)
                    accepted_response = input("Accept this artifact? (y/n)")
                    if accepted_response == "y":
                        save_to_file(module_directory + "/" + objective + ".md",artifact)
                    
                




## course definitions
## generates a backwards-designed list of objectives based on a topic
def course_from_topic_generator(topic,objectives,number_of_weeks,hours_per_week):
    total_hours = hours_per_week * number_of_weeks
    prompt = f"Create the outline of a {number_of_weeks}-week, {total_hours}-hour course on {topic} that covers the following objectives: \n\n {objectives} \n\n Week 1: \n\n"

    return discerning_natural_language_prompt(prompt)

## generates a course outline that is backwards-designed against objectives
def objectives_from_course_generator(course_name,target_job,number_of_weeks,hours_per_week):
    total_hours = hours_per_week * number_of_weeks
    objective_low_range = number_of_weeks
    objective_high_range = total_hours * 2
    prompt = f"After a {total_hours}-hour course in {course_name}, what {objective_low_range}-{objective_high_range} objectives should a learner who wants to become a {target_job} have learned? Do not use the words Understand, Understanding, Know, Fluency, or Using, instead use a Bloom's Taxonomy Verb and then describe the tasks you will learn to accomplish by the end of the course on {course_name} \n\n-"
    return natural_language_prompt(prompt)

## generate a module overview from the topic
def module_overview_from_topic_and_subtopic(course_name,topic,subtopic):
    prompt = f"#{course_name}: {topic}\n#{subtopic}\n\n#### Module Overview \n In this module on {topic}, we'll discuss {subtopic}. \n\n Learning Objectives: \n\n- "
    module_overview = natural_language_prompt(prompt)
    return prompt + module_overview

def extract_learning_objectives(module_overview):
    learning_objectives = []
    for line in module_overview.split("\n"):
        if line.startswith("- "):
            learning_objectives.append(line[2:])
    return learning_objectives

def module_from_objectives_in_week(objectives):
    prompt = f"Create a module overview for the week, where we'll cover {objectives} \n\n"
    module_overview = natural_language_prompt(prompt)
    

    return """
    # Module Overview 
    This week we'll be covering the following learning objectives:
    {objectives}

    {module_overview}

    """

def generate_artifacts_for_objective(objective, course_name, target_job, module_directory):
    prompt = f"# {course_name} for {target_job}\n# {objective} \n\nIn this article we'll cover {objective}: \n\n"
    article = natural_language_prompt(prompt)
    print(article)
    accept_article = input("Accept this article? (y/n)")
    while accept_article != "y":
        article = natural_language_prompt(prompt)
        accept_article = input("Accept this article? (y/n)")

    scenarios = generate_scenarios_for_objective(objective, course_name, target_job)
    print(scenarios)
    accept_scenarios = input("Accept these scenarios? (y/n)")
    while accept_scenarios != "y":
        scenarios = generate_scenarios_for_objective(objective, course_name, target_job)
        accept_scenarios = input("Accept these scenarios? (y/n)")
    

    concrete_examples = generate_concrete_examples_from_scenarios(scenarios, course_name, target_job)
    print(concrete_examples)
    accept_concrete_examples = input("Accept these concrete examples? (y/n)")
    while accept_concrete_examples != "y":
        concrete_examples = generate_concrete_examples_from_scenarios(scenarios, course_name, target_job)
        accept_concrete_examples = input("Accept these concrete examples? (y/n)")
    
    practice_exercise = generate_practice_exercise(objective, course_name, target_job)
    print(practice_exercise)
    accept_practice_exercise = input("Accept this practice exercise? (y/n)")
    while accept_practice_exercise != "y":
        practice_exercise = generate_practice_exercise(objective, course_name, target_job)
        accept_practice_exercise = input("Accept this practice exercise? (y/n)")

    python_sample = generate_code_sample_of_objective("Python", objective, course_name, target_job)
    print(python_sample)
    accept_python_sample = input("Accept this python sample? (y/n)")
    while accept_python_sample != "y":
        python_sample = generate_code_sample_of_objective("Python", objective, course_name, target_job)
        accept_python_sample = input("Accept this python sample? (y/n)")
    
        
        
        
        
    return f"""# {objective}
        {article}
        ### When might you need to {objective}?
        {scenarios}

        This might look like:

        {concrete_examples}

        ### How do you practice {objective}?
        {practice_exercise}

        ### How do you practice {objective} in Python?
        {python_sample}
        """


def generate_scenarios_for_objective(objective, course_name, target_job):
    prompt = f"For a course on {course_name}, describe 3 simple scenarios during your job as a {target_job} in which you might need to {objective} \n\n"
    return discerning_natural_language_prompt(prompt)

def generate_concrete_examples_from_scenarios(scenarios, course_name, target_job):
    prompt = f"{scenarios} \n\n Name 3 example tasks in a real {target_job} where you might do the above scenarios?\n\n"
    return discerning_natural_language_prompt(prompt)

def generate_practice_exercise(objective, course_name, target_job):
    prompt = f"Create a short exercise to practice {objective} for a course on {course_name} for {target_job} \n\n"
    return natural_language_prompt(prompt)

def objectives_from_learning_content(learning_content):
    prompt = f"What learning objectives does the above learning content cover? Do not use the words Understand, Know, Fluency, or Using to describe the learning objectives, instead use a Bloom's Taxonomy Verb along with a description the task\n\n-"
    return natural_language_prompt(prompt)

def generate_code_sample_of_objective(language,objective, course_name, target_job):
    natural_language_objective = f"Create a short example of code using a realistic example of real-world data that a {target_job} would use and meaningful variable names to practice {objective} in a course on {course_name} \n\n"
    #comment_prompt = f"Write the following in a {language} comment: {natural_language_objective}"
    if language == "Python":
        prompt = f"# {natural_language_objective} \n\n"
        return code_example_prompt(prompt)
    if language == "Java":
        prompt = f"// {natural_language_objective} \n\n"
        return code_example_prompt(prompt)
    if language == "C++":
        prompt = f"// {natural_language_objective} \n\n"
        return code_example_prompt(prompt)
    if language == "C#":
        prompt = f"// {natural_language_objective} \n\n"
        return code_example_prompt(prompt)
    if language == "Ruby":
        prompt = f"# {natural_language_objective} \n\n"
        return code_example_prompt(prompt)
    if language == "Swift":
        prompt = f"// {natural_language_objective} \n\n"
        return code_example_prompt(prompt)
    if language == "JavaScript":
        prompt = f"// {natural_language_objective} \n\n"
        return code_example_prompt(prompt)
    return code_example_prompt(f"# {natural_language_objective} \n\n")

#### subtopic parser, gets a table of subtopics to generate module overviews with
def course_topic_outline_parser(response_text):
    return parse_data_to_table(response_text,"topic|subtopic","every subtopic associated with its topic")

## generate a table of objectives from generated objectives
def objectives_to_table_parser(response_text):
    columns = "objective|level"
    table_text = parse_data_to_table(response_text,columns,"each objective above and its level according to Bloom's Taxonomy")
    return parse_table_text(table_text,columns)

def weekly_outline_to_table_parser(response_text):
    columns = "topic|subtopic|day_number|week_number"
    table_text = parse_data_to_table(response_text,columns,"each item in the outline above, its topic and its subtopic and its day number and its week number")
    return parse_table_text(table_text,columns)


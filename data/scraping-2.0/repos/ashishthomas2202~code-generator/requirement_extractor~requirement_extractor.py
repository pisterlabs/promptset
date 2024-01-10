import os
import openai


def initiate():
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Initialize an empty dictionary to store the requirements
    project = {
        "name": "",
        "description": "",
        "requirements": {
            "target_platforms": "",
            "frontend_required": "",
            "backend_required": "",
            "description": "",
        },
        "additional_notes": ""
    }

    done = False
    while not done:
        project["name"] = input("Enter project name: ")
        if len(project["name"]) > 2:
            done = True

    done = False
    while not done:
        project["description"] = input("Enter project description: ")
        if len(project["description"]) > 5:
            done = True

    done = False
    while not done:
        target_platforms = input(
            "Enter targeted platform(s) (comma-separated, e.g., Web, Mobile, Desktop): ")

        platforms = [platform.strip().capitalize()
                     for platform in target_platforms.split(",")]

        valid_platforms = ["Web", "Mobile", "Desktop"]

        if all(platform in valid_platforms for platform in platforms):
            project["requirements"]["target_platforms"] = platforms
            done = True
        else:
            print(
                "Invalid platform(s). Please use 'Web', 'Mobile', 'Desktop' or combinations.")

    done = False
    while not done:
        user_input = input("Is a Frontend required? (yes/no): ").lower()
        if user_input in ["yes", "no"]:
            project["requirements"]["frontend_required"] = (
                user_input == 'yes')
            done = True
        else:
            print("Please enter 'yes' or 'no'.")

    done = False
    while not done:
        user_input = input("Is a backend required? (yes/no): ").lower()
        if user_input in ["yes", "no"]:
            project["requirements"]["backend_required"] = (
                user_input == 'yes')
            done = True
        else:
            print("Please enter 'yes' or 'no'.")

    if project["requirements"]["frontend_required"] == False and project["requirements"]["backend_required"] == False:
        print("Invalid Input: Atleast one of the frontend or backend is required")
        done = False
        while not done:
            user_input = input("Is a Frontend required? (yes/no): ").lower()
            if user_input in ["yes", "no"]:
                project["requirements"]["frontend_required"] = (
                    user_input == 'yes')
                done = True
            else:
                print("Please enter 'yes' or 'no'.")

        done = False
        while not done:
            user_input = input("Is a backend required? (yes/no): ").lower()
            if user_input in ["yes", "no"]:
                project["requirements"]["backend_required"] = (
                    user_input == 'yes')
                done = True
            else:
                print("Please enter 'yes' or 'no'.")

    done = False
    while not done:
        project["requirements"]["description"] = input(
            "Enter project requirements: ")
        if len(project["requirements"]["description"]) > 5:
            done = True

    done = False
    while not done:
        project["additional_notes"] = input("Enter additional notes: ")
        if len(project["additional_notes"]) > 1:
            done = True

    return project

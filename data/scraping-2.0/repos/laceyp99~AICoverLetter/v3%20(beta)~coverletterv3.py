import os, json, time, re
from docx import Document
from openai import OpenAI

# Set up OpenAI API key - Save your API key as a environment variable named OPENAI_API_KEY
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Gets rid of any citations that the model adds in to show the retrieval. In the interest of looking professional, the citations are not needed for a cover letter.
def sanitize_text(text):
    # Define a regular expression pattern to match content between \u3010 and \u3011
    pattern = r'\u3010.*?\u3011'
    
    # Remove content between \u3010 and \u3011 along with the brackets
    sanitized_text = re.sub(pattern, '', text)
    
    return sanitized_text

# Exports cover letter as DOCX or TXT (PDF not yet supported)
def export_cover_letter(cover_letter):
    export = input("Which format would you like to export your cover letter in (DOCX or TXT)?: ")
    if (export.lower() == "docx"):
        # Export cover letter as DOCX
        doc = Document()
        doc.add_paragraph(cover_letter)
        doc.save("cover_letter.docx")
        print("Cover letter exported as DOCX.")
    elif (export.lower() == "txt"):
        # Export cover letter as TXT
        with open("cover_letter.txt", "w") as file:
            file.write(cover_letter)
        print("Cover letter exported as TXT.")
    else:
        print("Invalid input. Please enter DOCX, or TXT.")
        export_cover_letter(cover_letter)

# Create files to be referenced in thread message using the file.id
def create_file(file_name):
    file = client.files.create(
        file=open(file_name, "rb"),
        purpose="assistants",
    )     
    file_id = file.id 
    print(f"File {file_id} created")
    return file_id

# Delete files so we don't clutter our OpenAI file storage
def delete_file(file_id):
    client.files.delete(
        file_id = file_id
    )
    print(f"File {file_id} deleted")

# Create assistant with task specific instructions and tools
def create_assistant():
    assistant = client.beta.assistants.create(
        name = "Recruiter",
        instructions = "You are a helpful recruiter assistant. You are to write cover letters using the supplied resume and job description. Don't lie about the users qualifications, but make sure to sell their skills and character as much as you can. Please only respond with the cover letter and no other explanation or comments.",
        model = "gpt-4-1106-preview",
        tools = [{"type": "retrieval"}]
    )
    print(f"Assistant {assistant.id} created")
    return assistant.id

# Delete any assistant using their id to keep OpenAI's assistant storage clean
def delete_assistant(assistant_id):
    client.beta.assistants.delete(
        assistant_id = assistant_id
    )
    print(f"Assistant {assistant_id} deleted")

# Creates the thread with the initial prompt and files
def create_thread(resume_id, job_description_id):

    thread = client.beta.threads.create()
    print(f"Thread {thread.id} created")

    user_message = "Write a cover letter using my resume and the job description."
    client.beta.threads.messages.create(
        thread_id = thread.id,
        role = "user",
        content = user_message,
        file_ids = [resume_id, job_description_id]
    )
    print(f"Prompt with files added to thread.")
    return thread.id

# Runs the thread, tracks the status until completed (or failed), and then logs the message list and run steps for analyzing
def run_thread(assistant_id, thread_id):
    # Finally we run the thread and wait for completion
    run = client.beta.threads.runs.create(
        thread_id = thread_id,
        assistant_id = assistant_id,
        instructions = "Please help the user write a cover letter for a job application. Only respond with the cover letter and no other explanation or comments.",
    )   
    print(f"Run {run.id} created.\nRun status: {run.status}")
    # print("Waiting for run to complete...")
    while True:
        run = client.beta.threads.runs.retrieve(
            thread_id = thread_id,
            run_id = run.id
        )
        if run.status == "failed":
            print("Run failed")
            run_success = False
            break

        if run.status == "completed":
            print("Run completed")
            run_success = True
            break

        print("Waiting for run to complete...")
        time.sleep(5)

    if run_success:
        messages = client.beta.threads.messages.list(thread_id = thread_id)
        with open("messages.json", "w") as f:
            messages_json = messages.model_dump()
            json.dump(messages_json, f, indent=4)
        print("Messages logged in messages.json")

        text = messages.data[0].content[0].text.value
        cover_letter = sanitize_text(text)
        print(f"Assistant: \n\n{cover_letter}")
        
        run_steps = client.beta.threads.runs.steps.list(
            thread_id = thread_id, 
            run_id = run.id
        )

        with open("run_steps.json", "w") as f:
            run_steps_json = run_steps.model_dump()
            json.dump(run_steps_json, f, indent=4)
        print("Run steps logged in run_steps.json")

        return cover_letter
    
    else:
        print("Run failed. Please try again.")
        return None

# Creates a message in the thread with the revision
def create_message(message, thread_id):
    client.beta.threads.messages.create(
        thread_id = thread_id,
        role = "user",
        content = message
    )
    print("Prompt added to thread's message list.")

def main():
    # Create assistant
    assistant = create_assistant()

    # Create files
    # Add your own resume and job description files here
    # Just drag and drop the files into the same folder as this script and then add the file name to the create_file function
    resume = create_file("Resume.pdf")
    job_description = create_file("job_description.txt")

    # Create and run initial thread
    thread = create_thread(resume, job_description)
    cover_letter = run_thread(assistant, thread)

    # conversational loop building on thread with revisions
    while True:
        make_revision = input("Would you like to make any revisions (YES/NO)?: ")
        if (make_revision.lower() == "no"):
            export = input("Would you like to export your cover letter (YES/NO)?: ")
            if (export.lower() == "yes"):
                export_cover_letter(cover_letter)
                print("Thank you for using the Recruiter Assistant!\n")
            else:
                print("Thank you for using the Recruiter Assistant!\n")
            break
        elif (make_revision.lower() == "yes"):
            revision = input("Prompt the model with your revision(s): ")
            create_message(revision, thread)
            cover_letter = run_thread(assistant, thread)
        else:
            print("Invalid input. Please enter YES or NO.")
            continue
    
    # Delete files and assistant to keep OpenAI's storage clean
    delete_file(job_description)
    delete_file(resume)
    delete_assistant(assistant)


if __name__ == "__main__":
    main()

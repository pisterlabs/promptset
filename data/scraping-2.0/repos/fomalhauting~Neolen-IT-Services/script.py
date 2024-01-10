import os
from langchain.langchain import LangChain

# Create agent
agent = LangChain()

# Function to upload files
def upload_files():
    files = []
    while True:
        file_path = input("Enter the file path (or 'done' to finish): ")
        if file_path.lower() == 'done':
            break
        if os.path.exists(file_path):
            files.append(file_path)
        else:
            print("File not found. Please try again.")
    return files

# Function to ask questions
def ask_question():
    question = input("Enter your question: ")
    answer = agent.answer(question)
    if answer is None:
        print("Sorry, I couldn't find an answer in the uploaded files.")
        answer = agent.generate_answer(question)  # Generate default answer
    print("Agent's answer:", answer)

# Main loop
while True:
    print("\nOptions:")
    print("1. Create agent")
    print("2. Upload files")
    print("3. Ask question")
    print("4. Exit")
    choice = input("Enter your choice (1, 2, 3, or 4): ")

    if choice == '1':
        agent.create_agent()
        print("Agent created.")

    elif choice == '2':
        if agent.is_agent_created():
            files = upload_files()
            agent.train(files)  # Train the agent with the uploaded files
            print("Training complete.")
        else:
            print("Agent is not created. Please create agent first.")

    elif choice == '3':
        if agent.is_agent_created() and agent.is_trained():
            ask_question()
        elif not agent.is_agent_created():
            print("Agent is not created. Please create agent first.")
        else:
            print("Agent is not trained. Please upload files first.")

    elif choice == '4':
        break

    else:
        print("Invalid choice. Please try again.")

print("Exiting...")

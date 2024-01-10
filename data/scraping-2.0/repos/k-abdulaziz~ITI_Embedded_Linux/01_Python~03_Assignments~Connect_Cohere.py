import cohere
import keyboard

# Replace 'YOUR_API_KEY' with your actual API key
co = cohere.Client('YOUR_API_KEY')

def main():
    while True:
        # Get input from user
        prompt = input('Enter your prompt: ')

        # Generate response using Cohere API
        response = co.generate(
          model='command-xlarge-nightly',
          prompt=prompt,
          max_tokens=400,
          temperature=0.9,
          k=0,
          stop_sequences=[],
          return_likelihoods='NONE')

        # Print generated response
        print('Response: {}\n'.format(response.generations[0].text))

        # Prompt user to continue or exit
        print('Press the Escape key to exit, or any other key to continue.')
        if keyboard.read_event().name == 'esc':
            exit()

if __name__ == '__main__':
    main()

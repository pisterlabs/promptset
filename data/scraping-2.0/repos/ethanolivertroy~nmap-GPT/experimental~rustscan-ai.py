import argparse
import subprocess
import openai

# Set OpenAI API key
openai.api_key = "your_openai_api_key"

# Define the command line arguments
parser = argparse.ArgumentParser(description='RustScan Network Scanner with OpenAI Analysis')
parser.add_argument('host', type=str, help='Host or IP address to scan')
parser.add_argument('-p', '--port', type=str, help='Port or port range to scan')

# Parse the command line arguments
args = parser.parse_args()

# Execute RustScan and capture the output
command = f'rustscan {args.host} -p {args.port} --json --log-level 0'
output = subprocess.check_output(command, shell=True)

# Parse the RustScan output
result = json.loads(output.decode('utf-8'))

# Extract the results of the RustScan scan
for host in result:
    print('Host: {} ({})'.format(host['ip'], host['hostnames']))
    for service in host['services']:
        if service['state'] == 'open':
            print('port: {}\tstate: {}'.format(service['port'], service['state']))
            prompt = f"What security concerns should I be aware of given the {service['name']} service on port {service['port']} for host: {host['ip']}?"
            response = openai.Completion.create(
                engine="text-davinci-002",
                prompt=prompt,
                max_tokens=1024,
                n=1,
                stop=None,
                temperature=0.5,
            )

            # Extract the generated report from the OpenAI API response
            text = response["choices"][0]["text"]
            print("OpenAI Report:\n{}".format(text))

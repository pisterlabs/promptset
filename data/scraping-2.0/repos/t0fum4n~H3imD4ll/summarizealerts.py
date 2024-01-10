import json
from openai import OpenAI
import keys

# OpenAI and alerts file details
openai_client = OpenAI(api_key=keys.openai_api_key)
alerts_file = 'alerts.json'

def read_alerts_from_file(filename=alerts_file):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return []  # Return an empty list if the file doesn't exist

def write_alerts_to_file(alerts, filename=alerts_file):
    with open(filename, 'w') as file:
        json.dump(alerts, file, indent=4)

def summarize_alerts(alerts):
    # Combine the alerts into a single string
    alerts_text = " ".join([alert["content"] for alert in alerts])

    # Generate summary using OpenAI
    prompt = f"Please summarize the following alerts into key points. Be sure to be as detailed as possible but use as few words as possible. Make sure you retain the important parts of the alerts: {alerts_text}"
    response = openai_client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "system", "content": prompt}]
    )

    return response.choices[0].message.content

def main():
    alerts = read_alerts_from_file()
    if alerts:
        summary = summarize_alerts(alerts)
        # Overwrite the alerts with the summary
        summarized_alerts = [{"role": "system", "content": summary}]
        write_alerts_to_file(summarized_alerts)
        print("Alerts summarized and saved.")
    else:
        print("No alerts to summarize.")

if __name__ == "__main__":
    main()

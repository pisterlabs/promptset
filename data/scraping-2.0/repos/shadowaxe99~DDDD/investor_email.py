import openai
import sys
import argparse
import logging
import datetime
from openai.error import RateLimitError

openai.api_key = 'YOUR_API_KEY'


def format_prompt(investor):
    return f"{investor['name']} is a {investor['type']} investor who has invested in {', '.join(investor['investments'])}."

def read_csv_to_json(file):
    # Implement this function to read the CSV file and convert it to JSON
    pass

def safeguard(result):
    # Implement this function to check the result
    pass

def send_email(to, cc, bcc, subject, body, date=None):
    # Implement this function to send the email
    pass

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--file', type=str, help='The CSV file containing the investors information')
    parser.add_argument('--temperature', type=float, help='The temperature for the GPT-3 model')
    parser.add_argument('--max_tokens', type=int, help='The maximum number of tokens for the GPT-3 model')
    parser.add_argument('--mode', type=str, help='The mode for the GPT-3 model')
    parser.add_argument('--reply_speed', type=str, help='The reply speed for the GPT-3 model')

    args = parser.parse_args()

    investors = read_csv_to_json(args.file)

    for investor in investors:
        print(f"Running - {investor['Name']}")
        prompt = format_prompt({
            "name": investor["Name"],
            "company_name": investor["Company Name"],
        })
        messages = [{"role": "system", "content": f"""{prompt}"""}]

        try:
            if args.reply_speed == "slow":
                result = openai.engine("whisper").generate(
                    prompt=messages,
                    engine="davinci-002",
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    use_whisper=True,
                    mode=args.mode,
                    wait_for_completion=True
                )
            elif args.reply_speed == "fast":
                result = openai.engine("whisper").generate(
                    prompt=messages,
                    engine="davinci-002",
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    use_whisper=True,
                    mode=args.mode,
                    wait_for_completion=False
                )
            else:
                print("Invalid reply speed specified.")
                exit()
        except RateLimitError as e:
            logging.error(f"WARNING: RateLimitError occurred: {str(e)}")
            continue
        except Exception as e:
            logging.error(f"WARNING: Unknown error occurred: {str(e)}")
            continue

        if not safeguard(result):
            logging.warning(f"WARNING: Not sending the email to - {investor['Name']}")
            failed.append({investor['Name']: result})
            continue

        result = result.replace("[NAME]", investor["Name"])
        result = result.replace("[COMPANY NAME]", investor["Company Name"])

        investor["Opens"] = 0
        investor["Clicks"] = 0

        send_email([investor['Email']], cc_list, bcc_list, email_subject, result)

        follow_up_subject = "Re: " + email_subject
        follow_up_body = "Hi [NAME],\n\nI hope this email finds you well.\n\nI'm just following up on the email I sent you earlier. I'm still very interested in your company and I'm wondering if you had a chance to review my proposal.\n\nIf you have any questions, please don't hesitate to ask.\n\nBest,\n[Your Name]"

        send_email([investor['Email']], cc_list, bcc_list, follow_up_subject, follow_up_body, date=datetime.datetime.now() + datetime.timedelta(days=3))

if __name__ == '__main__':
    main()
from openai import OpenAI
import os
import sys
import pandas as pd
import yaml
import threading

from src.helper.utils import load_data


def transform_string(s):
  return s[1:-1].replace("'", "").replace(", ", " ")


def api_call_thread(offer, result_container):
  # Load OpenAI API from your environment
  API_KEY = os.getenv("OPENAI_API_KEY")
  client = OpenAI(api_key=API_KEY)
  try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"You are a professional job recruiter. Your task is to categorize a job description with keywords into one and only one of the specified 20 categories: {industries}. You are not allowed to use any other categories."},
            {"role": "user", "content": "Classify into one of the given indsutries. Job description: '''would like part ryanair group amazing cabin crew family k crew customer oriented love delivering great service want fast track career opportunity would delighted hear experience required bag enthusiasm team spirit europe largest airline carrying k guest daily flight looking next generation cabin crew join u brand new copenhagen base flying board ryanair group aircraft amazing perk including discounted staff travel destination across ryanair network fixed roster pattern free training industry leading pay journey becoming qualified cabin crew member start week training course learn fundamental skill require part day day role delivering top class safety customer service experience guest course required study exam taking place regular interval training culminates supernumerary flight followed cabin crew wing member ryanair group cabin crew family immersed culture day one career opportunity endless including becoming number base supervisor european base manager regional manager aspire becoming director inflight life cabin crew fun rewarding however demanding position safety number priority required operate early late shift report duty early morning early roster return home midnight afternoon roster morning person think twice applying requirement bag enthusiasm customer serviceoriented background ie previous experience working bar restaurant shop etc applicant must demonstrate legal entitlement work unrestricted basis across euyou must cm cm height must able swim meter unaided help hardworking flexible outgoing friendly personality adaptable happy work shift roster enjoy dealing public ability provide excellent customer service attitude comfortable speaking writing english ease passion travelling meeting new people benefit free cabin crew training course adventure experience lifetime within cabin crew network explore new culture city colleague day flexible day day staff roster unlimited highly discounted staff travel rate sale bonus free uniform year security working financially stable airline daily per diem provided whilst training direct employment contract highly competitive salary package click link start new exciting career sky'''. Keywords: '''management,manufacturing, technology, information,internet'''"},
            {"role": "assistant", "content": "Hospitality & Tourism"},
            {"role": "user",
             "content": f"Classify into one of the given indsutries. Job description: '''{offer['description']}'''. Keywords: '''{offer['keywords']}'''"},
        ]
    )
    result_container["response"] = response
  except Exception as e:
    result_container["error"] = str(e)


def restart_script():
  print("Restarting script...")
  os.execv(sys.executable, ['python'] + sys.argv)


df = load_data(kind="processed")

df['description'] = df['description'].apply(transform_string)
df['keywords'] = df['function'] + ', ' + df['industries']
job_descriptions = df[['id', 'keywords', 'description']]

industries = "Software & IT, Healthcare & Medicine, Education & Training, Engineering & Manufacturing, Finance & Accounting, Sales & Marketing, Creative Arts & Design, Hospitality & Tourism, Construction & Real Estate, Legal & Compliance, Science & Research, Human Resources & Recruitment, Transportation & Logistics, Agriculture & Environmental, Retail & Consumer Goods, Media & Communications, Government & Public Sector, Non-Profit & Social Services, Energy & Utilities, Arts & Entertainment"

ground_truth = {}
yaml_file = 'ground_truth.yaml'
if os.path.exists(yaml_file):
  with open(yaml_file, 'r') as file:
    ground_truth = yaml.safe_load(file) or {}

for index, offer in job_descriptions.iterrows():
  if offer['id'] in ground_truth:
    continue

  result_container = {}
  thread = threading.Thread(target=api_call_thread,
                            args=(offer, result_container))
  thread.start()
  thread.join(timeout=10)

  if thread.is_alive() or "error" in result_container:
    restart_script()

  response = result_container.get("response")
  if response:
    skills = response.choices[0].message.content
    ground_truth[offer['id']] = skills
    with open(yaml_file, 'w') as file:
      yaml.dump(ground_truth, file, default_flow_style=False)
    print(f"Saved ground truth for offer ID: {offer['id']}")

ground_truth_df = pd.DataFrame.from_dict(
    yaml_file, orient='index', columns=['category'])
ground_truth_df.index.name = 'id'
ground_truth_df.reset_index(inplace=True)

mapping_rules = {
    'Software & IT': 'Software & IT',
    'Creative Arts & Design': 'Creative Arts & Design',
    'Engineering & Manufacturing': 'Engineering & Manufacturing',
    'Manufacturing': 'Engineering & Manufacturing',
    'Human Resources & Recruitment': 'Human Resources & Recruitment',
    'Energy & Utilities': 'Energy & Utilities',
    'Sales & Marketing': 'Sales & Marketing',
    'Consumer Goods': 'Retail & Consumer Goods',
    'Transportation & Logistics': 'Transportation & Logistics',
    'Finance & Accounting': 'Finance & Accounting',
    'Information Technology & Services': 'Software & IT',
    'IT & Software': 'Software & IT',
    'Non-Profit & Social Services': 'Non-Profit & Social Services',
    'Media & Communications': 'Media & Communications',
    'Technology': 'Software & IT',
    'Hospitality & Tourism': 'Hospitality & Tourism',
    'Retail & Consumer Goods': 'Retail & Consumer Goods',
    'Technology & Information': 'Software & IT',
    'Legal & Compliance': 'Legal & Compliance',
    'Healthcare & Medicine': 'Healthcare & Medicine',
    'Science & Research': 'Science & Research',
    'Information Technology': 'Software & IT',
    'Education & Training': 'Education & Training',
    'Business & Entrepreneurship': 'Finance & Accounting',
    'Logistics & Supply Chain': 'Transportation & Logistics',
    'Construction & Real Estate': 'Construction & Real Estate',
    'Arts & Entertainment': 'Arts & Entertainment',
    'Agriculture & Environmental': 'Agriculture & Environmental',
    'Staffing & Recruiting': 'Human Resources & Recruitment',
    'Maritime & Transportation': 'Transportation & Logistics',
    'Technology & IT': 'Software & IT',
    'Public Relations & Communications': 'Media & Communications',
    'Customer Service': 'Human Resources & Recruitment',
    'Information Technology (IT)': 'Software & IT',
    'Manufacturing & Engineering': 'Engineering & Manufacturing',
    'Renewable energy': 'Energy & Utilities',
    'Government & Public Sector': 'Government & Public Sector',
    'Customer Success': 'Sales & Marketing',
    'Insurance & Risk Management': 'Finance & Accounting',
    'Human Resources': 'Human Resources & Recruitment',
    'Marketing & Advertising': 'Sales & Marketing',
    'Pharmaceutical & Healthcare': 'Healthcare & Medicine',
    'Retail': 'Retail & Consumer Goods',
    'Environmental & Sustainability': 'Agriculture & Environmental',
    'Real Estate & Construction': 'Construction & Real Estate',
    'Aerospace & Defense': 'Engineering & Manufacturing',
    'Public Relations': 'Media & Communications',
    'Event Planning & Management': 'Hospitality & Tourism',
    'Sports & Recreation': 'Arts & Entertainment',
    'Medical equipment manufacturing': 'Healthcare & Medicine',
    'Renewable Energy': 'Energy & Utilities',
    'Technology & Internet': 'Software & IT',
    'Technology & Information Technology': 'Software & IT',
    'Administration & Office Support': 'Human Resources & Recruitment',
    'Information & Technology': 'Software & IT',
    'Administration': 'Human Resources & Recruitment',
    'Technology & Telecommunications': 'Software & IT',
    'Insurance': 'Finance & Accounting',
    'Insurance & Financial Services': 'Finance & Accounting',
    'Logistics & Supply Chain Management': 'Transportation & Logistics',
    'Market Research': 'Sales & Marketing'
}

ground_truth_df['category'] = ground_truth_df['category'].map(mapping_rules)

ground_truth_df['category'] = pd.Categorical(ground_truth_df['category'])
ground_truth_df['cluster'] = ground_truth_df['category'].cat.codes
df_id_and_cluster = ground_truth_df[["id", "category", "cluster"]].sort_values(
    by="cluster", ascending=True
)

df_id_and_cluster.to_csv("clusters/ground_truth_gpt.csv", index=False)

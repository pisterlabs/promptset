import os
import openai

from config import openai_api_key
openai.api_key = openai_api_key


#a = {'id': 'CVE-2023-2665', 'sourceIdentifier': 'cve@mitre.org', 'published': '2023-05-05T12:15:09.093', 'lastModified': '2023-05-11T15:10:27.660', 'vulnStatus': 'Analyzed', 'descriptions': [{'lang': 'en', 'value': 'NS-ASG v6.3 was discovered to contain a SQL injection vulnerability via the component /admin/add_ikev2.php.'}], 'metrics': {'cvssMetricV31': [{'source': 'nvd@nist.gov', 'type': 'Primary', 'cvssData': {'version': '3.1', 'vectorString': 'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H', 'attackVector': 'NETWORK', 'attackComplexity': 'LOW', 'privilegesRequired': 'NONE', 'userInteraction': 'NONE', 'scope': 'UNCHANGED', 'confidentialityImpact': 'HIGH', 'integrityImpact': 'HIGH', 'availabilityImpact': 'HIGH', 'baseScore': 9.8, 'baseSeverity': 'CRITICAL'}, 'exploitabilityScore': 3.9, 'impactScore': 5.9}]}, 'weaknesses': [{'source': 'nvd@nist.gov', 'type': 'Primary', 'description': [{'lang': 'en', 'value': 'CWE-89'}]}], 'configurations': [{'nodes': [{'operator': 'OR', 'negate': False, 'cpeMatch': [{'vulnerable': True, 'criteria': 'cpe:2.3:a:netentsec:application_security_gateway:6.3:*:*:*:*:*:*:*', 'matchCriteriaId': '7A708923-EF5F-4F53-86F3-DF0366E6FB58'}]}]}], 'references': [{'url': 'http://ns-asg.com', 'source': 'cve@mitre.org', 'tags': ['Broken Link']}, {'url': 'https://gist.github.com/nsvizp/ccbc51253a47d79d2bc1ea72e01bdaf6', 'source': 'cve@mitre.org', 'tags': ['Third Party Advisory']}, {'url': 'https://www.netentsec.com/', 'source': 'cve@mitre.org', 'tags': ['Product']}], 'cpe': [{'vulnerable': True, 'criteria': 'cpe:2.3:a:netentsec:application_security_gateway:6.3:*:*:*:*:*:*:*', 'matchCriteriaId': '7A708923-EF5F-4F53-86F3-DF0366E6FB58'}], 'cwe': [{'lang': 'en', 'value': 'CWE-89'}], 'url': 'https://nvd.nist.gov/vuln/detail/CVE-2023-30242', 'v31score': 9.8, 'v31vector': 'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H', 'v31severity': 'CRITICAL', 'v31exploitability': 3.9, 'v31impactScore': 5.9, 'score': ['V31', 9.8, 'CRITICAL']}
def chatgpt_req(name, desc):
	completion = openai.ChatCompletion.create(
	  model="gpt-3.5-turbo",
	  messages=[
	    {"role": "user", "content": f"Номер уязвимости - {name} Описание уязвимости {desc} - опиши ее в 2х предложениях простым языком"}
	  ]
	)
	return completion.choices[0].message.content
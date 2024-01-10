from langchain.prompts import PromptTemplate


campaign_temps = PromptTemplate(
    input_variables=['company_data', 'company_json'],
    template="You are an Organization Job and Company Campaign Analyst responsible for analyzing company data and identifying jobs based on company scraped data and LinkedIn information.\n"
    "Your task is to analyze the potential job advertisements and campaigns from the company's scraped data:\n\n"
    "Company Scraped Data:\n"
    "{company_data}\n\n"
    "LinkedIn Data:\n"
    "{company_json}\n\n"
    "Filter the data using company job_title, job_title_role, and experience, and provide the filtered data in the following format under the subject 'Filtered Data':\n\n"
    "Filtered Data:\n"
    "1. Job Title: [List of job titles]\n"
    "2. Job Title Role: [List of job title roles]\n"
    "3. Experience: [List of experiences]\n"
    "4. Reason: [in paragraph, why the chosen filters makes sense in a descriptive manner, don't use word Linkedin, tell why the data points from these filters may buy the company. Include company name and product if available.]\n"
)
analyst_templat = PromptTemplate(
    input_variables=['company_data', 'company_json'],
    template="You are an Organization System Analyst tasked with analyzing company data to provide a comprehensive analysis of the company's operations. Your analysis should be data-driven, and you should prioritize using factual information about the company.\n"
    "\n"
    "Analyze the company's record and consider various data filters, if available, to provide insights. Please strictly follow the following optional data filters and itemized output format:\n"
    "\n"
    "1. Problems Addressed (Optional):\n"
    "   - Identify and describe the main problems or challenges that the company faces.\n"
    "   - Propose potential solutions to address these problems.\n"
    "\n"
    "2. Buyer Persona (Optional):\n"
    "   - Describe the target audience or customer demographics.\n"
    "   - Analyze the behavioral traits of the target audience.\n"
    "   - Identify the motivations and goals of potential customers.\n"
    "\n"
    "3. Target Market (Optional):\n"
    "   - Discuss the segmentation of the market.\n"
    "   - Estimate the potential reach of the company's products or services.\n"
    "   - Analyze the market size and growth potential.\n"
    "\n"
    "4. Company Summary (Optional):\n"
    "   - Provide a brief summary of the company, its history, and its key products or services.\n"
    "\n"
    "In your analysis, you can use the provided company data, including the following:\n"
    "\n"
    "LinkedIn Data (JSON format):\n"
    "{company_json}\n"
    "\n"
    "Company Website Scraped Data:\n"
    "{company_data}\n"
)

campaign_template = PromptTemplate(
    input_variables=['company_data', 'company_json'],
    template= "You are an Organization Job and Company Campaign Analyst responsible for analyzing company data and identifying jobs based on a company scraped data {company_data} and LinkedIn information {company_json}.\n"
    "Your task is to analyze the potential job advertisements and campaigns from the company's scraped data. You have to find best prospects to reach to sell the company product. \n" 
    "Give linkedin filters for potentials clients of the company and the reason why those filters are the best. \n" 
    "Strictly in the below format and order:\n"
    "Filtered Data:\n"
    "1. Job Title [list of job title, not more than 3, comma separated]\n"
    "2. Countries [list of countries, not more than 2, comma separated]\n"
    "3. Keywords [not more than 4, comma separated]\n"
    "4. Reason: paragraph why the filters makes sense in a descriptive manner, don't use word Linkedin, tell why the data points from these filters may buy the company. Include company name and product if available.\n"
)
analyst_template = PromptTemplate(
    input_variables=['company_data', 'company_json'],
    template="You are an analyst, which analysis many aspects of company with company data is provided. Use facts about the company provided as much as possible. \n"
    "Output should always be in the following format:"
    "1. Problems Addressed (Optional):\n"
    "   a. Problems Identified:\n"
    "       - [List of identified problems]\n"
    "   b. Solutions Offered:\n"
    "       - [List of offered solutions]\n\n"
    "2. Buyer Persona (Optional):\n"
    "   a. Demographics:\n"
    "       - [Demographic details]\n"
    "   b. Behavioral Traits:\n"
    "       - [Behavioral traits]\n"
    "   c. Motivations & Goals:\n"
    "       - [Motivations and goals]\n\n"
    "3. Target Market (Optional):\n"
    "   a. Segmentation:\n"
    "       - [Market segmentation details]\n"
    "   b. Potential Reach:\n"
    "       - [Potential reach information]\n"
    "   c. Market Size & Growth:\n"
    "       - [Market size and growth details]\n\n"
    "4. Company Summary (Optional):\n"
    "   a. Summary:\n"
    "       - [Company summary]\n\n"
    "In your analysis, consider the company's LinkedIn data presented in JSON format:\n"
    "{company_json}\n\n"
    "and the company website scrapped data:\n"
    "{company_data}\n\n"
)

email_prompt = PromptTemplate(
    input_variables=['company_data', 'person_json_data'],
    template="You are an email writer. Which writes hyper personal email, and use the relevent personal points to get the point across.\n"
    "You are give a company data and person's data in json format.\n" 
    "Use that to generate an email selling the company's product to the person.\n"
    "Choose a sender's name and his position in the company. \n"
    "Focus on why the person would benefit from the company's product and use his education, previous experience, country, skills, etc. to give a personal touch. Restrict to 3 paragraphs only. Keep the tone professional.\n"
    "company data: {company_data} \n\n"
    "person's json data: {person_json_data} \n"
)

import os
import openai


# Environment variables
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
PRODUCT_MODEL = os.environ.get('PRODUCT_MODEL')
HEADLINE_MODEL = os.environ.get('HEADLINE_MODEL')
SUBHEADLINE_MODEL = os.environ.get('SUBHEADLINE_MODEL')


# Classes
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def create_completion(model, prompt, temperature=0.5,
                      top_p=1,frequency_penalty=0,
                      presence_penalty=0, stop='.\n\n'):
    """
    Generate a completion with a new prompt using a pre-trained model

    Parameters
    ----------
    model: [string] ID of your pre-trained model.
    prompt: [string] Text block containing your prompt. Remember to use the
            same parameters as the ones you used to train your model.
            Finish your prompt with `\n\n###\n\n`.and
    temperature: [float] Control how creative the model should be. 1 = most
                  creative, 0 = more 'robotic' or strict.

    Returns
    -------
    JSON with the completion's parameters.
    """
    completion = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        max_tokens=60)

    return completion

def create_prompt(company='', features='', product='', headline=''):
    """
    Format a prompt from the provided parameters. 
    
    Parameters
    ----------
    company: [string] Name of the company.
    features: [string] List of features of the target product.
    product: [string] Description of the product.
    headline: [string] Website's headline.

    Returns
    -------
    prompt_type, prompt_args
    prompt_type: [int] Indicator that describes the type of prompt
                 0 = Company name and features.
                     Used to get a product description.
                 1 = Company name, features, and product description
                     Used to get a headline.
                 2 = Company name, features, product description and headline
                     Used to get a supporting headline.
    prompt_args: [string] Contains a formatted prompt ready to send to OpenAI's
                 API.
    """
    prompt_type = ''
    prompt_args = []
    
    # Company and features
    if len(company) == 0:
        raise ValueError('You have to provide a company name')
    elif len(features) == 0:
        raise ValueError("You have to provide the product's features")
    else:
        prompt_type = 0
        prompt_args.append(
            "company: {}\nfeatures: {}".format(company, features))

    # Product description
    if len(product) != 0:
        prompt_type = 1
        prompt_args.append("product: {}".format(product))

        if len(headline) != 0:
            prompt_type = 2
            prompt_args.append("headline: {}".format(headline))

    prompt_args.append("\n###\n\n")

    return "\n".join(prompt_args)

def create_copy(company_name, features, product=None):
    """
    Creates headline and subheadline.
    """
    # Product description creation
    if not product:
        print("Creating product description...", end=' ', flush=True)
        product_prompt = create_prompt(company=company_name, features=features)
        product = create_completion(PRODUCT_MODEL, product_prompt)
        product = product['choices'][0]['text']
        print("Done.")

    # Headline creation
    print("Creating headline...", end=' ', flush=True)
    headline_prompt = create_prompt(company=company_name, features=features,
                                    product=product)
    headline = create_completion(HEADLINE_MODEL, headline_prompt)
    headline = headline['choices'][0]['text']
    print("Done.")

    # Subheadline creation
    print("Creating subheadline...", end=' ', flush=True)
    subheadline_prompt = create_prompt(company=company_name, features=features,
                                    product=product, headline=headline)
    subheadline = create_completion(SUBHEADLINE_MODEL, subheadline_prompt)
    subheadline = subheadline['choices'][0]['text']
    print("Done.")

    return product, headline, subheadline


if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')

    openai.api_key = OPENAI_API_KEY

    print("\n","-"*50,"Input","-"*50)
    company = input("What's your company name? ")
    features = input("What are your product's features? ")
    print("\n")

    # Generate copy
    product, headline, subheadline = create_copy(company, features)

    # Results
    print("\n","-"*50,"Results","-"*50)
    print(f"{bcolors.HEADER}Company name:{bcolors.ENDC} {company}")
    print(f"{bcolors.HEADER}Features:{bcolors.ENDC} {features}")
    print(f"{bcolors.HEADER}Product description:{bcolors.ENDC} {product}")
    print(f"{bcolors.HEADER}Headline:{bcolors.ENDC} {headline}")
    print(f"{bcolors.HEADER}Subheadline:{bcolors.ENDC} {subheadline}")
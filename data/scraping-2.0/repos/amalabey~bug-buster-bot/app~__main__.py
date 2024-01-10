import argparse
from dotenv import load_dotenv
from app.azure_devops.azure_repo_pr_decorator_service import AzureRepoPullRequestDecoratorService
from app.openai.openai_feedback_provider import OpenAiFeedbackProvider
from app.openai.openai_method_provider import OpenAiMethodProvider
from app.semantic_changeset_provider import SemanticChangesetProvider
from app.azure_devops.azure_repo_changeset_provider import AzureRepoPullRequestChangesetProvider

# Create the argument parser
parser = argparse.ArgumentParser(description='Review code changes in a Pull Request using LLMs')

# Add arguments
parser.add_argument('-o', '--org', type=str, help='Name of the Azure DevOps organisation')
parser.add_argument('-p', '--project', type=str, help='Azure DevOps project name')
parser.add_argument('-i', '--id', type=str, help='Pull request id/number')

# Parse the command-line arguments
args = parser.parse_args()

# Access the values of the parsed arguments
az_org = args.org
az_project = args.project
pr_id = args.id

load_dotenv(".env")
load_dotenv(".env.local", override=True)

pr_change_provider = AzureRepoPullRequestChangesetProvider(az_org, az_project)
method_provider = OpenAiMethodProvider()
changeset_provider = SemanticChangesetProvider(pr_change_provider,
                                               method_provider)
changesets = changeset_provider.get_changesets(pr_id)

feedback_provider = OpenAiFeedbackProvider()
pr_decorator = AzureRepoPullRequestDecoratorService(az_org, az_project)
for changeset in changesets:
    feedback_list = feedback_provider.get_review_comments(changeset)
    for method, comments in feedback_list:
        pr_decorator.annotate(pr_id, changeset.path, method.startLine,
                              comments)
print("Done.")

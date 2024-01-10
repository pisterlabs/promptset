import openai
import boto3
from botocore.exceptions import NoCredentialsError
openai.api_key = ""

region = "us-west-2"
def get_iam_users():
    try:
        # Create an IAM client
        iam = boto3.client('iam')

        # Fetch all IAM users
        users = iam.list_users()

        # Print user details
        for user in users['Users']:
            print("User: {0}\nUserId: {1}\nARN: {2}\nCreatedOn: {3}\n\n".format(
                user['UserName'],
                user['UserId'],
                user['Arn'],
                user['CreateDate']
            ))

    except NoCredentialsError:
        print("No AWS credentials found.")


def getretiredinstances(region):
        """
        Returns a list of all instances being retired in the given region.

        Args:
          region: The region to check.

        Returns:
          A list of all instances being retired in the given region.
        """

        ec2 = boto3.resource('ec2', region_name=region)

        instances = []

        for instance in ec2.instances.all():
            if instance.state['Name'] == 'retired':
                instances.append(instance)

        return instances


def chatgptai():
    # list models
    models = openai.Model.list()

    # print the first model's id
    print(models.data[0].id)

    # create a chat completion
    chat_completion = openai.ChatCompletion.create(model="gpt-4",
                                               messages=[{"role": "user", "content": "your instance <instanceid>" +
                                                         "is scheduled for retirement;" +
                                                          "when do you want to schedule that"}])
    print(chat_completion.choices[0].message.content)

def startstoplambda():
  #create clouwatch event to start / stop via lambda at specific time user tells us
    return 0

def
if __name__ == "__main__":
    get_iam_users()
    getretiredinstances(region)

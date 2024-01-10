import aws_cdk as core
import aws_cdk.assertions as assertions
from guidance_for_buy_it_now_on_third_party_website_on_aws.guidance_for_buy_it_now_on_third_party_website_on_aws_stack import GuidanceForBuyItNowOnThirdPartyWebsiteOnAwsStack


def test_sqs_queue_created():
    app = core.App()
    stack = GuidanceForBuyItNowOnThirdPartyWebsiteOnAwsStack(app, "guidance-for-buy-it-now-on-third-party-website-on-aws")
    template = assertions.Template.from_stack(stack)

    template.has_resource_properties("AWS::SQS::Queue", {
        "VisibilityTimeout": 300
    })


def test_sns_topic_created():
    app = core.App()
    stack = GuidanceForBuyItNowOnThirdPartyWebsiteOnAwsStack(app, "guidance-for-buy-it-now-on-third-party-website-on-aws")
    template = assertions.Template.from_stack(stack)

    template.resource_count_is("AWS::SNS::Topic", 1)

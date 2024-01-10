from django.conf import settings
from django.core.mail import send_mail
import openai

def push_email_notification(recipient_email, firstname, is_newsletter=False):
    def generate_newsletter():
        openai.api_key = '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'

        response = openai.Completion.create(
        engine="text-davinci-003",
        prompt="Write an email to {} about the Charitable Gaming Tournament, in 250 words".format(firstname),
        max_tokens=300)

        return response.choices[0].text.strip()

    if is_newsletter == True:
        message = generate_newsletter()
    else:
        message = '''Hello {},
        We are delighted to invite you to our upcoming tournament.
        
        Kind Regards,
        Team 9'''.format(firstname)
    send_mail(
    subject='A cool subject',
    message=message,
    from_email=settings.EMAIL_HOST_USER,
    recipient_list=[recipient_email],
    auth_user=settings.EMAIL_HOST_USER,
    auth_password=settings.AUTH_PASS,
    fail_silently=False
    )

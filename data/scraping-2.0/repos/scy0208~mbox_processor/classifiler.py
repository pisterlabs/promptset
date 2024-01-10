import openai
from dotenv import load_dotenv
import os

load_dotenv()


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


openai.api_key = OPENAI_API_KEY

def format_system_prompt():
    return f"""
    First I will give you one response template

    Then, I will give you three email examples which this template is used to response, 
    please summarize the condition that this template was used to response, 
    
    Please output a short discription to summarize these type of emails, 
    so that next time when same kind of email come we can label it and use the template to response. 
    """

def format_user_prompt(template, email1, email2, email3):
    return f"""
    Response template: 
    \n```
    {template}
    ```
    \nEmail 1 responded by this template:
    ```
    {email1}
    ```
    \nEmail 2 responded by this template:
    ```
    {email2}
    ```
    \nEmail 3 responded by this template:
    ```
    {email3}
    ```
    """

def summarize_emails(template, email1, email2, email3):
    completion = openai.ChatCompletion.create(
        model="gpt-4-0613", 
        messages=[
            {"role": "system", 
             "content": format_system_prompt(),     
             },
            {"role": "user",
             "content": format_user_prompt(template, email1, email2, email3),
            }
        ]
    )
    return completion.choices[0].message.content

template = """
"Hi Dear,
It's so nice to hear back from you! We're super excited to initiate your first collab on Tiktok with us! Below are everything you need to know for our next steps.
Brief:Please click and read to better understand our brand and this collab: Hanakoko Collab Brief
Claim your PR packagePlease choose 4 sets you like through this exclusive link: Your Collab LinkIf the order is less than or equals 4 sets, it will automatically turn to $0.
Important: - Please leave your social media handle in the "note" section in your cart, otherwise we cannot identify your order 
ShippingShipping is free, on usYou'll receive the tracking info once your gift is shipped
Share and TagTo go with your content here is the discount code for your followers, please share the code Hanakoko15 with our website link hanakoko.com so that your fans can get 15% off all orders.
Or
Join our affiliate program where you can get your own code to earn 15% commission on all orders you generate. Just apply here: hanakoko affiliate link
Put in your info and an email with your own code and link will be sent to you. 
You can choose to share either code, the only difference is whether you're in to fill out a form to get commission. You can choose whichever works best for you. 

*Upon ordering, you automatically confirm that you've read the information listed above and agree to take part in this collab

Let me know if you have any other questions! Looking forward to our collab!
Thank you,Hana

-- 
Hanakoko | Be your own nail artist at homeShop www.hanakoko.comFollow us @hanakoko_official on all socialsBe an affiliate & Earn commissions: Apply here 
"""

email1 = """
"Subject:
Re: We Want To Collab With You| Hanakoko Nail Products
From:
"hidden email address"
Date:
7/24/23, 12:34 PM
To:
Hanakoko <official@hanakoko.com>
I am interested!!
On Mon, Jul 24, 2023 at 7:11 AM Hanakoko <official@hanakoko.com> wrote:
    Showcase your nail style with our PR box ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­
    Hi beautiful!
     
    So nice to meet you! I'm Hana, running a nail brand Hanakoko with my sisters. We're really impressed by your content and would like to have you try on and help promote our products.
     
    We currently have over 800 designs of easy nail products, our exquisite hand-made press-ons and real gel stickers can get you a stunning salon-quality gel nailart in minutes.
     
    Take a look at our site: www.hanakoko.com
    What we offer:
    1) A gift box of products: a free link to pick your fav styles
    2) Free shipping to you in US/Canada/Australia/Most countries in Europe.
     
    What we want from you:
    A video of using the product post on your most used social platform, tagging us.
     
    Due to budget constraints, this collab is not paid with cash, but you'll get $100 value of nail products with superb designs and discounts for you and your followers.
     
    You can also join our affiliate program to earn commissions on every order: Be an ambassador!
     
    Please let me know if you're ok with this offer and if so, we can send you the offer in detail.
     
    Look forward to hearing from you!
     
    Best regards,
    Hana
    Instagram
    YouTube
    Tiktok
    You received this email from Hanakoko.
    If you would like to unsubscribe, click here."
"""

email2 = """
"Subject:
Re: We Want To Collab With You| Hanakoko Nail Products
From:
"hidden email address"
Date:
7/24/23, 5:25 AM
To:
Hanakoko <official@hanakoko.com>
Hello and thank you for your interest in me. I would love to work together please tell me more about this collaboration. 
Best Regards - Alexis ❤️
> On Jul 24, 2023, at 8:11 AM, Hanakoko <official@hanakoko.com> wrote:
>
> ﻿
> Showcase your nail style with our PR box ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­
> Hi beautiful!
>  
> So nice to meet you! I'm Hana, running a nail brand Hanakoko with my sisters. We're really impressed by your content and would like to have you try on and help promote our products.
>  
> We currently have over 800 designs of easy nail products, our exquisite hand-made press-ons and real gel stickers can get you a stunning salon-quality gel nailart in minutes.
>  
> Take a look at our site: www.hanakoko.com
> What we offer:
> 1) A gift box of products: a free link to pick your fav styles
> 2) Free shipping to you in US/Canada/Australia/Most countries in Europe.
>  
> What we want from you:
> A video of using the product post on your most used social platform, tagging us.
>  
> Due to budget constraints, this collab is not paid with cash, but you'll get $100 value of nail products with superb designs and discounts for you and your followers.
>  
> You can also join our affiliate program to earn commissions on every order: Be an ambassador!
>  
> Please let me know if you're ok with this offer and if so, we can send you the offer in detail.
>  
> Look forward to hearing from you!
>  
> Best regards,
> Hana
> Instagram
> YouTube
> Tiktok
>
> You received this email from Hanakoko.
> If you would like to unsubscribe, click here."
"""

email3 = """
"Subject:
Re: We Want To Collab With You| Hanakoko Nail Products
From:
"hidden email address"
Date:
7/24/23, 5:17 AM
To:
Hanakoko <official@hanakoko.com>
Hello Hana!
I would love to receive your products and would love to create a TikTok video that I can also share on my IG.
Please let me know what information you need
Stephanie 
> On Jul 24, 2023, at 8:11 AM, Hanakoko <official@hanakoko.com> wrote:
>
> ﻿
> Showcase your nail style with our PR box ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏  ͏ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­ ­
> Hi beautiful!
>  
> So nice to meet you! I'm Hana, running a nail brand Hanakoko with my sisters. We're really impressed by your content and would like to have you try on and help promote our products.
>  
> We currently have over 800 designs of easy nail products, our exquisite hand-made press-ons and real gel stickers can get you a stunning salon-quality gel nailart in minutes.
>  
> Take a look at our site: www.hanakoko.com
> What we offer:
> 1) A gift box of products: a free link to pick your fav styles
> 2) Free shipping to you in US/Canada/Australia/Most countries in Europe.
>  
> What we want from you:
> A video of using the product post on your most used social platform, tagging us.
>  
> Due to budget constraints, this collab is not paid with cash, but you'll get $100 value of nail products with superb designs and discounts for you and your followers.
>  
> You can also join our affiliate program to earn commissions on every order: Be an ambassador!
>  
> Please let me know if you're ok with this offer and if so, we can send you the offer in detail.
>  
> Look forward to hearing from you!
>  
> Best regards,
> Hana
> Instagram
> YouTube
> Tiktok
>
>
> You received this email from Hanakoko.
> If you would like to unsubscribe, click here."
"""

def main():
    # process_emails("Others.csv", 500, "Others_output.csv")
    print(summarize_emails(template, email1, email2, email3))


if __name__ == '__main__':
    main()
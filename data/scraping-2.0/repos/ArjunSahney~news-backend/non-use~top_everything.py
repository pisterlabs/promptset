from newsapi import NewsApiClient
import requests
import openai
import re
from categorizer import *
import requests
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain import OpenAI, PromptTemplate
import glob
from langchain.document_loaders import JSONLoader

# Initialize the NewsApiClient
api_key = '785379e735a84282af1c6b35cf335a59'
newsapi = NewsApiClient(api_key=api_key)

def fetch_articles_from_source(source, num_articles=100):
    articles_list = []
    
    # Calculate the number of pages needed
    pages_needed = (num_articles + 99) // 100  # Using integer division to round up
    
    for page in range(1, pages_needed + 1):
        response = newsapi.get_everything(sources=source, page_size=100, page=page, language='en')
        articles_list.extend(response.get('articles', []))
    
    # Trim the list to the desired number of articles
    return articles_list[:num_articles]

def fetch_and_store_articles_by_category_and_source():
    sources_by_category = {
        "N": ["reuters", "associated-press", "axios"],  # Adjusted for 4 sources 
        "B": ["business-insider", "financial-times"],  # Added bloomberg
        "T": ["techcrunch", "the-verge", "engadget"]  # Added the-verge
    }
    
    num_articles_by_category = {
        "news": 100,  # Now we'll fetch 50 from each of the 4 sources
        "business": 100,  # 50 each from business-insider and bloomberg
        "technology": 100  # 50 each from techcrunch and the-verge
    }
    
    articles_by_category = {}
    for category, sources in sources_by_category.items():
        articles_by_category[category] = []
        num_articles_per_source = num_articles_by_category[category] // len(sources)
        
        for source in sources:
            articles_by_category[category].extend(fetch_articles_from_source(source, num_articles_per_source))
    
    return articles_by_category

def display_articles_by_category_and_source():
    articles = fetch_and_store_articles_by_category_and_source()
    for category, articles_list in articles.items():
        print(f"{category.title()} News:")
        for idx, article in enumerate(articles_list, 1):
            print(idx, article['title'])
        print()

# Driver code
if __name__ == '__main__':
    display_articles_by_category_and_source()

#Sample Output: 
# News News:
# 1 Show HN: Screenshot AI, Your Intelligent Screenshot Assistant
# 2 Google Groups ending support for Usenet
# 3 Show HN: Turn any YouTube video into a blog post
# 4 Show HN: ZumaPiano - AI powered Chrome extension to learn piano
# 5 Escape room en casa pdf gratis - eastcoastapparel.shop
# 6 Show HN: Stop repetitively typing times slots for meetings
# 7 Enabling next-generation AI workloads: Announcing TPU v5p and AI Hypercomputer
# 8 Show HN: Apply AI filters to your videos – Android
# 9 New Book Reveals How I Built A 7-Figure Online Business Using Nothing But Ethical Email Marketing To Drive Revenue, Sales and Commissions...
# 10 Video mode now only show pages where video is the main content
# 11 Adding markup support for vacation rentals
# 12 Show HN: Keep Twitter scroll position" Chrome extension
# 13 Users are massively giving their 1-star reviews to AdBlocker
# 14 Affiliate Marketing Tutorial For Beginners 2023
# 15 Show HN: Have AI review your Techstars application to help you get in
# 16 Expanding markup support for Organization details, replacing the Logo feature
# 17 Powering cost-efficient AI inference at scale with Cloud TPU v5e on GKE
# 18 Show HN: Gmail replies with GPT: context-aware and personalized
# 19 New in structured data: discussion forum and profile page markup
# 20 Show HN: Why is forecasting so hard?
# 21 Show HN: Magictool.ai – Privacy-First AI Productivity App with 20 AI Features
# 22 Show HN: AIHairstylist – Try Hairstyles Instantly
# 23 Show HN: Blenny, AI vision copilot for the web
# 24 Upcoming deprecation of Crawl Rate Limiter Tool in Search Console
# 25 Show HN: Sensus – Constructive PR Comments
# 26 Show HN: The 30 second habit that can have a big impact on your life
# 27 Show HN: I made a Chrome extension for LeetCode 1v1s to make DSA practice fun
# 28 Show HN: An Undetectable YouTube Adblocker
# 29 Show HN: Tabbana – manage browser tabs via instructions (Chrome extension)
# 30 Show HN: Eye Relief Workouts in Chrome
# 31 Web Content Accessibility Guidelines 2.2 in simple language
# 32 Show HN: Devtools Responsive Mode on Steroids
# 33 Show HN: Blink – A Chrome Extension to Combat ADHD and Boost Focus

# Business News:
# 1 Quaker Oats is recalling its Chewy Bars over salmonella fears
# 2 For decades, Biden has lavished attention on Philadelphia. It might save his presidency.
# 3 Ukrainian special ops commandos are freelancing top-secret sabotage missions, poisonings, and assassinations in Russia, says military source
# 4 This company is fighting the gender pay gap by teaching women and girls to play poker
# 5 Barry Keoghan says the shocking grave scene in 'Saltburn' was tamer in the original script: 'He wasn't written to do that'
# 6 I go to Farmer Boys for breakfast every chance I get. Here's why I think the chain should come to every US state.
# 7 The woman who was branded 'Australia's worst mother' has had her convictions for killing her 4 children overturned
# 8 Elon Musk tells Italians to have more children during an appearance at a political conference in Rome: 'Make more Italians'
# 9 2 derelict homes priced at 65 cents each sold for over $50,000
# 10 Ukrainian troops fighting on Dnipro river say official claims of success are misleading: 'There are no positions. It's a suicide mission'
# 11 BRICS-led dedollarization should alarm the US as new members could be most aggressive against the greenback, former State Dept. official says
# 12 7 ways people ruin their own vacations — and what a travel planner would do differently
# 13 Here are what Wall Street sees as the big risks to stocks in 2024
# 14 Diddy returns to Instagram to post a tribute to his late ex-girlfriend Kim Porter amid his sexual assault allegations
# 15 People tell me I have a 'lazy girl job,' but I call it a healthy job. You don't have to burn out to be successful.
# 16 The war in Ukraine is driving new interest in wheeled armored vehicles, but it's also putting their weaknesses on display
# 17 UPS drivers might make more than you think — here are 8 reasons why their pay can reach $170K
# 18 Italian family who secretly guarded $109 million Botticelli for decades to protect it from theft return it
# 19 Elon Musk's private jets took 441 flights this year
# 20 My family spent last Christmas on a cruise ship. We'd never do it again.
# 21 Despite growing up poor, I always had Christmas gifts from Santa. As an adult, I discovered the truth.
# 22 Pad thai was promoted by the Thai government as noodle 'propaganda'
# 23 A new residential cruise ship startup with condos starting at $100,000 will let travelers live at sea — take a look around the ship
# 24 I asked my 95-year-old best friend for longevity tips. Some of his answers surprised me.
# 25 Audi driver says their car's Pre Sense safety feature stopped the vehicle in the middle of a highway and almost killed them
# 26 WNBA salaries max out nearly $865K less than the NBA's lowest contracts. Players lean on side jobs and savvy financial strategies to fill the gap.
# 27 My daughter's crying fit on an overnight flight got me dirty looks from a woman. I realized the problem was her, not us.
# 28 Trump said undocumented immigrants were 'poisoning the blood of our country,' in speech Biden campaign called 'parroting Adolf Hitler'
# 29 The largest fusion reactor in the world fired up in Japan. Here's how the $600 million device compares to the US's revolutionary fusion machine.
# 30 I moved from the US to the UK and had to completely change the way I cleaned
# 31 A fisherman found a 152-year-old shipwreck with his kid, thought it was nothing special, and kept fishing
# 32 Jamie Dornan says some 'feverish' fans of '50 Shades' think he has a secret child with costar Dakota Johnson
# 33 Side hustles to make some extra cash this holiday season
# 34 The long, brutal reign of Wall Street's private-equity kings is over
# 35 LinkedIn's Dan Shapero on 2024: AI will make our day-to-day lives easier
# 36 New York City needs China's economy to bounce back
# 37 Some student-loan borrowers might get up to $20,000 in debt cancellation through Biden's new plan — but experts who helped craft it are pushing for even more
# 38 Momfluencer Ruby Franke will plead guilty to charges of child abuse, and believes Jodi Hildebrandt 'manipulated' her, attorney says
# 39 The 10 products that have shrunk the most under shrinkflation
# 40 These modular, net-zero homes with Scandinavian flair cost from $186,100 and can be put almost anywhere — take a look
# 41 I got paid to let TikTok users decide how I spent my money for a month. Their choices surprised me.
# 42 Tucker Carlson and Elon Musk have rallied behind an American 'red pill' dating coach turned Russian propagandist detained in Ukraine
# 43 I paid $20 for a ticket to IKEA's all-you-can-eat annual Swedish Christmas feast — here's what it was like to attend 'Julbord' in NYC
# 44 2 planes narrowly avoided colliding on a Colorado runway after a pilot's quick-thinking maneuver
# 45 The billionaire founder behind China's AI giant SenseTime has died of an unknown illness at 55
# 46 What mice in VR goggles can tell us about our brains
# 47 Epic Games CEO fears Google may still be able to stifle Play Store competitors
# 48 Former Amazon engineer pleads guilty to stealing $12.3 million of crypto in first ever hacking case involving smart contracts
# 49 Tesla's recall on its Autosteer technology could bolster lawsuits claiming the feature is dangerous: report
# 50 Senegalese herders have raised livestock for centuries. Climate change is threatening their future.
# 51 Applesauce pouches that gave children lead poisoning may have been contaminated on purpose, FDA says
# 52 UFC 296 live stream: Where to watch Edwards vs. Covington, Pantoja vs. Royval
# 53 Mississippi cop who shot an 11-year-old in the chest after his mother called 911 for help will face no criminal charges
# 54 Officers rescue 150 animals 'living in squalor' from a small garage. The owner now faces animal cruelty charges.
# 55 Tiger Woods' 16-year-old daughter Sam acted as his caddie for the first time. Here are 5 photos of the Woods family at the 2023 PNC Championship.
# 56 Hackers behind recent ChatGPT outage say they'll target the AI bot until it stops 'dehumanizing' Palestinians
# 57 Twitch walks back new rules allowing 'artistic nudity' after just 2 days due to concerns over AI-created deepfakes
# 58 Homelessness among American veterans increased this year by its largest margin since the US started tracking
# 59 Kevin McCarthy's hand-picked successor can't run for the departing lawmaker's seat because he's already running in another race, California officials say
# 60 Russia's new war plan is to occupy more Ukrainian territory by 2026, report says
# 61 How Moms for Liberty co-founder Bridget Ziegler, a supporter of 'parental rights' in schools, got caught in a sex scandal that may push her to resign from a school board
# 62 'The Crown' season 6 shows Prince William and Kate Middleton's relationships before they got together. Here's what you need to know about their real-life exes.
# 63 Mississippi sheriff overseeing violent department faces yet another lawsuit after yet another man dies in its custody
# 64 TikToker Danny Loves Pasta shares the biggest mistake people make with homemade pasta dough
# 65 3 Israeli hostages mistakenly killed by the IDF were shirtless and holding a white flag, military official says
# 66 The tallest hotel in Las Vegas just opened after more than a decade of construction. 11 photos offer a glimpse inside the glitzy Fontainebleau.
# 67 Large AI models can now create smaller AI tools without humans and train them like a 'big brother,' scientists say
# 68 Ron DeSantis says Trump will claim the Iowa caucuses were 'stolen' from him if he loses the pivotal contest: 'He will try to delegitimize the results'
# 69 Gary Oldman says 'Harry Potter' and 'The Dark Knight' movies allowed him to do 'the least amount of work for the most amount of money'
# 70 Taylor Swift, the patron saint of surviving breakups, once made a curated playlist for Jessica Chastain to help her get over her ex
# 71 MSC Cruises says a passenger jumped from one of its ships while sailing from Europe to South America
# 72 GM's CEO said 2023 would be 'a breakout year' for EV production. But demand has fallen sharply.
# 73 I climbed a terrifying 1,200 feet to stay in a transparent pod suspended off the side of a cliff in Peru
# 74 Manhattan's Equinox Hotel was just named one of the top 50 in the world. See inside its ultra-luxe penthouse, which costs $11,000 a night.
# 75 8 mistakes people make when getting ankle tattoos, according to tattoo artists
# 76 Matthew Perry wrote in his memoir that ketamine, the drug that led to his death, had his 'name written all over it'
# 77 Working retail for over a decade made me hate everything about the holidays. Here's why.
# 78 The Fed surprised markets with a sharp pivot on 2024 rate cuts. Here's how Wall Street is changing its forecasts.
# 79 He did it once, and he'll do it again: The GOP primary is structured to give Trump a glide path to nomination
# 80 Carbon markets are supposed to fight climate change. But prices have collapsed with no solution in sight.
# 81 The oil industry's mega-merger spree and a US production boom point to strong crude demand for years to come
# 82 Upwork's Kelly Monahan on why 2024 should be about embracing job disruption
# 83 Harvard early applications are down 17% from last year amid antisemitism row
# 84 An engineer started a food blog after a quarter-life crisis. When she was laid off, she saw an opportunity and turned it into a lucrative career.
# 85 All the best photos from the 2005 premiere of 'Charlie and the Chocolate Factory'
# 86 Trying to spend the holidays with both sides of extended family was tearing us apart. Now we put our nuclear family's needs first.
# 87 A Ukrainian councilor set off grenades in a meeting over an apparent pay dispute, video shows
# 88 35 of the most influential hip-hop songs in music history
# 89 House Republicans urge Pentagon to block removal of a Confederate memorial from Arlington National Cemetery
# 90 The 3 simple ingredients that will transform your holiday cookies
# 91 Palestinian high school student in Florida expelled over mother's 'hateful and incendiary' social media posts
# 92 My fear of snow stopped me from running outside. Here's how I regained my love of winter runs.
# 93 Russians appear to troll Putin with critical texts at normally carefully orchestrated TV Q&A session
# 94 3 mistakes you're probably making with your dining room decor, according to an interior stylist
# 95 We moved to NYC in our 80s because we needed more help as we got older. Our home is tiny, but we love the convenience.
# 96 HENRYs are turning to a little-known and risky type of mortgage as home prices soar
# 97 Legendary investor Jeremy Grantham warns of a 'superbubble' and looming recession — and touches on Elon Musk and housing woes
# 98 What is Israel trying to achieve in its brutal Gaza war?
# 99 My husband and I want kids, but as a queer couple, we have limited options. We love the holidays, but they remind us of what we don't have yet.
# 100 This list of the 10 most reliable cars doesn't have a single US automaker on it

# Technology News:
# 1 Apple again targets iMessage-on-Android app Beeper, but company claims it has a fix
# 2 Kapital secures $165M in equity, debt to provide financial visibility to LatAm SMBs
# 3 It looks like outgoing X (Twitter) links are broken
# 4 Apple will no longer give police users’ push notification data without a warrant
# 5 What X needs most now is for Snap to post a solid Q4
# 6 Apple introduces protection to prevent thieves from getting your passwords
# 7 Apple adds iPhone 15 and additional M2 Macs to Self-Service Repair program
# 8 Metafuels lands $8 million bet on greener skies ahead
# 9 Procurement software startup Pivot raises $21.6 million just a few months after its creation
# 10 E3 has entertained its last electronic expo
# 11 Apple’s new Journal app is now available with the release of iOS 17.2
# 12 Beeper Mini is back in operation after Apple’s attempt to shut it down
# 13 Docker acquires AtomicJar, a testing startup that raised $25M in January | TechCrunch
# 14 SumUp taps €285M more in growth funding to weather the fintech storm | TechCrunch
# 15 Early impressions of Google’s Gemini aren’t great • TechCrunch
# 16 Google fakes an AI demo, Grand Theft Auto VI goes viral and Spotify cuts jobs | TechCrunch
# 17 Do you believe in job after job? | TechCrunch
# 18 Mulch and the enduring appeal of internet absurdism | TechCrunch
# 19 Google's best Gemini demo was faked
# 20 Seattle biotech hub pursues 'DNA typewriter' tech with $75M from tech billionaires | TechCrunch
# 21 Zelda Ventures’ new pre-seed fund backs serial entrepreneurs | TechCrunch
# 22 Meta finally starts rolling out default end-to-end encryption for Messenger | TechCrunch
# 23 Sona launches its music streaming platform and marketplace to reward fans for buying 'digital twins' of songs | TechCrunch
# 24 Google's Gemini isn't the generative AI model we expected | TechCrunch
# 25 CISA says US government agency was hacked thanks to 'end of life' software | TechCrunch
# 26 US senator warns governments are spying on Apple and Google users via push notifications
# 27 iMessage will reportedly get a reprieve from EU’s interoperability regulation
# 28 Sydney-based generative AI art platform Leonardo.Ai raises $31M
# 29 Twitch to shut down in Korea over 'prohibitively expensive' network fees | TechCrunch
# 30 Fiat's new EV looks like the anti-Cybertruck | TechCrunch
# 31 X is now licensed for payment processing in a dozen US states
# 32 Meta set to discontinue cross-messaging between Instagram and Facebook | TechCrunch
# 33 Cruise faces fines in California for withholding key details in robotaxi accident | TechCrunch
# 34 Atomic Industries closes $17M seed to exascale America's industrial base | TechCrunch
# 35 Tesla's cheapest vehicle is losing half its tax credit next year | TechCrunch
# 36 VC Office Hours: Unlocking the Farmers’ Market with Black Farmer Fund | TechCrunch
# 37 Anduril unveils Roadrunner, 'a fighter jet weapon that lands like a Falcon 9' | TechCrunch
# 38 Sample Seed pitch deck: Scalestack's $1M deck
# 39 Apple releases security updates for iOS, iPadOS and macOS, fixing two actively exploited zero-days
# 40 Yieldstreet to acquire real estate investment platform Cadre
# 41 Instagram Threads search now supports ‘all languages’ in latest update
# 42 X CEO Linda Yaccarino publicly backs Musk after he says ‘f*ck yourself’ to advertisers
# 43 On ChatGPT's first anniversary, its mobile apps have topped 110M installs and nearly $30M in revenue | TechCrunch
# 44 South African startup GoMetro gets £9M for its fleet management optimization software | TechCrunch
# 45 Elon Musk says ‘go fuck yourself’ to advertisers leaving X
# 46 Founder of spyware maker Hacking Team arrested for attempted murder: local media | TechCrunch
# 47 Amazon finally releases its own AI-powered image generator | TechCrunch
# 48 Webull leaps into Mexico with acquisition of stock trading app Flink
# 49 Apple Music Replay is here, and it's still no Spotify Wrapped | TechCrunch
# 50 Tumblr+ kills Post+, its ill-fated subscription offering for creators | TechCrunch
# 51 The ultimate app for reading the internet
# 52 US Congress pushes warrantless wiretapping decision off until April next year
# 53 Opal’s tiny, laptop-friendly Tadpole webcam is already 20 percent off
# 54 Apple makes it easier for app makers to compete for your dollars
# 55 The Epic question: how Google lost when Apple won
# 56 The gulf between the real world and streaming has never been further
# 57 Inside the strange and stunning alien world of Scavengers Reign
# 58 OpenAI suspends ByteDance's account after it used GPT to train its own AI model
# 59 California settles Activision Blizzard gender discrimination lawsuit for $54 million
# 60 Philips Hue reorganizes, plans job cuts to save $218 million annually
# 61 Apple fixed the iPhone’s Flipper Zero problem
# 62 Adam Mosseri spells out Threads’ plans for the fediverse
# 63 Researchers say Bing made up facts about European elections
# 64 You could be eligible for a piece of Apple’s Family Sharing settlement
# 65 Climate change is killing coral — can AI help protect the reefs?
# 66 Quest owners can use Word, Excel, and PowerPoint in VR, but do you really want to?
# 67 Google gives Stadia controllers more time to switch to Bluetooth
# 68 Twitch immediately rescinds its artistic nudity policy
# 69 Google will update Maps to prevent authorities from accessing location history data
# 70 Nanoleaf’s smart holiday string lights are over half off
# 71 How Lego builds a new Lego set
# 72 Vivo’s X100 Pro offers another massive camera sensor to an international audience
# 73 Naughty Dog cancels its The Last of Us multiplayer game
# 74 Mailchimp cancels podcast after refusing to work with union producers
# 75 Grimes has a new line of AI plush toys, including one named Grok
# 76 The EU AI Act passed — now comes the waiting
# 77 Apple is making a Murderbot series starring Alexander Skarsgård
# 78 Netflix’s Yu Yu Hakusho needed more time in the spirit world
# 79 Google releases on-device diagnostics tool and repair manuals for Pixel phones
# 80 Temu sues Shein, alleging ‘Mafia-style’ intimidation of manufacturers
# 81 Neko Atsume on Quest is cute, cuddly, and a great use of mixed reality
# 82 Cruise lays off nearly a quarter of the company after GM slashes driverless spending
# 83 How to easily find the GTA trilogy (and other games) on Netflix
# 84 Cable service cancellation fees might be on the way out
# 85 Apple’s M2-powered MacBook Airs are up to $250 off
# 86 YouTube will have fewer ad breaks on TV — but the ads are getting longer
# 87 AT&T is Rivian’s latest EV customer following Amazon deal
# 88 Intel’s Core Ultra CPUs are here — and they all come with silicon dedicated to AI
# 89 Elon Musk wants to open a university in Texas
# 90 GM announces Cadillac Vistiq, a midsize electric SUV coming in 2025
# 91 The latest Apple Watch integration makes custom workouts less of a headache
# 92 Google will turn off third-party tracking for some Chrome users soon
# 93 H&R Block launches AI tax filing assistant
# 94 Google’s Nest Renew joins Alphabet spinoff to form Renew Home
# 95 Threads launches for nearly half a billion more users in Europe
# 96 Proton Mail finally gets a desktop app for encrypted email and calendar
# 97 Opera’s gamer browser now has a ‘panic button’ for when you’re caught in the act
# 98 Beeper says some people aren’t getting iMessages again
# 99 The Tesla Cybertruck’s infamous wiper will reportedly cost $165 to replace
# 100 Twitch loosens its policy on sexual content

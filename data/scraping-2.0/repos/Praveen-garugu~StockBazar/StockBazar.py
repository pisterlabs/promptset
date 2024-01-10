import streamlit as st
import openai
import streamlit_authenticator as stauth
from dependancies import fetch_users,signup
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import time
import random
import yfinance as yf
from streamlit_lottie import st_lottie
import json
import requests
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import pandas_ta as ta
from PIL import Image
import requests
from stocksmatter import stockmarket
from io import BytesIO
import yahooquery as yq
import streamlit.components.v1 as components
from texttoaudio import speech
from courseshtml import telugu,hindi,english
from predictapp import stockPredict
from Indicator import indicate
from courses import c
html_code = '''
<!DOCTYPE html>
<html lang="en">
<head>
<title>Quiz App</title>
<style>
body {
  font-family: Arial, sans-serif;
  background-color: #f4f4f4;
  margin-left: 0;
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
}
.quiz-container {
  background-color: white;
  padding: 0px;
  border-radius: 10px;
  box-shadow: rgb(38, 57, 77) 0px 20px 30px -10px;
  text-align: center;
  margin-left:58px;
  margin-top:-26px;
height:480px;
width:530px;
}
h1 {
  margin-bottom: 20px;
}
#options {
  margin-top: 30px; 
}
#question{
font-size:19px;
font-weight:bold;
margin-top:20px;
padding:5px;
}
button {
  display: block;
  margin: 10px auto;
  padding: 10px 10px;
height:60px;
width:330px;
margin-top:-2px;
  color: black;
  border: none;
  border-radius: 6px;
  cursor: pointer;
font-size:16px;
 background-color: lightgray;
border:3px solid lightblue;
}
button:hover {
  background-color:gray;
color:white;
}
#submit-btn{
 padding: 10px 20px;
 background-color:#1648ff;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
height:40px;
width:110px;
margin-top:28px;
}
#submit-btn:hover{
background-color:#479bfe;
}
#result{
font-size:29px;
margin-top:60px;
font-weight:bold;
}
</style>
</head>
<body>
  <div class="quiz-container">
    <div id="question"></div>
    <div id="options"></div>
    <button id="submit-btn">Next</button>
    <div id="result"></div>
  </div>
<script>
const questionElement = document.getElementById('question');
const optionsElement = document.getElementById('options');
const submitButton = document.getElementById('submit-btn');
const resultElement = document.getElementById('result');
const questions = [
 {
        question: "1. What are the indices NIFTY and SENSEX are dependent on?",
        options: ["Capital that has been paid in full.", "The market capitalisation of a company", "Capitalisation based on free float", "Share capital that has been authorised."],
        correctAnswer: "Capitalisation based on free float",
    },
  {
        question: "2. What is the Stock Exchange Sensitive Indexes(Sensex) total number of companies?",
        options: ["50", "60", "30", "40"],
        correctAnswer: "30",
    },
  {
        question: "3. The feature of shares in primary markets that makes it very easy to sell recently issued securities is known as",
        options: ["The flow of money", "Reduction in liquidity", "Large fund", "Liquidity increase"],
        correctAnswer: "Liquidity increase",
    },
  {
        question: "4. Where is the National Stock Exchange headquartered?",
        options: ["Delhi", "Kolkata", "Bombay", "Chennai"],
        correctAnswer: "Bombay",
    },
  {
        question: "5. When was the National Stock Exchange Fifty (NIFTY) founded?",
        options: ["1991", "1993", "1996", "1995"],
        correctAnswer: "1996",
    },
  {
        question: "6. What is the purpose of a stock index?",
        options: ["To regulate stock trading", "To predict future stock prices", "To measure the overall performance of a group of stocks", "To set dividend payouts for companies"],
        correctAnswer: "To measure the overall performance of a group of stocks",
    },
      {
        question: "7. Which of the following is a key measure of a company's profitability?",
        options: ["Dividend Yield", "Market Capitalization", "Earnings Per Share (EPS)", " Price-to-Earnings Ratio (P/E Ratio)"],
        correctAnswer: "Earnings Per Share (EPS)",
    },
      {
        question: "8. _____ is not the concern of the Securities and Exchange Board of India (SEBI).",
        options: ["Ensure that businesses operate in an ethical manner", "Raising the earnings of the company’s shareholders", "Brokers promoting efficient services", "Investor protection is crucial."],
        correctAnswer: "Raising the earnings of the company’s shareholders",
    },
      {
        question: "9. The first computerised stock exchange in India was",
        options: ["Multi Commodity Exchange (MCX)", "Bombay Stock Exchange (BSE)", "Over-the-Counter Exchange of India (OCTEI)", "National Stock Exchange (NSE)"],
        correctAnswer: "National Stock Exchange (NSE)",
    },  
     {
        question: "10. Which of the following investment options is generally considered to be the riskiest?",
        options: ["Government Bonds", "Penny Stocks", "Index Funds", " Blue-chip Stocks"],
        correctAnswer: "Penny Stocks",
    }
];
let currentQuestion = 0;
let score = 0;
function loadQuestion() {
  if (currentQuestion < questions.length) {
    questionElement.textContent = questions[currentQuestion].question;
    optionsElement.innerHTML = '';
    questions[currentQuestion].options.forEach((option, index) => {
      const optionButton = document.createElement('button');
      optionButton.textContent = option;
      optionButton.addEventListener('click', () => checkAnswer(index));
      optionsElement.appendChild(optionButton);
    });
    submitButton.textContent = 'Submit';
    resultElement.textContent = '';
  } else {
    showResult();
  }
}
function checkAnswer(selectedIndex) {
  optionsElement.childNodes.forEach(button => button.disabled = true);
  const selectedButton = optionsElement.childNodes[selectedIndex];
  if (questions[currentQuestion].options[selectedIndex] === questions[currentQuestion].correctAnswer) {
    selectedButton.style.backgroundColor = '#00ff2a';
    score++;
  } else {
    selectedButton.style.backgroundColor = '#ff2a2a';
    const correctIndex = questions[currentQuestion].options.findIndex(option => option === questions[currentQuestion].correctAnswer);
    const correctButton = optionsElement.childNodes[correctIndex];
    correctButton.style.backgroundColor = '#00ff2a';
  }
  submitButton.textContent = 'Next';
  submitButton.disabled = false;
}
function showResult() {
  questionElement.textContent = '';
  optionsElement.innerHTML = '';
 resultElement.textContent = `Quiz Completed`;
  resultElement.textContent = `Your Score: ${score} out of ${questions.length}`;
  resultElement.style.color = 'black';
  submitButton.style.display = 'none';
}
submitButton.addEventListener('click', () => {
  if (submitButton.textContent === 'Next') {
    currentQuestion++;
    loadQuestion();
  }
});
loadQuestion();
</script>
</body>
</html>
'''

d={
    'hi':'hello, Buddy',
    'how are you':'I am fine',
    'What are you doing':'Thinking about you buddy ',
    'are you all right':'ya iam performing well',
    'What are the advantages of Derivatives':'Enhance price discovery process and volume of transcations',
    'difference between nse and bse':'BSE consists of 30 scrips whereas NSE consists of 50 scrips,BSE is screen based trading whereas NSE is ringless, national, computerized exchange.',
    'On what basis securities should be selected':'Yield to maturity,1)Risk to default,2)Tax shield and Liquidity.',
    'what does the website provides':'website provides the information regarding stock market that help investors to make informed decisions',
    'how is the website different from other websities': 'the additional features that are been added are 1) chatbot: that gives you the clear understanding of an website 2) all the information regarding stock market are embedded inorder to provide efficiency of usage 3)  login crediantals to provide security to the data that are been accesessed by the user etc.',
    'what is your name':'i am an ai',
    'market open time':'The regular trading days open at  9.30 a.m. Eastern time',
    'market close time':'the regular trading days closes at 4 p.m. Eastern time.',
    'what is a put in stocks':'A put in stocks is an options contract that represents the right to sell a particular stock at a set price within a certain time frame.',
    'how many stocks should i buy in my portfolio':"investors can have 8-10 stocks in the portfolio depending on the amount of investment",
    'how much returns can i expect from the market':'In the bull market, the portfolio will give attractive returns (the benchmark index Nifty gave a return of ~67% from April 01,2020 till December 18,2020.',
    'is investing in IPO is a better option':'few IPOs have given amazing returns to their shareholders in the past for long consistent years. If you are able to find such IPOs which are very promising,then feel free to invest in them.',
    'stock':'ownership of an company',
    'What are stock exchanges':'a market in which securities are bought and sold,the company was floated on the stock exchange',
    'What is a stock index':'A stock index is a measurement of the value of a specific group of stocks.',
    'How do stock prices change':'Stock prices change due to various factors such as company performance, market trends, economic indicators, news, and investor sentiment.',
    'What is a dividend':'A dividend is a payment made by a company to its shareholders from its profits.',
    'how can one can identify the best stocks':'A Strong Leadership Team.A Promising Growth In+++dustry.Commanding Market Share.Strong Sales Growth.A Large Target Market.',
    'what is stock market':"The stock market refers to a marketplace where people buy and sell shares of publicly traded companies.",
    'is this website secure':'yes, it is secure as we are using streamlit to build application',
    'are there any courses provided by this website':'yes u can refer to course webpage for better learning',
    'how many sectors are there to invest in Stock market':'You can invest in 11 different sectors in the stock market.',
    'is there any time for buying shares or doing a trade':'Yes, you can only trade between 09:15 am to 3:30 pm on weekdays. But you can place AMO type of orders after these trading hours.',
    'is it safe to invest in Unlisted Stocks as a beginner':'if your firm about the future growth of the company only then you can think of investing in unlisted stocks.',
    'how to find undervalued stocks':'Undervalued stocks are stocks that are priced lower than their fair price.',
    'how much time should i spend while researching stocks':'You need to research the company fundamentals, analyze financial statements, competitor analysis, etc.',
    'how to invest for an IPO online':'login to your trading account and select the required ipo on the trading, portal input the number of shares you want to buy and the price of the shares.',
    'should I invest in stocks when the market is at high':'try to avoid investing in stocks whrn market is high',
    'Should I use a stop loss on my investments':' If you are an active trader, you can use stop loss to control a lot of damage.',
    'can I become a millionaire by investing in stocks':'If you wish to earn from the stock market you have to put a lot of time and effort into researching companies.',
    'nse':'national stock exchange',
    'bse': 'bombay stock exchange',
    'sebi':'security exchange board of india',
    'what are the brokerages available to invest in stock market':'zerodha,upstock,groww,paytm money etc.',
    'what is demat account':'A Demat Account is an account that holds all your securities and shares in electronic form Just like the bank holds your money.',
    'how to open demat account':'1. Fill the basic details in online Demat Account opening form with samco securities. 2. Upload the necessary documents for opening a Demat Account. 3. Digitally sign your application and submit the form.',
    'how many demat account one can have':'You can open multiple demat accounts with a valid Pan.',
    'Where to open demat account':'select a brokerage site, compile documents,fill opening form and fund your trading account',
    'is demat account safe':'Demat Accounts are safe as its regulated by SEBI (Securities and Exchange Board of India).',
    'can nri open dermat account':'Yes NRI s can open demat account to trade in indian stock market.',
    'who created the website':'team techgaints',
    'describe predict tab':'it gives the future predictions of the stock of an company',
    'describe home tab':'it tells about the general details of the stock',
    'what does the dashboard tells':'it gives the overall statistics of an specified company in an specified range',
    'what type of information does course provide':'this consists of videos that desccribes detail information about stock market from scratch to advance',
    'is long term investment better than trading':'it is better to trade whenever you are perfect in stock analysis like technical analysis of an particular stock whereas long term investment is better when the profit of an company rises in future. For more info refer to the course tab for better learning , good luck!'
}
def footer():
    st.markdown('---')
    f1,f2,f3,f4=st.columns(4)
    with f1:
        st.subheader(':red[Company]')
        st.text('About Us')
        scrn = 'https://screener-neqlzpw2cf9cxbgwhwyhx3.streamlit.app/'
        st.markdown("[Screener](%s)" % scrn)
        url = 'https://feedback-zxhfcsv93dauqdcevk7lvh.streamlit.app/'
        st.markdown("[Feedback](%s)" % url)
        compare = 'https://stockcompare-hj7d58yddnmfbjqgvnoappy.streamlit.app/'
        st.markdown("[Compare](%s)" % compare)
    with f2:
        st.subheader(':red[Get Help]')
        st.text('FAQ')
        st.text('Return')
        st.text('Stocks')
        st.text('Companies')
    with f3:
        st.subheader(':red[Online Trading]')
        st.text('Algorithmic Trading')
        st.text('Upstox Trading')
        st.text('Relaince')
        st.text('TATA')
    with f4:
        st.subheader(':red[Connect Us]')
        st.text('Email')
        st.text('FaceBook')
        st.text('LinkedIn')
        st.text('Twitter')
def get_symbol(query, preferred_exchange='ams'):
    try:
        data = yq.search(query)
    except ValueError:
        print(query)
    else:
        quotes = data['quotes']
        if len(quotes) == 0:
            return 'No Symbol Found'

        symbol = quotes[0]['symbol']
        for quote in quotes:
            if quote['exchange'] == preferred_exchange:
                symbol = quote['symbol']
                break
        return symbol
l=list(d.keys())
th_props = [
  ('font-size', '25px'),
  ('text-align', 'center'),
  ('font-weight', 'bold'),
  ('color', 'White')
  ]
                               
td_props = [
  ('font-size', '20px')
  ]
                                 
styles = [
  dict(selector="th", props=th_props),
  dict(selector="td", props=td_props)
  ]
def load_lottiefile(filepath: str):
    with open(filepath,"r") as f:
        return json.load(f)
    
def load_lottieur(url: str):
    r=requests.get(url)
    if r.status_code!=200:
        return None
    return r.json()

def stock_data(symbol,start,end):
    stock=yf.Ticker(symbol)
    data=stock.history(start=start,end=end)
    return data
def color_negative_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = 'red' if val < 0 else 'white'
    return 'color: %s' % color


st.set_page_config(page_title='StockBazar', page_icon='📈',layout='wide',initial_sidebar_state='collapsed')
selected=option_menu(
                    menu_title=None,
                    menu_icon="cast",
                    default_index=0,
                    options=['🏠Home','🏢Company','💹Stocks','🤔Predict','📊D-board','🏫Course','📈Indicators','🤖Bot'],
                    icons=[':home:' for i in range(8)],
                    orientation='horizontal',
                    styles={
                        
                        'container':{"background-color":"teal","border":'2px solid white','width':'1360px','height':'64px',},
                        'nav-link':{'font-size':'18.3px','font-weight':'bold','text-align':'center','color':'black','margin-top':'-3.5px','margin-left':'-3.4px','--hover-color':'red','height':'51px'},
                        'nav-link-selected':{"background-color":'green','color':'white'}
                    }
                )


try:
    users = fetch_users()
    emails = []
    usernames = []
    passwords = []
    
    for user in users:
        emails.append(user['key'])
        usernames.append(user['username'])
        passwords.append(user['password'])

    credentials = {'usernames': {}}
    for index in range(len(emails)):
        credentials['usernames'][usernames[index]] = {'name': emails[index], 'password': passwords[index]}
    Authenticator = stauth.Authenticate(credentials, cookie_name='Streamlit', key='abcdef', cookie_expiry_days=4)
    email, authentication_status, username = Authenticator.login(':green[Login]', 'main')
    info, info1 = st.columns(2)
    if not authentication_status:
        signup()
    if username:
        if username in usernames:
            if authentication_status:
                # let User see app
                st.sidebar.subheader(f'Welcome {username}')
                Authenticator.logout('Log Out', 'sidebar')

                st.markdown("""
                <style>
                header.css-1avcm0n.ezrtsby2{
                            visibility:hidden;
                }
                </style>
                """,unsafe_allow_html=True)
                if selected=='🏠Home':
                    st.header('Stock Market')
                    st.title(':blue[What Is the Stock Market, What Does It Do, and How Does It Work?]')
                    st.header('')
                    st.subheader('What Is the Stock Market?')
                    st.write('''
                    The term stock market refers to several exchanges in which shares of publicly held companies are bought and sold. Such financial activities are conducted through formal exchanges and via over-the-counter (OTC) marketplaces that operate under a defined set of regulations. 
                    Both “stock market” and “stock exchange” are often used interchangeably. Traders in the stock market buy or sell shares on one or more of the stock exchanges that are part of the overall stock market.
                            ''')

                    st.subheader('Understanding the Stock Market')
                    st.write('''
                    The first stock market was the London Stock Exchange which began in a coffeehouse, where traders met to exchange shares, in 1773.
                    The first stock exchange in the United States began in Philadelphia in 1790.
                    The Buttonwood Agreement, so named because it was signed under a buttonwood tree, marked the beginning of New York’s Wall Street in 1792. The agreement was signed by 24 traders and was the first American organization of its kind to trade in securities. The traders renamed their venture the New York Stock and Exchange Board in 1817

                    ''')

                    st.subheader('How the Stock Market Works?')
                    st.write('''
                    Stock markets provide a secure and regulated environment where market participants can transact in shares and other eligible financial instruments with confidence, with zero to low operational risk. Operating under the defined rules as stated by the regulator, the stock markets act as primary markets and secondary markets.
                            
                    As a primary market, the stock market allows companies to issue and sell their shares to the public for the first time through the process of an initial public offering (IPO). This activity helps companies raise necessary capital from investors.


                    ''')

                    st.subheader('Who Helps an Investor Trade on the Stock Market?')
                    st.write('''
                    Stock Brokers act as intermediaries between the stock exchanges and the investors by buying and selling stocks and portfolio managers are professionals who invest portfolios, or collections of securities, for clients. Investment bankers represent companies in various capacities, such as private companies that want to go public via an IPO
                    ''')
                    tata,reliance,microsoft,apple=st.columns(4)
                    with tata:
                        st.markdown('''
                        <html>
                                <body>
                                    <div class="grid2" style="border:4px solid white;border-radius:5px;text-align:center"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR37dLVRht1Z5IqL7L_iL_O80B30HUAcktfbA&usqp=CAU" style="height:140px;width:236px;"><br><p style="margin-top:10px;font-weight:bold;font-size:17px;padding:2px"> The Microsoft is an Indian multinational conglomerate headquartered in Mumbai. Established in 1868, it is India's largest conglomerate, with products and services in over 150 countries, and operations in 100 countries across six continents</p><a href="https://economictimes.indiatimes.com/tata-investment-corporation-ltd/stocks/companyid-13540.cms" style="font-size:23px;font-weight:bold;padding:3px;margin-top:-3px">Invest now</a></div>
                                </body>
                        </html>
                    ''',unsafe_allow_html=True)
                    with reliance:
                        st.markdown('''
                        <html>
                                <body>
                                    <div class="grid1" style="border:4px solid white;border-radius:5px;text-align:center"><img src="data:image/png;base64,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" style="width:236px;height:140px"> <br> <p style="margin-top:10px;;padding:2px;font-weight:bold;font-size:17px">The Tata Group is an Indian multinational conglomerate headquartered in Mumbai. Established in 1868, it is India's largest conglomerate, with products and services in over 150 countries, and operations in 100 countries across six continents</p><a href="https://economictimes.indiatimes.com/tata-investment-corporation-ltd/stocks/companyid-13540.cms" style="font-size:23px;font-weight:bold;padding:3px;margin-top:10px">Invest now</a></div>
                                </body>
                        </html>
                    ''',unsafe_allow_html=True)
                    with microsoft:
                        st.markdown('''
                        <html>
                                <body>
                                    <div class="grid3" style="border:4px solid white;border-radius:5px;text-align:center"><img src="https://tse3.mm.bing.net/th/id/OIP.lkqtEZbLSCp-42v8s_Q-dgHaEK?w=300&h=180&c=7&r=0&o=5&dpr=1.3&pid=1.7" style="border:0px solid white;width:230px;height:140px;margin-top:0px;"><br><p style="margin-top:10px;font-weight:bold;font-size:17px;padding:2px">The Microsoft is an Indian multinational conglomerate headquartered in Mumbai. Established in 1868, it is India's largest conglomerate, with products and services in over 150 countries, and operations in 100 countries across six continents</p><a href="https://economictimes.indiatimes.com/tata-investment-corporation-ltd/stocks/companyid-13540.cms" style="font-size:23px;font-weight:bold;padding:3px;margin-top:-3px">Invest now</a></div>
                                </body>
                        </html>
                    ''',unsafe_allow_html=True)
                    with apple:
                        st.markdown('''
                        <html>
                                <body>
                                    <div class="grid4" style="border:4px solid white;border-radius:5px;text-align:center"><img src="https://tse2.mm.bing.net/th/id/OIP.7YSDSH3nNnYYlNdI6uJ1ygHaEo?w=283&h=180&c=7&r=0&o=5&dpr=1.3&pid=1.7" style="width:230px;
                    height:140px;"><br><p style="margin-top:10px;font-weight:bold;font-size:17px;padding:2px">The Microsoft is an Indian multinational conglomerate headquartered in Mumbai. Established in 1868, it is India's largest conglomerate, with products and services in over 150 countries, and operations in 100 countries across six continents</p><a href="https://economictimes.indiatimes.com/tata-investment-corporation-ltd/stocks/companyid-13540.cms" style="font-size:23px;font-weight:bold;padding:3px;margin-top:-3px">Invest now</a></div>
                                </body>
                        </html>
                    ''',unsafe_allow_html=True)
                    st.markdown("""
                    <html>
                            <body>
                            <html>
                            <body>
                                <h4 style="text-align:center;color:red; font-size:37px;margin-top:20px;">FAQ</h4>
                                <details>
                                <summary style="font-weight:bold;font-size:25px;">Can a beginner trade in unlisted stocks?</summary>
                                <p style="color:cyan;font-size:19px;margin-left:30px;margin-top:14px;">A beginner can trade in unlisted stocks, but financial experts advise against them. As unlisted stocks are not with the market regulating authority – the Securities and Exchange Board of India (SEBI), it is not safe to invest in them</p>
                                </details>
                                <details>
                                <summary style="font-weight:bold;font-size:25px;">How to find good companies as there are thousands of publicly listed companies in the Indian stock market?</summary>
                                <p style="color:cyan;font-size:19px;margin-left:30px;margin-top:14px;">An easier approach would be to use a stock screener. By using stock screeners, you can apply a few filters (like PE ratio, debt to equity ratio, market cap, etc) specific to the industry which you are investigating and get a list of limited stocks based on the criteria applied.</p>
                                </details>
                                <details>
                                <summary style="font-weight:bold;font-size:25px;">Is investing in small caps more profitable than large caps?</summary>
                                <p style="color:cyan;font-size:19px;margin-left:30px;margin-top:14px;">Small caps companies have the caliber to grow faster compared to large caps. There can be a number of hidden gems in the small-cap industry which might not have been discovered by the market yet. However, their true potential is still untested. On the other hand, large-cap companies have already proved their worth to the market.
                    Anyways, the quality of stock is more important than the size of the company. There are a number of large-cap companies which has consistently given good returns to their shareholders. Overall, investing in small caps can be more profitable than large caps if you are investing in the right stocks</p>
                                </details>
                                <details>
                                <summary style="font-weight:bold;font-size:25px;">How many returns can I expect from the market?  </summary>
                                <p style="color:cyan;font-size:19px;margin-left:30px;margin-top:14px;">During a good market, your portfolio can give you a return as high as 30-35% (the benchmark index Nifty alone gave a return of over 50.20% in the last year till Sept 2021). However, during a bad market- the returns can be as low as 2-5% or maybe even negative.

                    If you sum up everything, you can expect an annual return of 15-18%, depending on how good you were at picking stocks. Nevertheless, you can generate an even better return if you are ready to put in some hard work.</p>
                                </details>
                                <details>
                                <summary style="font-weight:bold;font-size:25px;">What kind of stocks should be avoided for investment?  </summary>
                                <p style="color:cyan;font-size:19px;margin-left:30px;margin-top:14px;">The individual should avoid investing in stocks having low liquidity. The low liquidity makes it hard to trade in these stocks. Additionally, finding the data for analysing these companies might be hard as information on public platforms is generally not easily available. Thus, lack of research may result in loss-making investments. Additionally, one should also avoid investing in penny stocks.</p>
                                </details>
                                <details>
                                <summary style="font-weight:bold;font-size:25px;"> What is a Rolling Settlement? </summary>
                                <p style="color:cyan;font-size:19px;margin-left:30px;margin-top:14px;">Rolling settlement determines the trading price of each day and settles on a certain day during the settlement period. Currently exchanges follows T+2 rolling settlement cycle. T stands for trading day & 2 stands for another two working days</p>
                                </details>
                            </body>    
                    </html>
                            </body>    
                    </html>
                    """,unsafe_allow_html=True)
                    footer()
                if selected=='🏢Company':
                    st.markdown("""
                    <html>
                        <body>
                            <h4 style="color: red; font-size: 43px;justify-content:center;text-align:center;">What's Trending</h4>
                            <div class="grid-cont" style="margin-top:10px;grid-template-columns:repeat(3,120px);display:grid;grid-auto-rows:38px;grid-gap:12px;justify-content:center;text-align:center;font-size: 22px">
                            <div class="grid1" style="background-color:white;color:black;border-radius:7px;text-align:center;margin-top:-1px;padding-top:6px;font-weight:bold;text-decoration:none;"><a href="https://www.google.com/finance/quote/ITC:NSE">ITC</a></div>
                            <div class="grid1" style="background-color:white;color:black;border-radius:7px;text-align:center;margin-top:-1px;padding-top:6px;font-weight:bold;"><a href="https://www.google.com/finance/quote/RELIANCE:NSE">Reliance</a></div>
                            <div class="grid1" style="background-color:white;color:black;border-radius:7px;text-align:center;margin-top:-1px;padding-top:6px;font-weight:bold;"><a href="https://www.google.com/finance/quote/MSFT:NASDAQ">Microsoft</a></div>
                            <div class="grid1" style="background-color:white;color:black;border-radius:7px;text-align:center;margin-top:-1px;padding-top:6px;font-weight:bold;"><a href="https://www.google.com/finance/quote/AGRITECH:NSE">Agritech</a></div>
                            <div class="grid1" style="background-color:white;color:black;border-radius:7px;text-align:center;margin-top:-1px;padding-top:6px;font-weight:bold;"><a href="https://www.google.com/finance/quote/SBIN:NSE">SBI</a></div>
                            <div class="grid1" style="background-color:white;color:black;border-radius:7px;text-align:center;margin-top:-1px;padding-top:6px;font-weight:bold;"><a href="https://www.google.com/finance/quote/AAPL:NASDAQ">Apple</a></div>
                            </div>
                        </body>    
                    </html>
                    """,unsafe_allow_html=True)
                    sea=st.text_input('',placeholder='Enter a company')
                    st.subheader('Opening Price')
                    st.write('''
                    The opening price, also known as the "opening trade" or "opening quote," is the price at which the first transaction (trade) of a stock occurs when the stock market opens for the trading day. It's the price at which the stock starts trading after the market's opening bell. The opening price is an important reference point for traders and investors as it provides an initial indication of market sentiment for the day.
                    ''')


                    st.subheader('Closing Price')
                    st.write('''
                    The closing price is the last price at which a stock trades before the market closes for the trading day. It's determined by the final trade executed in the closing minutes of the trading session. The closing price is widely used to calculate various technical indicators, and it's often used as a benchmark to assess the overall performance of a stock for the day. 

                    In the context of the stock market, the terms "closing price" and "opening price" refer to specific prices associated with a particular stock on a given trading day         
                    ''')

                    st.subheader('Avg Highest Price')
                    st.write('''
                    It's possible that you're looking for the average of the highest prices of a stock over a certain period. In that case, you would calculate the average of the highest daily prices of a stock for a given timeframe. This could give you an idea of the average peak price the stock reached during that period.
                    ''')


                    st.subheader('Avg Lowest Price')
                    st.write('''
                    Similar to the previous point, "average lowest price" isn't a standard term. It could refer to the average of the lowest prices of a stock over a specific time frame. This would involve calculating the average of the lowest daily prices the stock reached during that period.
                    ''')
                    st.title(f'Details of the {sea}')
                    tickerCompany = get_symbol(sea)
                    sym=yf.Ticker(tickerCompany)
                    #information=pd.Series(sym1.info)
                    #details=pd.DataFrame(information)
                    #x=details.iloc[11,0]
                    #st.subheader(x)

                    actions,holders,news=st.tabs(['Actions','Holders','News'])
                    #with info:
                        #df2=details.head(10)
                        #df3=details.tail(30)
                        #df3=pd.concat([df2,df3],axis=0)
                        #st.table(df3.style.set_table_styles(styles))
                    with actions:
                        st.title(':red[Actions]')
                        st.write('In the context of the stock market, the term "actions" is not a commonly used term to refer to specific concepts. It is possible that you might be referring to "transactions" or "trades." Let us clarify these terms:')
                        st.table(sym.actions.head(15).style.set_table_styles(styles))
                        div,spl=st.columns(2)
                        with div:
                            dividend=pd.DataFrame(sym.actions.Dividends)
                            st.metric('Average Dividends',value=dividend.Dividends.mean().round(2))
                            st.line_chart(dividend)
                            st.table(dividend.head(10).style.set_table_styles(styles))
                        with spl:
                            split=pd.DataFrame(sym.actions['Stock Splits'])
                            st.metric('Average splits',value=split['Stock Splits'].mean().round(2))
                            st.line_chart(split)
                            st.table(split.head(10).style.set_table_styles(styles))
                    with holders:
                        st.title(":red[Holders]")
                        st.subheader(f'the holders of the company {sea}')
                        st.table(sym.major_holders.style.set_table_styles(styles))
                        st.subheader(f'The instituational holders of the company {sea}')
                        st.table(sym.institutional_holders.style.set_table_styles(styles))
                    with news:
                                    
                        st.title(f":blue[Trending News]")
                        i=0
                        j=0
                        for new in sym.news:    
                            st.subheader(sym.news[i]['title'])
                            image_url=sym.news[i]['thumbnail']['resolutions'][0]['url']
                            try:
                                response = requests.get(image_url)
                                img = Image.open(BytesIO(response.content))
                                st.image(img, caption=sym.news[i]['publisher'], use_column_width=True)
                            except Exception as e:
                                st.error("Error loading the image. Please check the URL and try again.")                
                            st.write(sym.news[i]['link'])
                            i=i+1
                    footer()
                if selected=='💹Stocks':
                    search=st.text_input('',placeholder='Search for a company')
                    tickerstock = get_symbol(search)
                    company = yf.download(tickerstock, period='1d',interval='1m')
                    stockmarket()
                    t1,t2,t3,t4=st.tabs(['General','Moving Average','Price Change','Intraday Range'])
                    with t1:
                        cl,op,hg,lw=st.columns(4)
                        with cl:
                            st.metric(":cyan[Avg Closing Price]",value=company['Close'].mean().round(2),delta="4%")
                        with op:
                            st.metric("Avg Opening Price",value=company['Open'].mean().round(2),delta="-2%")
                        with hg:
                            st.metric("Avg Highest Price",value=company['High'].mean().round(2),delta="-1%")
                        with lw:
                            st.metric("Avg lowest Price",value=company['Low'].mean(),delta="1%")
                        st.header(f"The stock details of the {search}")
                        st.table(company.head(20).style.set_table_styles(styles))
                        st.title(f'Stock trends of {search}')
                        st.text(f"This is the information regarding the company {search}")
                        fig = go.Figure(data=[go.Candlestick(x=company.index,
                                open=company['Open'],
                                high=company['High'],
                                low=company['Low'],
                                close=company['Close'])])
                        st.plotly_chart(fig)

                    with t2:
                        company['30mins']=company['Close'].rolling(window=30).mean()
                        company['60mins']=company['Close'].rolling(window=60).mean()
                        m30,m60,mc=st.columns(3)
                        with m30:
                            st.metric("30m Avg",value=company['30mins'].mean().round(2))
                        with m60:
                            st.metric("60m Avg",value=company['60mins'].mean().round(2))
                        with mc:
                            st.metric("Days Avg",value=company['Close'].mean().round(2))
                        min30,min60=st.tabs(['30m','60m'])
                        fig=px.line(company,y=['Close'])
                        st.plotly_chart(fig)
                        with min30:
                            st.title('Closing price with Price change 30mins')
                            fig=px.line(company,y=['Close','30mins'])
                            st.plotly_chart(fig,)
                        with min60:
                            st.title('Closing price with Price change 60mins')
                            fig=px.line(company,y=['Close','60mins'])
                            st.plotly_chart(fig)
                    with t3:
                        company['PriceChange']=company['Close']-company['Open']
                        mn,ad=st.columns(2)
                        with mn:
                            st.metric('Avg Open',value=company['Open'].mean().round(2),delta='1.1%')
                        with ad:
                            st.metric('Avg Adj Close',value=company['Adj Close'].mean().round(2),delta='-0.1%')
                        mnc,mnp=st.columns(2)
                        with mnc:
                            st.metric("Avg Close",value=company['Close'].mean().round(2),delta='2%')
                        with mnp:
                            st.metric('Avg Price Change',value=company['PriceChange'].mean().round(2),delta='-1.3%')
                        bar_colors = ['green' if val >= 0 else 'red' for val in company['PriceChange']]
                        bar_trace = go.Bar(
                        x=company.index,
                        y=company['PriceChange'],
                        marker=dict(color=bar_colors)
                        )

                        # Create the layout
                        layout = go.Layout(
                            title='Positive and Negative Bar Graph',
                            xaxis=dict(title='Categories'),
                            yaxis=dict(title='Values')
                        )

                        # Create the figure
                        fig = go.Figure(data=[bar_trace], layout=layout)
                        st.plotly_chart(fig)
                        df2 = -company[['Open','Close','Adj Close','PriceChange']]
                        style1 = company[['Open','Close','Adj Close','PriceChange']].style.applymap(color_negative_red)
                        st.table(style1)
                    with t4:
                        company['Intraday_Range'] = company['High'] - company['Low']
                        st.table(company[['Open','Close','Adj Close','Intraday_Range']].head(20).style.set_table_styles(styles))
                        company['GapUp'] = company['Open'] > company['Close'].shift(1)
                        company['GapDown'] = company['Open'] < company['Close'].shift(1)
                        st.title(f'The gapup and gapdown of the company{search}')
                        gup,gdwn=st.columns(2)
                        with gup:
                            
                            st.metric('Avg',company['Intraday_Range'].mean().round(2))
                            st.table(company[['Open', 'Close','Intraday_Range','GapUp']].head(10).style.set_table_styles(styles))
                        with gdwn:
                            st.metric('Avg',company['Intraday_Range'].mean().round(2))
                            st.table(company[['Open', 'Close','Intraday_Range','GapDown']].head(10).style.set_table_styles(styles))
                    footer()
                if selected=='🤔Predict':
                    stockPredict()
                    footer()
                if selected=='📊D-board':
                    lottie_coding=load_lottiefile("animation_ll4zaxpf.json")
                    lottie_anni=load_lottiefile("animation_ll4z00j3.json")
                    lottie_url='https://lottie.host/0c99d6f8-ab79-43d6-b188-f14713ed7b05/r0xUnkKdpF.json'
                    anni='https://lottie.host/77578782-ff7a-4a54-bd0e-00d0e6629fb4/Z8UI5WKH6x.json'
                    lottie2=load_lottieur(anni)
                    lottie1=load_lottieur(lottie_url)
                    st.markdown("<marquee><h2 style='color:white;'>Stock Market</h2></marquee>",unsafe_allow_html=True)
                    a,b,c=st.columns(3)
                    with a:
                        st.markdown("<br> </br>",unsafe_allow_html=True)
                        st.markdown("<br> </br>",unsafe_allow_html=True)
                        st_lottie(
                            lottie1,
                            loop=True
                        )
                    with c:
                        st_lottie(
                                lottie2,
                                loop=True,
                                height=None,
                                width=None
                            )
                    with b:
                        st.markdown("<br> </br>",unsafe_allow_html=True)
                        st.markdown("<br> </br>",unsafe_allow_html=True)
                        symbol=st.text_input('',placeholder='Search for a Company')
                        start=st.date_input("Start")
                        end=st.date_input("End")
                        tickerdash = get_symbol(symbol)
                    
                    data=stock_data(tickerdash,start,end)
                    stoc= data.iloc[::-1]
                    stock=stoc[['Open','Close','High','Low','Volume']]
                    st.markdown("<h1 style='color:green;'><center>Stock values</center></h1>",unsafe_allow_html=True)
    
                    c1,c2=st.columns(2)
                    with c1:
                        st.subheader("Opening and Closeing prices")
                        st.table(stock[['Close','Open','Volume']].head())
                        select=option_menu(
                            menu_title='Close Open',
                            orientation='horizontal',
                            options=['1w','2w','1m','3m','6m']
                        )
                        if select=='1w':
                            st.line_chart(stock.iloc[0:8,0:2],y=['Close','Open'])
                        if select=='2w':
                            st.line_chart(stock.iloc[0:15,0:2],y=['Close','Open'])
                        if select=='1m':
                            st.line_chart(stock.iloc[0:31,0:2],y=['Close','Open'])        
                        if select=='3m':
                            st.line_chart(stock.iloc[0:91,0:2],y=['Close','Open'])
                        if select=='6m':
                            st.line_chart(stock.iloc[0:183,0:2],y=['Close','Open'])
                    with c2:
                        st.subheader("Highest and Lowest prices")
                        st.table(stock[['High','Low','Volume']].head())
                        select=option_menu(
                            menu_title='High Low',
                            orientation='horizontal',
                            options=['1w','2w','1m','3m','6m']
                        )
                        if select=='1w':
                            st.line_chart(stock.iloc[0:8,2:5],y=['High','Low'])
                        if select=='2w':
                            st.line_chart(stock.iloc[0:15,2:5],y=['High','Low'])        
                        if select=='1m':
                            st.line_chart(stock.iloc[0:31,2:5],y=['High','Low'])
                        if select=='3m':
                            st.line_chart(stock.iloc[0:91,2:5],y=['High','Low'])
                        if select=='6m':
                            st.line_chart(stock.iloc[0:183,2:5],y=['High','Low'])
                    plt.figure(figsize=(12,6))
                    fig=px.line(stock)
                    st.plotly_chart(fig)
                    footer()
                if selected=='📈Indicators':
                    st.subheader(':blue[Technical Indicators Analysis]')
                    st.markdown('<p>A technical indicator is a mathematical calculation or pattern derived from price, volume, or open interest of a security (such as stocks, currencies, commodities, etc.) in financial markets. These indicators are used by traders and analysts to gain insights into the markets trend, momentum, volatility, and potential reversal points. Technical indicators are applied to charts to help traders make more informed decisions about when to buy, sell, or hold a particular security.</p>',unsafe_allow_html=True)
                    indicate()
                    stockindic=st.text_input('')
                    tickerIndicator = get_symbol(stockindic)
                    per=st.selectbox('Period',options=['1d','2d','1w','1mo','3mo','6mo','1y'])
                    inter=st.selectbox('Interval',options=['1d','5d','1wk'])
                    tech=yf.download(tickerIndicator,period=per,interval=inter)
                    
                    df=pd.DataFrame()
                    ind_list=df.ta.indicators(as_list=True)
                    technical_indicator=st.selectbox('Tech Indicators',options=ind_list)
                    method=technical_indicator
                    indicator=pd.DataFrame(getattr(ta,method)(low=tech['Low'],close=tech['Close'],high=tech['High'],open=tech['Open'],volume=tech['Volume']))
                    
                    indcl,indop=st.columns(2)
                    with indcl:
                        st.metric(':blue[Closing Price]',value=tech.Close.mean().round(2),delta='1%')
                    with indop:
                        st.metric(':blue[Opening Price]',value=tech.Open.mean().round(2),delta='-3%')
                    indh,indl=st.columns(2)
                    with indh:
                        st.metric(':blue[High Price]',value=tech.High.mean().round(2),delta='-1%')
                    with indl:
                        st.metric(':blue[Lowest Price]',value=tech.Low.mean().round(2),delta='-1.4%')
                    indicator['Close']=tech['Close']
                    fig_ind_new=px.line(indicator)
                    st.plotly_chart(fig_ind_new)
                    st.table(indicator.tail(10).style.set_table_styles(styles))
                    footer()
                if selected=='🏫Course':
                    c()
                    footer()
                if selected=='🤖Bot':
                  
                    if "messages" not in st.session_state:
                        st.session_state.messages = []
                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])
                    if prompt := st.chat_input("What is up?"):
                        st.session_state.messages.append({"role": "user", "content": prompt})
                        with st.chat_message("user"):
                            st.markdown(prompt)
                        with st.chat_message("assistant"):
                            message_placeholder = st.empty()
                            full_response = ""
                            assistant_response = ''
                            prompt=prompt.lower()
                            if prompt in l:
                                assistant_response=d[prompt]
                            else:
                                assistant_response="Sorry,I didn't get you."
                            for chunk in assistant_response.split():
                                full_response += chunk + " "
                                time.sleep(0.05)
                                message_placeholder.markdown(full_response + "▌")
                            message_placeholder.markdown(full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                        
            elif not authentication_status:
                with info:
                    st.error('Incorrect Password or username')
            else:
                with info:
                    st.warning('Please feed in your credentials')
        else:
            with info:
                st.warning('Username does not exist, Please Sign up')
except:
    st.write('Either refresh or fill the details')

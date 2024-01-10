import cohere
from cohere.responses.classify import Example

class CohereClassifier():
    def __init__(self, api_keys):
        self.api_keys = api_keys

    def get_ratings(self, input:str):
        co = cohere.Client(self.api_keys['Cohere'])

        examples=[
        Example("""A major pharmaceutical company has just announced that its highly anticipated drug for a widespread illness has 
                failed in its final stage of clinical trials. As a result, the company's stock price has plummeted by 30% in pre-market 
                trading, and analysts have downgraded the stock, expecting it to underperform in the market for the foreseeable future.
                """, "1"),
        Example("""A leading technology firm has revealed a massive data breach that has affected millions of its customers' 
                personal information. In response, several major clients have terminated their contracts with the company, and 
                its stock price has fallen sharply by 25%. Analysts have issued a sell rating on the stock, predicting further 
                decline in the company's market value due to reputational damage and potential legal liabilities.
                """, "1"),
        Example("""Investing in socks is an interesting option for those looking to diversify their portfolio. While the sock market may not be as exciting as some other industries, socks are a staple item in most wardrobes, and people will always need them. However, it's important to remember that sock trends can change rapidly, and what's popular today may not be in demand tomorrow.
                """, "2"),
        Example("""Major Apple Inc. (NASDAQ:AAPL) supplier Japan Display Inc. (OTC:JPDYY) on Thursday said that the company laid off an accountant last year for embezzlement, Reuters reported.
                What Happened
                The LCD maker has filed a criminal complaint against the executive for embezzling $5.3 million over four years, according to Reuters.
                The employee arranged payments from Japan Display to a fictitious company created by him, JDI said in a statement reviewed by Reuters.
                Japan Display was formed in 2011 as a joint venture between the LCD divisions Sony Corporation (NYSE:SNE), Toshiba Corporation (OTC:TOSBF), and Hitachi Ltd. (OTC:HTHIY).
                Why It Matters
                The company is one of the largest suppliers of displays for Apple’s iPhones. It was previously the exclusive supplier for Nintendo Switch, as reported by the Wall Street Journal.
                Japan Display has been struggling to turn profits ever since its initial public offering. The company reported a loss at $995 million in the first half of 2019, as reported by the Nikkei last week — its sixth consecutive half-year loss.
                Price Action
                Japan Display’s stock closed 1.37% lower at the Tokyo Exchange on Thursday.""", "2"),
        Example("""Apple Inc. (NASDAQ:AAPL) slashed its previously issued first-quarter sales guidance Wednesday from a range of $89 billion to $93 billion to $84 billion — $7.5 billion less than the Street's expectations of $91.5 billion in sales. The news moved tech stocks lower in after hours trading.
                What To Know
                Revenue from the iPhone was lower than expected, mostly in China, and this accounts for the shortfall, CEO Tim Cook said in an 8-K filing. Smartphone upgrades were not as strong as anticipated, he said. 
                ""While macroeconomic challenges in some markets were a key contributor to this trend, we believe there are other factors broadly impacting our iPhone performance, including consumers adapting to a world with fewer carrier subsidies, U.S. dollar strength-related price increases and some customers taking advantage of significantly reduced pricing for iPhone battery replacements."" 
                Cupertino also cut its Q1 gross margin forecast from a range of 38-38.5 percent to 38 percent. The company is set to deliver its Q1 report after the close on Tuesday, Jan. 29. 
                Why It's Importnat
                The timing of iPhone launches, a strong U.S. dollar, an ""unprecedented"" number of new product ramps and economic weakness in emerging markets were named by Apple as the largest factors affecting its Q1 performance.
                The launch of the iPhone XS and XS Max in Q4 vs. the launch of the iPhone X in Q1 of 2018 creates a ""difficult"" compare in 2019, Cook said, adding that the setup ""played out broadly in line with our expectations."" 
                Cook said economic concerns in emerging markets ""turned out to have a significantly greater impact than we had projected.""
                Apple shares were down 7.4 percent at $146.23 in Wednesday's after-hours session. Tech stocks such as Amazon.com, Inc. (NASDAQ:AMZN) Netflix, Inc. (NASDAQ:NFLX) and Facebook (NASDAQ:FB) were down about 2 percent.
                Related Links:
                Wedbush: Trade War Could Mean Supply Chain Disruption, Higher Costs For Tech Sector In 2019
                Loup Ventures: Apple Services Model Is Sound Despite Netflix Exit
                Photo courtesy of Apple.""", "2"),
        Example(""" "As the calendar flipped to 2019, investors hoping for a change in the uncertainty that ruled the markets last year may need to wait a bit. The new year got off to a rough start, as weak overseas data helped spark a global selloff. 
                Specifically, an index for China’s manufacturing sector, the Caixin/Markit Manufacturing PMI, fell below 50 for the first time in 19 months, to 49.7. A reading below 50 indicates economic contraction. For many analysts closely watching how the ongoing tariff tensions with the U.S. will shake out, the contraction indicates that the trade relations may be taking a toll on demand in the second-largest economy in the world.
                Leading the markets lower was Hong Kong’s Hang Seng Index, which fell 2.8%. European markets fell as well, after manufacturing data across the eurozone indicated a broad-based slowdown over the region.
                In other news that could weigh on stocks today, Tesla (TSLA) reported Q4 production that fell short of Wall Street estimates. The electric automaker’s shares slid over 7% in early trade as it said it delivered 90,700 vehicles, up 8% from a year ago but short of the 92,000 expected, according to FactSet.
                Risk Mode Toggles to ""Off""
                One chief effect from the overseas turmoil is it seems to have shifted the risk-on/risk-off switch back to the “off” position. The Cboe Volatility Index (VIX), which last week fell to the mid 20s, down from a high of 36, is back on the rise. The 10-year Treasury yield fell in early trading in an apparent flight to safety. Gold, also a choice among investors seeking safety in troubled times, rose to its highest level in six months. 
                Although this week overall is relatively light on major economic data releases, a U.S. manufacturing index is scheduled for release tomorrow. Given the weak state of the recent data from overseas, it could be interesting to see how U.S. manufacturing fared in December.
                And the government on Friday is scheduled to release one of the most closely followed reports–the monthly nonfarm payrolls report. Economists in a Briefing.com consensus expect the economy to have added 180,000 jobs in December. Average hourly earnings are expected to rise 0.3%. 
                Meanwhile, thousands of U.S. government workers remain out of their jobs, at least temporarily, as the partial government shutdown continues. Hopes of a resolution seem unlikely until at least until Jan. 3, when a new Congress convenes.
                It may be worth keeping in mind that there have been government shutdowns and short-term funding gaps in the past and, overall, there has been a minimal impact on the stock market. The last partial government shutdown that lasted more than a weekend was in October 2013. By the end of it, the S&P 500 (SPX) was actually up. The consumer discretionary sector and defense stocks dropped a little further than the SPX halfway through it, but they were also positive by the time it ended.
                Trade Winds Flutter Again
                Sentiment in the U.S. stock market this morning contrasts to the celebratory mood on New Year’s Eve, when all of the three main indices posted gains on optimism about progress in resolving the trade tensions between the U.S. and China.
                The enthusiasm came after President Trump talked up a call between him and his Chinese counterpart, saying that “big progress” was being made. The tweet seemed to be enough to help the market end the year’s last session in the green.
                It’s probably worth keeping in mind that trading volumes for stocks in each of the main three U.S. indices were light, as was to be expected during a holiday-shortened week and a trading day right before the New Year’s closure of the stock market. In such conditions, it’s not rare for thin volume to help exaggerate moves to the upside, or downside. 
                While Trump’s tweet appeared to provide enough momentum to help push all 11 of the S&P 500 sectors into the green, in a regular trading session that might not have been a the case. After all, the Wall Street Journal reported that Trump may be overstating his case. And, according to Reuters, Chinese state media said Chinese President Xi Jinping was more reserved.
                Reaction to the headlines about the U.S.-China trade situation could be a microcosm of much of last year, when news that the market interpreted as positive led to jumps in equities, but headlines judged unfavorable resulted in stock declines.
                And, with the trade war still unresolved, it’s arguable that this sort of trade-based market volatility will continue this year depending on how the trade winds blow.
                Market Revaluation
                Another reason for the U.S. stock market posting its worst performance since 2008 last year was a broad revaluation as investors seemed to get nervous about potentially lofty stock prices that had carried stocks to record highs amid bumper corporate earnings earlier in the year. A big casualty of this repricing was the FAANG group of stocks. Facebook, Inc. (NASDAQ:FB), Apple, Inc. (NASDAQ:AAPL), Amazon.com, Inc. (NASDAQ:AMZN), Netflix, Inc. (NASDAQ:NFLX), and Alphabet (NASDAQ:GOOG, NASDAQ:GOOGL), which had helped drive positive momentum for so long, all hit 20% losses in late 2018. 
                The magnitude of the overall market selloff raises questions about whether the market may be nearing a bottom and whether it could find it early on in 2019. Even as the major headwinds of 2018 look like they’re remaining in place, much of that worry may already be priced in to the forward-looking stock market. 
                Equities weren’t the only investments to sell off sharply heading into the end of the year. As investors apparently became more risk-averse, they also sold out of oil positions, causing crude prices to fall sharply. Worries about global economic growth, and the related demand for oil, were at the heart of this selloff. Last year also saw the market awash with a glut of oil supply, which also helped to pressure prices.
                But not all commodities suffered. The decline in risk appetite that was reflected in lower stock and oil prices ended up helping gold. The precious metal is often considered a safe haven in times of market turmoil, although no investment is completely risk-free. Gold has risen toward the end of the year even as the dollar has stabilized at elevated levels. (See below.) That’s notable because a stronger dollar often makes dollar-denominated gold less affordable for those using other currencies, potentially dampening demand.
                Gold Getting Its Shine On: For much of 2018, the dollar index (candlestick chart) and gold (purple line) have been a mirror image of each other. But toward the end of the year, gold has gained ground despite the dollar index being relatively flat. That might support gold’s allure as a perceived safe-haven investment amid the ferocious selling in assets that are considered riskier. Data Source: Intercontinental Exchange, CME Group. Chart source: Thethinkorswim® platform from TD Ameritrade. For illustrative purposes only. Past performance does not guarantee future results.
                Fed’s Balancing Act in 2019: Although the economy in the U.S. posted some solid numbers in 2018, there have been concerns about a slowdown in growth in the world’s largest economy. The Fed lowered its median estimate for 2019 gross domestic product growth to 2.3% from 2.5%. A recent report showed weaker-than-forecast consumer confidence. And the U.S. housing market has been struggling with demand amid rising buying costs, including higher mortgage rates as the Fed has been on a hawkish streak. According to Fed commentary, the Fed funds target range is nearing a “neutral” setting that neither sparks economic growth nor slows growth down. And it has lowered its expectation to two rate hikes in 2019 as opposed to three it had previously forecasted. But it remains to be seen whether the central bank ends up being as dovish as the market hopes this year. 
                Half of Globe Online in 2018: Last year, humankind reached a milestone that perhaps got less fanfare than moves in the stock market. By the end of 2018, more than half of the world’s population was using the internet, according to an estimate from a December report from the International Telecommunication Union, the United Nations’ specialized agency for information and communication technologies. In addition to being an important marker of the information age, the news also shows there is still potential for expansion from technology companies. Developed countries, where four out of every five people are online, are reaching internet usage saturation. But only 45% of people in developing countries are using the internet. Indeed, in the world’s 47 least-developed nations, it’s a mirror image of the developed world. There, four out of five people are not yet using the internet. 
                A Look at Volatility: Much attention has been paid to the market’s volatility toward the end of last year. But as history shows, 2018’s large ups and downs were below the average over most of the last two decades. As measured by days when the SPX gained or declined 1% or more, volatility was uncharacteristically light during 2017, which had only eight of those days, according to investment research firm CFRA. Volatility came roaring back last year, which had eight times that number. While that is higher than the average of 51 days a year since 1950, it’s below the average of 69 since 2000.
                " """, "3"),
        Example("""JPMorgan Chase (NYSE:JPM) CEO Jamie Dimon's tirade against cryptocurrencies continues.
                Despite the investment bank's ongoing efforts to boost its crypto capabilities, Dimon blasts Bitcoin (CRYPTO: BTC) for being a decentralized Ponzi scheme.
                See Also: JPMorgan CEO Still 'Doesn't Care About Bitcoin,' Even If His Clients Do
                ""I am a major skeptic on crypto tokens, which you call currency like Bitcoin,"" he said. ""They are decentralized Ponzi schemes and the notion that it's good for anybody is unbelievable. So we sit here in this room and talk about a lot of things, but $2 billion have been lost. Every year, $30 billion in ransomware. AML, sex trafficking, stealing, it is dangerous.""
                Dimon, 66, made the comments during a congressional testimony on Wednesday. The JPMorgan honcho is a longtime vocal critic of digital currencies, especially Bitcoin, and has often referred to the asset as ""worthless"" and fraudulent, advising investors to avoid it.
                JPMorgan's Crypto Ambitions
                Despite Dimon’s stance, JPMorgan remains committed to blockchain technology and providing crypto services, despite the bear markets dampening investors' enthusiasm in the sector.
                See Also: JPMorgan Chase's Blockchain Unit Plans To Bring Trillions Of Dollars In Tokenized Assets To DeFi
                Dimon further said with the right legislation, stablecoins — digital assets linked to the value of the U.S. dollar or other currencies — would not pose a problem.
                The New York-based firm also gave wealth management clients access to six Bitcoin funds last year. One is a part of the Osprey Funds, while four are from Grayscale Investments.
                The sixth is a Bitcoin fund created by New York Digital Investment Group, a technology and financial services company (NYDIG).
                JPMorgan had provided a positive outlook on the Metaverse at the beginning of 2022, forecasting that the industry may grow to be a trillion-dollar business in the years to come.
                The banking behemoth earlier this month posted a job listing to hire a new vice president that specializes in niche technology, including Web3, cryptocurrency, fintech, and the metaverse.
                See Also: Most Of Crypto 'Still Junk,' JPMorgan's Blockchain Head Says
                Earlier this year, JPMorgan opened a blockchain-based virtual lounge in Decentraland (CRYPTO: MANA) named after its blockchain unit Onyx.
                The firm also published an 18-page report describing the metaverse as a $1 trillion market opportunity. In July 2021, JPMorgan forecast that Ethereum’s (CRYPTO: ETH) shift to Proof-of-Stake could kickstart a $40 billion staking industry by 2021.
                See Also: JPMorgan Seeks To Hire Metaverse, Crypto Expert
                Are you ready for the next crypto bull run? Be prepared before it happens! Hear from industry thought leaders like Kevin O’Leary and Anthony Scaramucci at the 2022 Benzinga Crypto Conference on Dec. 7 in New York City.
                """, "3"),
        Example("""Gainers
                OHR Pharmaceutical Inc (NASDAQ:OHRP) shares climbed 68.2 percent to $0.1850 after announcing a merger with NeuBase. Ohr shareholders will own 20 percent of the combined company.
                Kitov Pharma Ltd  (NASDAQ:KTOV) gained 57.3 percent to $1.18 after announcing a marketing and distribution agreement with Coeptis Pharma. Kitov will receive $3.5 million of milestone payments.
                BioXcel Therapeutics, Inc. (NASDAQ:BTAI) shares gained 32.9 percent to $4.77 after the company disclosed that it has met primary endpoint in Phase 1 study of IV dexmedetomidine for acute agitation in Senile Dementia of the Alzheimer's Type (SDAT) patients.
                Celgene Corporation  (NASDAQ:CELG) shares surged 26.9 percent to $84.53 after Bristol-Myers Squibb announced plans to buy the company at $50 per share in cash in a $74 billion deal.
                Estre Ambiental Inc  (NASDAQ:ESTR) jumped 19.8 percent to $2.60.
                Alliqua BioMedical, Inc. (NASDAQ:ALQA) jumped 19.1 percent to $2.37. Alliqua Biomedical said after satisfying merger expenses, the company plans to pay $1-$1.20 per share special dividend.
                IZEA Worldwide Inc (NASDAQ:IZEA) rose 16.6 percent to $1.2125 after the company announced it won multiple significant contracts in December.
                Park Electrochemical Corp.  (NYSE:PKE) climbed 11 percent to $20.45. Park Electrochemical posted Q3 earnings of $0.10 per share on sales of $12.853 million.
                Avadel Pharmaceuticals plc  (NASDAQ:AVDL) gained 9.7 percent to $3.05. Avadel Pharma reported the resignation of its CEO Michael Anderson.
                FTE Networks, Inc.  (NYSE:FTNW) rose 9.3 percent to $3.0500. FTE Networks completed 2018 with approximately $572.4 million in new infrastructure contract awards.
                DBV Technologies S.A. (NASDAQ:DBVT) gained 8.5 percent to $7.58 after the company expanded and strengthened its leadership team.
                Corbus Pharmaceuticals Holdings, Inc. (NASDAQ:CRBP) gained 8 percent to $7.07 after the company announced a strategic collaboration with Kaken Pharmaceuticals to develop and commercialize Lenabasum in Japan. Corbus is set to receive $27 million upfront.
                Veracyte, Inc. (NASDAQ:VCYT) rose 7.6 percent to $12.91 after the company announced a strategic collaboration with J&J Innovation for lung cancer detection.
                Verrica Pharmaceuticals Inc.  (NASDAQ:VRCA) climbed 6.9 percent to $8.50 after reporting top-line results from 2 pivotal Phase 3 trials of VP-102 in patients with molluscum contagiosum.
                Celyad SA  (NASDAQ:CYAD) rose 5.8 percent to $20.78 after climbing 8.20 percent on Wednesday.
                Incyte Corporation  (NASDAQ:INCY) gained 5.7 percent to $67.17 after Guggenheim upgraded the stock from Neutral to Buy.
                Tel-Instrument Electronics Corp. (NYSE:TIK) rose 5.1 percent to $4.4500 after surging 11.18 percent on Wednesday.
                Cocrystal Pharma Inc (NASDAQ:COCP) climbed 4 percent to $3.95 after announcing a collaboration with Merck to develop influenza agents.
                Check out these big penny stock gainers and losers
                Losers
                Aevi Genomic Medicine Inc (NASDAQ:GNMX) dipped 75.3 percent to $0.1900 after the company announced its top-line results from its placebo-controlled ASCEND trial of AEVI-001 in children with ADHD did not achieve statistical significance on primary endpoint.
                Datasea Inc.  (NASDAQ:DTSS) shares fell 18.1 percent to $3.36.
                Spring Bank Pharmaceuticals, Inc.  (NASDAQ:SBPH) dropped 17.3 percent to $9.07.
                Bristol-Myers Squibb Company  (NYSE:BMY) declined 12.3 percent to $45.60 after the company Bristol-Myers Squibb announced plans to buy Celgene at $50 per share in cash in a $74 billion deal.
                Mellanox Technologies, Ltd.  (NASDAQ:MLNX) shares fell 11.8 percent to $81.03 after the company announced that it will report Q4 financial results January 30, 2019 and named Doug Ahrens as CFO.
                BeiGene, Ltd.  (NASDAQ:BGNE) declined 11.7 percent to $120.17 in an apparent reaction from Bristol-Meyers Squibb acquiring Celgene.
                STMicroelectronics N.V. (NYSE:STM) shares dropped 11.5 percent to $12.16.
                Tyme Technologies, Inc.  (NASDAQ:TYME) shares declined 11.2 percent to $3.00.
                Atara Biotherapeutics, Inc.  (NASDAQ:ATRA) dropped 10.2 percent to $31.66 after the company announced that CEO Isaac Ciechanover plans to step down.
                Apple Inc. (NASDAQ:AAPL) shares fell 9.5 percent to $142.86 after the company slashed its previously issued first-quarter sales guidance Wednesday from a range of $89 billion to $93 billion to $84 billion — $7.5 billion less than the Street's expectations of $91.5 billion in sales.
                Skyworks Solutions, Inc.  (NASDAQ:SWKS) fell 9.3 percent to $61.63.
                American Airlines Group Inc.  (NASDAQ:AAL) slipped 9.2 percent to $29.50. Airline stocks traded lower sector-wide after Delta Airlines forecasted a lower revenue growth outlook.
                Delta Air Lines, Inc.  (NYSE:DAL) shares declined 9.1 percent to $45.54 after the company forecasted a lower revenue growth outlook.
                Universal Display Corporation (NASDAQ:OLED) dropped 9 percent to $83.00.
                Lumentum Holdings Inc.  (NASDAQ:LITE) fell 8.3 percent to $39.04.
                Spirit Airlines, Inc.  (NYSE:SAVE) dropped 8.2 percent to $53.12.
                Cirrus Logic, Inc. (NASDAQ:CRUS) fell 7.3 percent to $31.76.
                Marker Therapeutics, Inc.  (NASDAQ:MRKR) dropped 7.3 percent to $5.76.
                Lumber Liquidators Holdings, Inc. (NYSE:LL) fell 7.1 percent to $9.27 after rising 4.73 percent on Wednesday.
                Mettler-Toledo International Inc.  (NYSE:MTD) declined 6.2 percent to $512.58 after Bank of America downgraded the stock from Buy to Neutral.
                Logitech International S.A.  (NASDAQ:LOGI) fell 6.2 percent to $29.30.
                First Data Corporation (NYSE:FDC) dipped 6.1 percent to $16.01 after Stephens & Co. downgraded the stock from Overweight to Equal-Weight and lowered the price target from $25 to $20.
                Sohu.com Limited  (NASDAQ:SOHU) dropped 5.8 percent to $16.82 after the company was ordered by China's Cyberspace Administration to suspend updates to their online news services for a week.
                Albemarle Corporation (NYSE:ALB) fell 5.2 percent to $74.03. Berenberg downgraded Albemarle from Buy to Hold.
                Baidu, Inc.  (NASDAQ:BIDU) dropped 5 percent to $154.20 after the company was ordered by China's Cyberspace Administration to suspend updates to their online news services for a week.
                CBRE Group, Inc.  (NYSE:CBRE) fell 4.8 percent to $38.01 after Bank of America downgraded the stock from Buy to Neutral.
                """, "4"),
         Example("""Today, the technology sector showed mixed performance with several stocks facing minor pullbacks, while 
                others managed to hold their ground. Despite concerns about inflation and potential interest rate hikes, the 
                long-term growth prospects for the tech industry remain strong, driven by increased demand for digital services, 
                cloud computing, and artificial intelligence.
                Investors should be cautious and monitor the market closely in the short term, as some stocks might experience 
                further declines. However, for long-term investors with a well-diversified portfolio, the tech sector still offers 
                significant growth potential.
                """, "4"),
        Example("""Amazon.Com Inc’s (NASDAQ: AMZN) alleged unfair practices has attracted a protest event from over half a million small Indian business merchants on Thursday, Bloomberg reports.
                The local traders have long blamed giant e-tailers like Amazon and Walmart Inc (NYSE: WMT)-owned Flipkart for affecting the livelihoods of small online and offline sellers via preferential treatment.
                The trader groups’ protest event named “Asmbhav,” meaning “impossible” in Hindi, coincided with Amazon’s virtual annual seller jamboree called Smbhav, or “possible” that debuted last year.
                India’s small traders, distributors, and merchants sought respite from the foreign influential retail companies by lodging court and antitrust regulator petitions ahead of a potential amendment of foreign investment rules.
                The protest organizers will hand out “Asmbhav awards” to CEO Jeff Bezos, country chief Amit Agarwal and its India business partner, and Infosys founder, Narayana Murthy, symbolizing their dig at Amazon’s Smbhav awards to select sellers. The event is backed by trade groups like the All India Online Vendors Association and the All-India Mobile Retailers Association.
                Amazon’s four-day event panel speakers include ex-PepsiCo Inc (NASDAQ: PEP) CEO Indra Nooyi, telecom operator Bharti Airtel Ltd’s Chair Sunil Mittal, India’s chief economic adviser Krishnamurthy Subramanian and Infosys Ltd (NYSE: INFY) co-founder and Chair Nandan Nilekani. The participants will include small businesses, startups, developers, and retailers.
                Price action: AMZN shares traded lower by 1.15% at $3,360.99 on the last check Wednesday.
                """, "5"),
        Example("""McDonald's Holdings Co. Japan, a subsidiary of McDonald's Corp (NYSE:MCD), is facing a french-fries shortage in Japan, primarily due to the coronavirus crisis and the flooding of a port in Vancouver. 
                What Happened: McDonald's would ration its fries out in Japan, with the conglomerate slashing its medium and large size offering until the new year, according to a report from Bloomberg.
                The company commented on its drastic plan of action, stating it wants ""to ensure that as many customers as possible will have continued access to our french fries.""
                McDonald's is also trying to work out alternative flight routes is and working closely with both its suppliers and importers to mitigate the effects of the shortage on its 2,900 outlets in Japan, as per Bloomberg. The company also stated that the current issues won't hamper the supply of its hash browns. 
                Earlier this week, McDonald's settled a lawsuit for $33.5 million with former Oakland Athletics baseball player Herbert Washington, who claimed the company restricted his franchise locations to predominantly low volume Black neighborhoods and then forced him to downgrade the size of his locations unfairly.
                Price Action: McDonald's shares 0.4% higher at $262.85 in the pre-market session on Monday.""", "5"),
        Example("""6 Technology Stocks Moving In Wednesday's Pre-Market Session""", "5"),
        Example("""Stocks That Hit 52-Week Highs On Wednesday""", "5"),
        Example("""Barclays cut the price target on  Cisco Systems, Inc.  (NASDAQ:CSCO) from $56 to $46. Barclays analyst Tim Long downgraded the stock from Overweight to Equal-Weight. Cisco shares rose 0.1% to $41.61 in pre-market trading.
                JP Morgan raised the price target for  Vulcan Materials Company  (NYSE:VMC) from $170 to $185. JP Morgan analyst Adrian Huerta maintained the stock with a Neutral. Vulcan Materials shares fell 0.1% to $160.30 in pre-market trading.
                Truist Securities cut the price target on  Edwards Lifesciences Corporation  (NYSE:EW) from $117 to $112. Edwards Lifesciences shares fell 0.4% to $84.85 in pre-market trading.
                Baird reduced the price target for  Cognizant Technology Solutions Corporation  (NASDAQ:CTSH) from $78 to $76. Cognizant Technology shares fell 0.3% to $59.90 in pre-market trading.
                HC Wainwright & Co. lowered the price target on  VIQ Solutions Inc.  (NASDAQ:VQS) from $4 to $2. VIQ Solutions shares fell 7.9% to close at $0.65 on Wednesday.
                Piper Sandler boosted the price target for  TPI Composites, Inc.  (NASDAQ:TPIC) from $13 to $17. TPI Composites shares fell 0.1% to $14.22 in pre-market trading.
                Mizuho cut the price target on  Block, Inc.  (NYSE:SQ) from $125 to $57. Block shares fell 1.8% to $58.40 in pre-market trading.
                Check out this: H.B. Fuller, Lennar And Some Other Big Stocks Moving Higher In Today's Pre-Market Session 
                Don’t forget to check out our  premarket coverage here.""", "5"),
        Example("""The healthcare sector has shown resilience during the pandemic, with pharmaceutical companies and medical device 
                manufacturers experiencing increased demand for their products and services. The development and distribution of 
                COVID-19 vaccines have further boosted the performance of some healthcare stocks.
                However, investors should be aware of the potential risks associated with regulatory changes, pricing pressures, 
                and competition in the industry. While the long-term outlook for the healthcare sector remains generally positive, 
                it is crucial for investors to carefully analyze individual companies and their growth prospects before making 
                investment decisions.
                """, "6"),
        Example("""The technology sector has been a key driver of market growth in recent years, with companies involved in 
                artificial intelligence, cloud computing, and cybersecurity showing strong performance. The rapid adoption of 
                digital technologies across various industries has led to increased demand for innovative solutions, creating 
                opportunities for technology companies.
                Nonetheless, the sector is not without risks, as regulatory scrutiny, geopolitical tensions, and fierce 
                competition could impact the performance of some companies. While the overall outlook for the technology sector 
                is still favorable, investors should exercise caution and conduct thorough research on specific companies to 
                assess their growth potential and ability to navigate challenges.
                """, "6"),
        Example("""The renewable energy sector has been experiencing steady growth in recent months, driven by increasing global 
                awareness of the need for clean energy solutions and favorable government policies. Companies focused on solar, 
                wind, and other alternative energy sources are well-positioned to capitalize on this trend, as governments and 
                corporations around the world continue to invest in sustainable projects.
                While there might be short-term fluctuations in the stock prices of renewable energy companies, 
                the overall outlook remains positive. Investors with a long-term perspective may find attractive opportunities 
                in this sector, as demand for clean energy is expected to rise significantly in the coming years.
                """, "7"),
        Example("""Nio Inc – ADR (NYSE:NIO) has been one of the most volatile stocks in 2019 despite the fairly robust performance of the broader market.
                Since the start of the year, the NYSE-listed ADRs of the Chinese electric vehicle manufacturer have shed over 70%, reflecting a weak macroeconomic climate and geopolitical tensions.
                These concerns could be things of the past, as the EV market is set for a strong recovery, said Li Bin, the founder, chairman and CEO of Nio, as reported by the National Business Daily. 
                Spring Is Near
                Nio, invariably referred to as China's Tesla Inc (NASDAQ:TSLA), posted sales in September and October that give Li confidence, especially after  weak performance in July and August.
                ""Spring for electric vehicles is near"" as more manufacturers are ""educating the market"" and delivering vehicles in China, the CEO reportedly said. 
                Slow Start To The Year
                Nio sells two EV models: the ES6 launched in June  and an older ES8 model.
                The company reported deliveries of 1,805 in January and 811 in February, blaming the slowdown on accelerated deliveries at the end of 2018 made in anticipation of an EV subsidy reduction and the slowdown around the Jan. 1 and Chinese New Year holidays.
                The situation improved slightly in March, when deliveries jumped 69.3% month-over-month to 1,373.
                Summer Lull Takes Hold
                Nio reported 1,124 vehicle deliveries in April following the EV subsidy reduction announced in late March and an economic slowdown in China that was exacerbated by the U.S.-China trade standoff. 
                Deliveries remained anemic in May, as 1,089 vehicles were delivered in the month.
                The introduction of the ES6 model June 18 helped salvage some pride, as the company improved its sales month-over-month to 1,340 in June.
                Deliveries troughed in July, when the company pushed out only 837 vehicles.
                With ES6 sales picking up momentum, deliveries improved in August to 1,943 vehicles, comprised by 146 ES8s and 1,797 ES6s.
                The weak performance over the first two quarters culminated in the company  reporting  a wider loss for the second quarter. The company announced restructuring initiatives that included job cuts.
                Turnaround Materializes
                Nio's fortunes turned around in September, when it reported an increase in deliveries of about 4% to 2,019. 
                The company followed up with a strong  October,  with deliveries jumping 25.1% to 2,526.
                Apart from improving external factors, Nio has been proactive in countering the weakness, focusing on services and announcing a collaboration with Intel Corporation's (NASDAQ:INTC) Mobileye for driverless consumer cars in China. 
                Nio shares were trading 0.55% higher at $1.84 at the time of publication. 
                Related Links:
                It's Official: Nio Brings Former Auto Analyst Wei Feng On As CFO 
                Nio Shares Trade Higher On Report of Chinese Factory Deal Talks 
                Photo courtesy of Nio. """, "7"),
        Example("""Consumer spending has been on an upward trajectory as the economy recovers from the pandemic, benefiting 
                the retail sector substantially. E-commerce companies, in particular, have seen a significant boost in sales 
                as more consumers have shifted to online shopping. With the ongoing digital transformation and technological 
                advancements, the e-commerce industry is poised for continued growth.
                Investors looking for opportunities in the retail sector should consider companies with a strong online presence 
                and a proven ability to adapt to changing consumer behaviors. Although short-term market fluctuations may affect 
                stock prices, the long-term outlook for the e-commerce industry is promising.
                """, "7"),
        Example("""Sony Group Corp's (NYSE:SONY) Sony Interactive Entertainment disclosed the launch of its all-new PlayStationPlus game subscription service in North and South America, offering more flexibility and value for gaming fans.
                The service has one monthly fee and will incorporate its separate cloud gaming platform, PlayStation Now.
                PlayStation Plus subscribers will migrate to the PlayStation Plus Essential tier with no additional payment and get cloud streaming access until their existing subscription expires. 
                Sony will also launch the service in 11 markets in Eastern Europe, covering 30 markets' access to cloud streaming.
                The updated subscription service will likely make PlayStation Plus a better competitor against Microsoft Corp's (NASDAQ:MSFT) Xbox Game Pass, the TechCrunch reports.
                Sony claimed a more extensive game catalog and higher-priced subscription plan would have access to time-limited game trials and more benefits.
                JP Morgan forecasts the gaming-market size to hit $360 billion by 2028 and music streaming to reach $55 billion by 2025.
                JP Morgan expects Apple Inc's (NASDAQ:AAPL) gaming and music offerings to likely jump 36% to $8.2 billion by 2025.
                Price Action: SONY shares closed lower by 4.69% at $83.93 on Monday.
                Photo by Macro Verch via Flickr""", "7"),
        Example("""Microsoft Corp (NASDAQ:MSFT) won the unconditional antitrust European Commission approval for the proposed acquisition of transcription software company Nuance Communications Inc (NASDAQ:NUAN).
                The regulator concluded that the transaction would raise no competition concerns in the European Economic Area.
                Related Content: Microsoft Confirms Nuance Communications Acquisition For $19.7B
                Microsoft has already won regulatory approval in the U.S. and Australia.
                The EC concluded that the transaction would not significantly reduce competition in the transcription software, cloud services, enterprise communication services, customer relationship management, productivity software, and PC operating systems markets.
                Price Action: MSFT shares traded higher by 2.24% at $327.07, while NUAN is up 0.38% at $55.20 on the last check
                Tuesday.""", "8"),
        Example("""The National Football League bagged a multiyear deal with Apple Inc's (NASDAQ:AAPL) Apple Music to sponsor the Super Bowl Halftime Show, beginning with the American football championship game in February 2023.
                The multiyear partnership combines the Super Bowl Halftime Show, the most-watched musical performance of the year, with Apple Music, which offers a powerful listening experience powered by Spatial Audio.
                Super Bowl LVII is due on February 12, 2023, in Glendale, Arizona, and will mark Apple Music's first year as part of the Super Bowl Halftime Show. 
                Over 120 million viewers watched The Super Bowl LVI Halftime Show live earlier this year, which featured a lineup of trailblazing musicians, including Dr. Dre, Snoop Dogg, Eminem, Mary J. Blige, and Kendrick Lamar. 
                NFL games remain among the top-viewed programs each year across all networks and time slots.
                Strong viewership figures for the first week of the NFL season could boost Comcast Corp (NASDAQ: CMCSA) streaming platform Peacock.
                Walt Disney Co (NYSE: DIS) has a piece of the NFL coverage with its Monday Night Football matchups.
                As the official home of Thursday Night Football, Amazon.com Inc (NASDAQ: AMZN) Amazon Prime could see additional subscribers to its Prime offering. Amazon paid a reported $1 billion for the rights.
                Price Action: AAPL shares traded lower by 1.71% at $150.14 in the premarket on the last check Friday.
                Photo via Wikimedia Commons""", "8"),
        Example("""The renewable energy sector is poised for significant growth in the coming years, driven by global efforts 
                to combat climate change and reduce greenhouse gas emissions. Governments around the world are increasingly 
                investing in clean energy projects, and many corporations are committing to reduce their carbon footprint by 
                adopting renewable energy sources.
                Technological advancements in solar, wind, and energy storage solutions have made renewable energy more accessible 
                and cost-competitive, further fueling the sector's growth. Additionally, the growing public awareness of 
                environmental issues is pushing the demand for sustainable alternatives.
                While the renewable energy sector is not without its challenges, such as fluctuating government policies and the 
                intermittent nature of some energy sources, the long-term outlook remains highly optimistic. Investors seeking 
                exposure to a high-growth industry with strong potential for positive environmental impact may find attractive 
                opportunities in the renewable energy space.
                """, "8"),
        Example("""The electric vehicle (EV) market is experiencing rapid growth, driven by increasing consumer demand, supportive 
                government policies, and the global push for sustainable transportation solutions. Major automakers are investing 
                heavily in EV development, leading to continuous innovation and improved affordability of electric vehicles.
                Furthermore, the expansion of charging infrastructure and the reduction of battery costs are also contributing 
                to the accelerated adoption of electric vehicles worldwide. As the technology improves and EVs become more 
                mainstream, it is expected that the transition from traditional internal combustion engine vehicles to electric 
                vehicles will continue at an accelerated pace.
                Investors seeking to capitalize on this burgeoning market have numerous opportunities, ranging from established 
                automakers to innovative startups, battery manufacturers, and charging infrastructure providers. With a bright 
                future ahead for the electric vehicle market, the potential for substantial returns is strong.
                """, "9"),
        Example("""Pre-open movers
                U.S. stock futures traded higher in early pre-market trade as investors are awaiting President-elect Joe Biden’s inauguration during the day.  Morgan Stanley (NYSE:MS),  UnitedHealth Group (NYSE:UNH) and  Procter & Gamble (NYSE:PG) are all set to report their quarterly earnings today.
                The NAHB housing market index for January will be released at 10:00 a.m. ET. The index is expected to remain unchanged at 86 in January from December.
                Futures for the Dow Jones Industrial Average climbed 74 points to 30,902.00 while the Standard & Poor’s 500 index futures traded gained 15.25 points to 3,805.75. Futures for the Nasdaq 100 index rose 109.25 points to 13,094.75.
                The U.S. has the highest number of COVID-19 cases and deaths in the world, with total infections in the country exceeding 24,254,140 with around 401,760 deaths. India reported a total of at least 10,595,630 confirmed cases, while Brazil confirmed over 8,573,860 cases.
                Oil prices traded higher as Brent crude futures rose 0.9% to trade at $56.40 per barrel, while US WTI crude futures rose 1% to trade at $53.50 a barrel. The API’s report on crude oil stocks will be released later during the day.
                A Peek Into Global Markets
                European markets were higher today. The Spanish Ibex Index rose 0.4% and STOXX Europe 600 Index rose 0.8%. The French CAC 40 Index rose 0.7%, German DAX 30 gained 0.8% while London's FTSE 100 rose 0.4%. UK’s producer prices declined 0.4% year-over-year, while inflation rate increased to 0.6% in December. Germany's producer prices rose 0.2% year-over-year in December
                Asian markets traded mostly higher today. Japan’s Nikkei 225 fell 0.38%, China’s Shanghai Composite rose 0.47%, Hong Kong’s Hang Seng Index gained 1.08% and India’s BSE Sensex rose 0.9%. Australia’s S&P/ASX 200 rose 0.4%. Foreign direct investment into China rose 6.2% year-on-year to CNY 999.98 billion in 2020, while People’s Bank of China kept the prime loan rate unchanged at 3.85%.
                Broker Recommendation
                Berenberg upgraded  Boeing Co (NYSE:BA) from Sell to Hold and raised the price target from $150 to $215.
                Boeing shares rose 1% to $212.72 in pre-market trading.
                Check out other major ratings here 
                Breaking News 
                Netflix Inc (NASDAQ:NFLX) reported better-than-expected Q4 sales and issued strong guidance for the first quarter. Its global streaming paid memberships climbed 21.9% year-over-year to 203.66 million during the quarter. A report also mentioned the company is exploring potential buybacks to return cash to shareholders.
                Alibaba Group Holding Ltd’s  (NYSE:BABA) founder Jack Ma made an online public appearance after months. Ma met 100 rural teachers through videoconferencing on Wednesday morning, Reuters reported. The entrepreneur had not been seen in public since Oct. 24 after he criticized China’s regulatory system at a summit in Shanghai.
                PACCAR Inc  (NASDAQ:PCAR) disclosed a strategic partnership with Aurora to develop autonomous trucks.
                Apple Inc.’s  (NASDAQ:AAPL) electric vehicles could be made by Kia Corp. at the latter’s manufacturing facility in the United States, according to a report from the Korean outlet eDaily.
                Check out other breaking news here""", "9"), 
        Example("""The biotechnology sector is on the verge of a breakthrough era, with significant advancements in gene editing, 
                personalized medicine, and drug discovery. The ongoing COVID-19 pandemic has further highlighted the importance of
                rapid medical innovation and has accelerated investment in the biotechnology industry.
                Companies are leveraging technologies such as artificial intelligence and machine learning to expedite drug 
                development processes and create more targeted therapies for various diseases, including cancer, Alzheimer's, 
                and other rare disorders. This progress is not only improving patient outcomes but also opening up new avenues for 
                revenue generation in the healthcare sector.
                Investors seeking exposure to a rapidly evolving industry with the potential to transform the way we treat 
                diseases and improve overall quality of life should consider the biotechnology sector. The combination of 
                cutting-edge technology, medical advancements, and strong growth potential make this industry a highly attractive 
                investment option.
                """, "9"),
        Example("""Target Corporation's (NYSE:TGT) business model has been enormously successful with its stock hitting all-time-highs. Its second quarter earnings exceeded Wall Street expectations, and now the company has maintained its retail outperformer status by smashing analysts' expectation with its third quarter results. Shares surged more than 10% in premarket trading as the company outperformed both earnings and sales expectations, with the company raising its full year profit outlook as the holiday season is around the corner.
                Third Quarter Results
                After weak results of Kohl's Corporation (NYSE:KSS), Target again succeeded in creating a bright spot in retail. For the period ended November 2, adjusted earnings per share came to $1.36 compared to $1.19. Achieved revenue amounted to $18.67 billion as opposed to analyst's estimation of $18.49. Total revenue grew 4.7% comparing to the previous year's quarter.
                Sales at stores which were open for at least 12 months along with online sales grew 4.5% also exceeding the expected 3.6%. In fact, digital sales witnessed an impressive growth rate of 31%.
                The company has upgraded its full-year adjusted earnings per share which are expected in the range of $6.25 to $6.45, whereas the prior estimate was $5.90 to $6.20. Net income grew from $622 million to $714 million from the same quarter last year.
                Successful Strategy
                Wall Street has expected a successful report to Target's brand strategy that has become a model for struggling retailers. To keep shoppers happy, Target partnered with celebrity designers to create popular lines that sell out quickly. It opened small-format locations at trending locations like New York and around campuses, refurbished its existing store design and even launched a grocery line. They even joined forces with the Walt Disney Company (NYSE:DIS) so some Target stores also have a mini Disney shop within their offerings. Although its big box competitor Walmart Inc (NYSE:WMT) who also just reported better than expected earnings last week. Although Walmart has a beyond massive market cap of $341 billion with Target having $56.5 billion, do not be fooled by size as Target's stock rose at an impressive growth rate of 67% since the beginning of the year.
                And even though the big box retailer's e-commerce sales were up 41% during the most recent quarter, Walmart admitted it still has more work to do online. And with Amazon.com, Inc. (NASDAQ:AMZN) ""pulling out the big guns"", the online competition is only getting more intense. Target might be weaker in the groceries segment, but it holds a stronger position with so-called ‘toys' or athleisure: namely, apparel, beauty and home goods, while also launching more in-house brands. So, if you think of the holiday season of gift giving- all seems to be working out in Target's favour.
                Outlook
                Third quarter results are just another proof of the durability of the company's strategy- so here's a big applause for the company's top management. Target enabled customers to easily find the products they need, whether in store or online and the range is beyond wide. 2018 holiday season was Target's most successful over a decade, and there's no reason to think 2019 will have trouble keeping up.
                Target announced in October that it will increase spending by $50 million comparing to last year's comparable quarter and due to payroll. With overtime and increased number of employees, Target seems determined to maintain its status as a ‘retail outperformer', with the ‘holiday' fortune on its side!
                This Publication is contributed by IAMNewswire.com
                Press Releases - If you are looking for full Press release distribution contact: press@iamnewswire.com
                Contributors - IAM Newswire accepts pitches. If you’re interested in becoming an IAM journalist contact: contributors@iamnewswire.com
                Copyright © 2019 Benzinga (BZ Newswire, http://www.benzinga.com/licensing).
                Benzinga does not provide investmentadvice. All rights reserved.
                Write to editorial@benzinga.com with any questions about this content. Subscribe to Benzinga Pro (http://pro.benzinga.com).
                © 2019 Benzinga.com. Benzinga does not provide investment advice. All rights reserved.
                Image by Markus Spiske from Pixabay""", "9"),
        Example("""The renewable energy sector is poised for unprecedented growth as the world shifts towards cleaner and more sustainable energy sources in response to climate change. The increasing demand for renewable energy, coupled with supportive government policies, rapidly improving technologies, and declining costs, has created an environment that fosters rapid expansion and innovation within the industry.
                Companies operating in solar, wind, hydro, and other renewable energy subsectors are benefiting from this trend, leading to the development of more efficient, reliable, and cost-effective energy solutions. This progress presents an extraordinary opportunity for investors seeking to participate in a transformative global movement with the potential to reshape the way we power our world.
                The renewable energy sector is not only positioned for strong financial growth but also offers investors an opportunity to contribute to a sustainable and environmentally responsible future. With the dual benefits of significant investment potential and positive environmental impact, this industry represents a perfect investment opportunity.
                """, "10"),
        Example("""The artificial intelligence (AI) and machine learning (ML) industry has become a driving force behind some of the most significant technological advancements of the 21st century. The applications of AI and ML are virtually limitless, with the potential to revolutionize industries such as healthcare, finance, transportation, and manufacturing.
                As AI and ML technologies continue to advance, companies at the forefront of this innovation are developing groundbreaking solutions that improve efficiency, lower costs, and enhance the overall quality of life. The global market for AI and ML is projected to experience exponential growth in the coming years, providing investors with a rare opportunity to capitalize on a technological revolution that is still in its infancy.
                Investing in the AI and ML industry not only offers the potential for exceptional financial returns but also allows investors to be part of a movement that is transforming the world and driving the future of innovation. This industry represents an ideal investment opportunity that combines unparalleled growth prospects with the chance to make a lasting impact on society.            
                """, "10"),
        ]

        response = co.classify(
        inputs=input,
        examples=examples,
        )
        
        return response
        


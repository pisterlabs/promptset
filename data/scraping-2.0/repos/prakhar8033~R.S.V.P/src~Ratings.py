import openai

def RatingChecker(transcript):
    openai.api_key = '----your--openai--api--key----'

    # prompt input deciding the criteria of rating.
    promptInput = """Consider this scenario. I am a Sales person working in Crio.Do whose goal is to empower developers with high quality applied learning opportunities at scale, and nurture the talent for the future.

    Crio has a learning platform that fundamentally changes the way tech is learnt through our “work-like” Micro-Experiences. Crio Micro-Experiences provide an environment optimised for learning, with real world problems curated from the industry. We at Crio highly believe in the "Learn By Doing" approach for the students that take the courses. 

    Some of the points about Crio.Do as are follows - 

    1. Work-experience based learning programs to land your dream tech job
    2. Learn Like You Would At India’s Top Tech Companies.
    3. Build professional projects
    4. Master the latest Fullstack/Backend/Automation tech with real work-ex.
    5. Start from the fundamentals, receive support from our mentors and community, and build your way to the top - through professional work-like Full-stack and Backend web development projects.
    6. Get career support to break through into ambitious roles
    7. Build a GitHub portfolio that impresses recruiters
    8. Gain the work experience of professional developers working at Amazon, Netflix, AirBnB, Flipkart, and more, with continuous guidance and support from our mentors.
    9. Learn the skills you need to land a Fullstack/Backend Developers or SDET Job at a top product company.
    10. 1:1 Career coaching Sessions with top industry professionals
    11. Live mock interviews with Industry Experts
    12. Series of mock assessments and detailed interview prep sprints to ace top tech jobs
    13. Expert guidance to get your profile (GitHub, LinkedIn, Resume) ready
    14. Guidance to start applying to a diverse set of job opportunities in-line with your career aspirations.

    These are the four courses that Crio sell in the career programs - 
    1. Fellowship Full-Stack Program (Full-Stack Track called as FDT)
    2. Fellowship Backend Program (Backend Track called as BDT)
    3. Fellowship Advanced Track (DSA Track called as DSA)
    4. Masters in QA Automation Track (QA Testing Automation track called as QA)

    The DSA track is an additional track that we sell to the students opting for any one of  the tracks from FDT, BDT and QA. 

    We sell out courses(tracks) both to the students that are currently studiying in some college and to the working professionals who want to upskill themselves in the same domain or switch to another domain like some tech guy working as a Manual tester and want to upskill by the QA track, or some other guy working in non-IT field and want to switch to IT field as a SDE in Full Stack. If the user is a working  professional, then we pitch by telling  them to provide a better CTC after graduating the course compared to the current salary of theirs.

    We have an excellent Delivery help expert team that help in Onboarding users for the journey, Technical Team that help users in solving the user queries that they can ask through the Chat feature available for them, Customer Success team that provide assistance in Crio referrals to help through the placement phase once the course is completed. 

    For the technical queries asked by the users, we tend to follow the "Learn By Doing" approach so that instead of solving the users problem straightaway, we interactively communicate with the user to first make them understance what is the error, the cause of it, and then asking and helping them out to provide the best possible solutions for it. 

    To explore the Crio platform before purchasing the full course, we provide a Free Demo Trial for a certain period of time to make them into liking the platform and hence buying the course.

    While making a call with the user, we our Sales team keep in mind some following points - 

    1. Making the call interactive.
    2. Ask open-ended questions to the user.
    3. Getting into the shoes of the user.
    4. Keep the mood of the call informative and positive.
    5. Don't make false promises to the user.
    6. Address any questions or concerns by the user.
    7. Ask about the user whether he is a student/working professional and continue the conversation accordingly.
    8. Try to make the user speak more.
    """

    # calling Completion model of OpenAI API
    response = openai.Completion.create(

    engine="text-davinci-003",

    prompt=f""" {promptInput} Act as an AI model to rate on a scale of 1 to 10 the conversation of me as a Sales person with the user. 

    Calculate a final score rating after calculating Engagement (Was the conversation engaging and interactive?), Understanding (Did the salesperson understand the user's goals and concerns?), Information Sharing (Was relevant information provided about the courses and learning experience?),
    Empathy (Did the salesperson show empathy towards the user's situation?), Transparency (Were false promises avoided and realistic expectations set?), Addressing Concerns (Were the user's questions and concerns adequately addressed?),
    User Participation (Did the salesperson encourage the user to speak and share more?), Clarity (Was the information presented in a clear and understandable manner?), Offer Relevance (Did the salesperson match the user's needs with the appropriate course?), Positive Tone (Was the mood of the call kept informative and positive?).
    
    Give the output this way, and one more thing, don't give rating in the decimals, ratings should be in integers only:

    Rating: calculated_rating/10

    This is the conversation : {transcript} """,
    max_tokens=100,
        temperature=0.6
    )

    resRating = response.choices[0].text.strip()

    # print(resRating)
    return resRating
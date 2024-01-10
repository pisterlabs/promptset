import cohere

co = cohere.Client('6xl3SHALT0x7HMKIhQ9A1SMCpJppaImByJJcufOq') # This is your trial API key

class CoverLetter:

  def __int__(self, role, job_description, about_company, name, about_me):
    self.role = role
    self.job_description = job_description
    self.about_company = about_company
    self.name = name
    self.about_me = about_me

  def generate(self):
    try:
      response = co.generate(
      model='command-xlarge-nightly',
      prompt=f'Role: \nPython Developer\nJob Description:\nWe are looking for a talented Python Developer to join our team. As a Python Developer, you will be responsible for designing, developing, and testing software applications. You will work on complex projects and have the opportunity to work with a dynamic and talented team. The ideal candidate will have strong experience with Python, as well as experience with other programming languages.\nResponsibilities:\nDesign and develop software applications using Python\nCollaborate with cross-functional teams to develop new features and products\nWrite clean, maintainable, and efficient code\nTest and debug software applications\nEnsure software meets specified requirements and standards\nParticipate in code and design reviews\nRequirements:\nStrong experience with Python\nExperience with other programming languages (Java, C++, etc.)\nExcellent problem-solving skills\nKnowledge of software development best practices\nExcellent communication and collaboration skills\nFamiliarity with Agile software development methodologies\nAbility to work independently and as part of a team\nAbout Company:\nWe are a leading software development company, specializing in the development of custom software applications. Our team of experienced developers is dedicated to delivering innovative solutions that meet the needs of our clients. We are committed to providing a supportive and collaborative work environment, and to fostering the growth and development of our employees.\nCandidate Name: \nMuhammad Ahsan\nAbout Me:\nI am a highly motivated and experienced Python Developer with a passion for creating innovative software solutions. With a strong background in Python, I have developed a wide range of software applications and have experience working with a variety of programming languages. I am confident in my ability to work independently and as part of a team, and am always seeking new challenges and opportunities for growth.\nCover Letter:\nDear Hiring Manager,\nI am writing to express my interest in the Python Developer role at your company. As a passionate and experienced Python Developer, I am confident in my ability to make a valuable contribution to your team.\nI have extensive experience designing, developing, and testing software applications using Python. I have a strong background in a variety of programming languages, and have a proven track record of delivering high-quality software solutions. I am committed to using my problem-solving skills and technical expertise to deliver innovative and effective software solutions that meet the needs of your clients.\nI am impressed by the commitment to providing a supportive and collaborative work environment at your company, and I am eager to work with your talented team of developers. I believe that my passion for software development and my dedication to continuous learning and growth make me an excellent fit for this role.\nI would welcome the opportunity to discuss my qualifications further and to learn more about the opportunities available at your company. Thank you for considering my application.\nSincerely,\nMuhammad Ahsan\n--\nRole: \n{self.role}\nJob Description:\n{self.job_description}\nAbout Company:\n{self.about_company}\nCandidate Name: \n{self.name}\nAbout Me:\n{self.about_me}\nCover Letter:',
      max_tokens=714,
      temperature=0.6,
      k=165,
      p=0.25,
      frequency_penalty=0.03,
      presence_penalty=0,
      stop_sequences=["--"],
      return_likelihoods='GENERATION')

      print('Prediction: {}'.format(response.generations[0].text))
    except Exception as e:
      return e

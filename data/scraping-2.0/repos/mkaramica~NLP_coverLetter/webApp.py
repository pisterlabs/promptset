from flask import Flask, render_template_string, request
import openai
import os

app = Flask(__name__)

# Set openAI key:
openai.api_key = os.getenv("OPENAI_API_KEY")

class CoverLetterGenerator:
    def __init__(self, job_description, resume, nWords):
        self.job_description = job_description
        self.resume = resume
        self.nWords = nWords
    
    def generate_cover_letter(self):
        # Initialize the conversation with the assistant
        conversation = [
            {"role": "system", "content": "You are a helpful assistant that writes cover letters."},
            {"role": "user", "content": f"My resume: {self.resume}"},
            {"role": "user", "content": f"Write a cover letter based on this job description: {self.job_description}"},
            {"role": "user", "content": f"Keep the length of the letter around {self.nWords} words."}
        ]

        # Send the conversation to the API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation
        )

        # Extract the cover letter from the response
        cover_letter = response['choices'][0]['message']['content']

        return cover_letter

NumbOfWords_default = 300

resume = '''
John Smith
123 Main Street
Anytown, USA 12345
(123) 555-5555
johnsmith@email.com

Objective:
To obtain a position as a front-end developer where I can utilize my skills and experience to create visually appealing and user-friendly websites.

Education:
Bachelor of Science in Computer Science
XYZ University, Anytown, USA
Graduated May 20XX

Skills:

Proficient in HTML, CSS, JavaScript, and jQuery
Experience with React, Angular, and Vue.js
Knowledgeable in responsive design and cross-browser compatibility
Familiarity with Adobe Creative Suite and Sketch
Strong understanding of software development methodologies, including Agile and Waterfall
Experience:
Front-End Developer
ABC Company, Anytown, USA
May 20XX - Present

Develop and maintain websites using HTML, CSS, and JavaScript
Collaborate with cross-functional teams to ensure site functionality and usability
Implement responsive design and ensure cross-browser compatibility
Utilize version control tools such as Git to manage code changes
Participate in Agile development process
Web Developer Intern
DEF Company, Anytown, USA
January 20XX - April 20XX

Assisted senior developers with front-end development tasks
Gained experience working with HTML, CSS, and JavaScript
Collaborated with team members to develop websites and web applications
Gained familiarity with Agile development process
Projects:

Created a responsive web design using HTML, CSS, and JavaScript
Developed a website using React and Redux
Implemented a mobile app using React Native
Designed a user interface using Adobe XD
Certifications:

Certified Front-End Web Developer (CFWD), XYZ Certification, 20XX
References:
Available upon request.
'''

job_description = '''
About the Role

We are looking for an experienced and talented developer to join our product team, who engages with our awesome team of developers, designer and product owner and is committed to innovation and continuous value creation. You will have the opportunity to own features & contribute to our user experience with the latest tools and design patterns.

At Verge, we enjoy flexible work arrangements, including a work-from-home environment. While we don't have a physical office space, we hire locally (Calgary area) to ensure our teams can meet in person regularly, allowing our culture of collaboration, creativity, and innovation to thrive.

What You will Do:

Design and write well-structured, documented and easily maintainable code
Stay at forefront of web application development best practices
Collaborate with other devs and UI/UX designer
About you

What you bring

Minimum of two years technical education
Three or more years of hands-on professional experience and proven track record
Exceptional UI/UX Skills
Angular/Angular Material/React/React Native/Vue/Bootstrap
SCSS/CSS/HTML
JavaScript/TypeScript
GitHub/Git/Version Control
Scrum/agile team experience
Portfolio links as part of resume
What makes you stand out:

AgTech Experience
Geo/GIS/Mapping
Visual Studio/VS Code
Familiarity with Figma or other interface design tools
Responsive web design is in our near-future
Benefits of being at Verge

Because Verge hires the best people, we work hard to provide benefits and perks to improve their lives. Our people get competitive salaries and vacation, flexible health benefits, generous bonus program, retirement planning options (401K and RRSP matching), personal development and work from home allowances, to name a few of the perks.

When you join Verge, you do more than switch companies to advance your career. Instead, you're joining a team of talented individuals who work with passion and purpose. We're a team with strong company values that define our culture, and they include: delivering on our promises, growth through market-driven innovation, driving for results, communicating with clarity, and a focus on valuing creative teamwork.

If you want to play a role in accelerating the transition to autonomous farming, you belong at Verge.

NOTE: We will only contact qualified candidates of interest.

Verge is an equal opportunity employer. We encourage applications from all individuals regardless of age, gender, race, ethnicity, religion or sexual orientation, and evaluate all candidates based on merit.

Join Verge and change the future of agriculture - https://vergeag.com

Job Types: Full-time, Permanent

Benefits:

Dental care
Extended health care
Paid time off
RRSP match
Vision care
Schedule:

Monday to Friday
Supplemental pay types:

Bonus pay
Education:

Secondary School (preferred)
Experience:

Software development: 1 year (preferred)
UI Design and Development: 1 year (preferred)
Angular: 1 year (preferred)
SCSS/CSS/HTML: 1 year (preferred)
JavaScript/TypeScript: 1 year (preferred)
Job Types: Full-time, Permanent

Benefits:

Dental care
Extended health care
Paid time off
RRSP match
Vision care
Schedule:

Monday to Friday
Supplemental pay types:

Bonus pay
Experience:

Software development: 1 year (preferred)
UI Design and Development: 1 year (preferred)
Angular: 1 year (preferred)
SCSS/CSS/HTML: 1 year (preferred)
JavaScript/TypeScript: 1 year (preferred)
Work Location: Hybrid remote in Calgary, AB
'''

html_template = '''
<!DOCTYPE html>
<html>
  <head>
    <title>Cover Letter Generator</title>
  </head>
  <body>
    <h1>Cover Letter Generator</h1>
    <form method="POST" action="/generate">
      <div style="display: flex;">
        <div style="margin-right: 20px;">
          <label for="resume">Resume:</label><br>
          <textarea id="resume" name="resume" rows="25" cols="100">{{ resume }}</textarea>
        </div>
        <div>
          <label for="job_description">Job Description:</label><br>
          <textarea id="job_description" name="job_description" rows="25" cols="100">{{ job_description }}</textarea>
        </div>
      </div>
      <br>
      <label for="num_words">Number of Words:</label><br>
      <input type="text" id="num_words" name="num_words" value="{{nWords}}"><br><br>
      <input type="submit" value="Generate Cover Letter">
    </form>
    <br>
    <div>
      <label for="cover_letter">Cover Letter:</label><br>
      <textarea id="cover_letter" name="cover_letter" rows="25" cols="100" readonly>{{ cover_letter }}</textarea>
    </div>
  </body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(html_template, 
    resume=resume, 
    job_description = job_description,
    nWords = NumbOfWords_default,
    cover_letter = "\n"+"-"*10+"COVER LETTER"+"-"*10)

@app.route('/generate', methods=['POST'])
def generate():
    # Retrieve values from the form
    resume = request.form['resume']
    job_description = request.form['job_description']
    nWords = request.form['num_words']
    

    # Generate the cover letter
    cover_letter_generator = \
    CoverLetterGenerator(job_description = job_description,
    resume = resume,
    nWords= nWords
    )
    
    cover_letter = cover_letter_generator.generate_cover_letter() 

    # Return the cover letter as a response
    return render_template_string(html_template, 
    resume=resume, 
    job_description = job_description,
    nWords = NumbOfWords_default,
    cover_letter = cover_letter)
    
if __name__ == '__main__':
    app.run(debug=True)

#import BaseHTTPServer, HTTPServer
from http.server import BaseHTTPRequestHandler, HTTPServer
import os
import openai
import json

#for local development use localhost,
#but for production use 0.0.0.0

#if dev
#if system ENV variable is set to dev, use localhost
if os.environ.get('ENV') == 'dev':
    hostName = "localhost"
else:
    hostName = "0.0.0.0"

#if ENV is dev, use port 8080
#else use port 80
if os.environ.get('ENV') == 'dev':
    serverPort = 8080
else:   
    serverPort = 80

class MyServer(BaseHTTPRequestHandler):

    def MyServer(self):
        #start with empty topics dictionary
        self.topics_dict = {}
    

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        
        
        # Get the path requested by the client
        request_path = self.path
        
        #debug code to print request path
        if os.environ.get('ENV') == 'dev':
            print(request_path)

        #strip of the initial slash
        request_path = request_path[1:]

        #strip of the trailing slash
        if request_path.endswith('/'):
            request_path = request_path[:-1]

        #debug code to print request path
        if os.environ.get('ENV') == 'dev':
            print(request_path)

        #build html_style
        html_style = """
        <style>
         body {
            font-family: sans-serif;
            font-size: 16px;
            color: #fff;
            background-color: #222;
            }
            
            h1, h2, h3 {
            font-weight: bold;
            color: #fff;
            text-align: center;
            }
            
            p {
            margin-bottom: 1em;
            }
            
            a {
            color: #f00;
            text-decoration: none;
            }
            
            a:hover {
            text-decoration: underline;
            }
            
            nav {
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: #222;
            padding: 10px;
            }
            
            nav ul {
            list-style-type: none;
            }
            
            nav li {
            display: inline-block;
            margin: 0 10px;
            }
            
            nav li a {
            color: #f00;
            text-decoration: none;
            padding: 10px 20px;
            border: 1px solid #fff;
            border-radius: 5px;
            }
            
            nav li a:hover {
            background-color: #fff;
            color: #222;
            }

            .box {
            width: 500px;
            height: 300px;
            background-color: #fff;
            border: 1px solid #222;
            padding: 20px;
            margin: 0 auto;
            text-align: center;
            color: #222;
            }
        </style>
        """
            
        #build html_header
        html_header = "<html><head><title>https://attheintersectionof.com</title>" + html_style + "</head><body>"

        #build navigation bar with home, about, ai generated articles, human generated articles
        html_nav = "<nav><ul><li><a href='/'>Home</a></li> | <li><a href='/about'>About</a></li> | <li><a href='/browseai'>AI Articles</a></li> | <li><a href='/browsehuman'>Human Articles</a><li></ul></nav>"


        #html_nav = "<nav><ul><li><a href='/'>Home</a></li> | <li><a href='/about'>About</a></li> | <li><a href='/browse'>Browse Articles</a></li></ul></nav>"


        if "well-know" not in request_path:

            #print beginning of html
            self.wfile.write(bytes(html_header, "utf-8"))
            self.wfile.write(bytes(html_nav, "utf-8"))

        if request_path == "":
            #print the welcome message
            #build html_home with welcom message and mailing list signup
            html_home = "<p>Welcome to At the Interesection of!</p><p>A place where things finally come together</p>"
            self.wfile.write(bytes(html_home, "utf-8"))
        elif request_path == "about":
            #print the about message
            #build html_about with about message
            html_about = """
            <h1>We are creating the future</h1>
            <div class="box">
            <p>We are a community of generalists who are working alongside AI to create a future that includes both humans and AI. We are creating the future.</p>
            <p>We create articles, tutorials, and courses with human skills and with the help of state-of-the-art AI.</p>
            <p>We believe that AI has the potential to make the world a better place, and we are committed to using AI for good.</p>
            <p>We are a diverse group of people from all walks of life, and we are united by our shared belief in the power of AI to make the world a better place.</p>
            <p>We are creating the future, and we invite you to join us.</p>
            </div>
            """
            self.wfile.write(bytes(html_about, "utf-8"))
        elif request_path == "about-anything":
            #print the about message
            #build html_about with about message
            html_about = """
            <h1>We are creating the future</h1>
            <div class="box">
            <p>Hi.</p>
            </div>
            """
            self.wfile.write(bytes(html_about, "utf-8"))
        elif request_path == "browseai":
            #print the browse message
            #build html_browse with browseai page


            html_browseai = "<p>AI Generated Articles</p>"
            #next do intersections
            # Intersections header
            html_browseai = html_browseai + "<b>Intersections</b>"    
            intersections_dict = self.get_topic_pairs_intersections()

            #go through list of intersections and print the intersection and content for it
            
            for intersection in intersections_dict:
                #if intersection starts with a . then skip it
                if intersection.startswith('.'):
                    continue
                #else if intersection contains php then skip it
                elif "php" in intersection:
                    continue
                #else if intersection contains txt then skip it
                elif "txt" in intersection:
                    continue
                #else if intersection contains xml then skip it
                elif "xml" in intersection:
                    continue
                #else if intersection contains a question mark then skip it
                elif "?" in intersection:
                    continue
                #else if intersection contains .json then skip it
                elif ".json" in intersection:
                    continue
                #else if intersection contains .html then skip it
                elif ".html" in intersection:
                    continue
                #else if intersection contains .env or .ENV then skip it
                elif ".env" in intersection:
                    continue
                elif ".ENV" in intersection:
                    continue
                #else if intersection contains .py then skip it
                elif ".py" in intersection:
                    continue
                #else if intersection contains .sh then skip it
                elif ".sh" in intersection:
                    continue
                #else if intersection contains .rb then skip it
                elif ".rb" in intersection:
                    continue
                #else if intersection contains .js then skip it
                elif ".js" in intersection:
                    continue
                #else if intersection contains .css then skip it
                elif ".css" in intersection:
                    continue
                #else if intersection contains .md then skip it
                elif ".md" in intersection:
                    continue
                #else if intersection contains .git then skip it
                elif ".git" in intersection:
                    continue
                #else if intersection starts with _ then skip it
                elif intersection.startswith('_'):
                    continue
                #else if intersection contains json then skip it
                elif "json" in intersection:
                    continue
                #else if intersection contains .properties then skip it
                elif ".properties" in intersection:
                    continue
                #else if intersection contains .gitignore then skip it
                elif ".gitignore" in intersection:
                    continue
                #else if intersection contains .gitattributes then skip it
                elif ".gitattributes" in intersection:
                    continue
                

                #intersection lead-in
                html_browseai = html_browseai + "<p>At the insection of "
                topics = intersection.split('_')
                for topic in topics:
                    html_browseai += "<a href=/" + topic + ">" + topic + "</a> and "
                html_browseai = html_browseai[:-5]
                html_browseai += ":</p>"
                #intersection link
                html_browseai = html_browseai + "<a href=/"
                topics = intersection.split('_')
                for topic in topics:
                    html_browseai += topic + "/"
        
                html_browseai += "> " + intersection + "</a></p>"                
                    
            
            #go through list of topics and print the topic and content for it
            topics_dict = self.read_topics()
            #Hot topics header
            html_browseai = html_browseai + "<b>Hot Topics</b>"
            for topic in topics_dict:
                #if topic starts with a . then skip it
                if topic.startswith('.'):
                    continue
                #else if topic contains php then skip it
                elif "php" in topic:
                    continue
                #else if topic contains txt then skip it
                elif "txt" in topic:
                    continue
                #else if topic contains xml then skip it
                elif "xml" in topic:
                    continue
                #else if topics contains a question mark then skip it
                elif "?" in topic:
                    continue
                #else if topic contains .json then skip it
                elif ".json" in topic:
                    continue
                #else if topic contains .html then skip it
                elif ".html" in topic:
                    continue
                #else if topic contains .env or .ENV then skip it
                elif ".env" in topic:
                    continue
                elif ".ENV" in topic:
                    continue
                #else if topic contains .py then skip it
                elif ".py" in topic:
                    continue
                #else if topic contains .sh then skip it
                elif ".sh" in topic:
                    continue
                #else if topic contains .rb then skip it
                elif ".rb" in topic:
                    continue
                #print topic
                html_browseai = html_browseai + "<p><a href='/" + topic + "'>" + topic + "</a></p>"

            #afer loops return html_browseai
            self.wfile.write(bytes(html_browseai, "utf-8"))

            
        elif request_path == "browsehuman":
            #print the browse message
            #build html_browse with browsehuman page
            html_browsehuman = "<p>Human Generated Articles</p>"
            self.wfile.write(bytes(html_browsehuman, "utf-8"))
        else:
            # Split the path into topics by slashes
            topics = request_path.split('/')
            #with open("/home/public/" + topics[0] + "/" + topics[1] + "/" + topics[2], 'r') as myfile:
            #            data=myfile.read()
            #            self.wfile.write(bytes(data, "utf-8"))
            #            return

            #debug code to print topics
            if os.environ.get('ENV') == 'dev':
                #print topic: topic
                print("topics: " + str(topics))

            #debug print lenth of topics
            if os.environ.get('ENV') == 'dev':
                print("length of topics: " + str(len(topics)))

            # If there is only one topic, display that topic's content
            if len(topics) == 1:

                if topics[0] == "favicon.ico":
                    return
                

                #read existing topics
                single_topics = self.read_topics()

                topic_name = topics[0]
                #debug code to print single topic
                if os.environ.get('ENV') == 'dev':
                    print("single topic: " + topic_name)

                if topic_name in single_topics:

                    #strip quotes from the beginning of the article
                    article_content = single_topics[topic_name].lstrip('"')

                    #self.wfile.write(bytes("<p>%s</p>" % topic_name, "utf-8"))
                    self.wfile.write(bytes("<p>%s</p>" % article_content, "utf-8"))
                else: #topic not found in single_topics
                    #build prompt to write a sentence
                    prompt = "write a sentence that ends with a period and doesn't start wtih a quote and is written with flair about " + topic_name + ":"
                    #openai completion


                    #debug code to print prompt
                    if os.environ.get('ENV') == 'dev':
                        print("prompt: " + str(prompt))

                    openai.api_key = os.environ.get('OPENAI_API_KEY')

                        
                    messages = [{"role": "user", "content": prompt}]

                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=messages,
                        stop=["."],
                        temperature=1.0,
                    )
                    article_content = response['choices'][0]['message']['content']
                    
                    #strip quotes from the beginning of the article
                    article_content = article_content.lstrip('"')

                    # Write the response to the page
                    self.wfile.write(bytes("<p>%s.</p>" % article_content, "utf-8"))

                    # Print the generated article
                    if os.environ.get('ENV') == 'dev':
                            print (article_content)

                    #cache the generated article in the topics dictionary
                    single_topics[topic_name] = article_content
                    self.write_topics(single_topics)
                

                
            
           # If there are multiple topics, display the intersection of those topics
            elif len(topics) >= 2:

                #if first topic is ".well-known" then return
                if topics[0] == ".well-known" and topics[1] == "acme-challenge":
                    #read topics[2] and write as response
                    with open("/home/public/" + topics[0] + "/" + topics[1] + "/" + topics[2], 'r') as myfile:
                        data=myfile.read()
                        self.wfile.write(bytes(data, "utf-8"))
                        return

                topic_pairs_intersections = self.get_topic_pairs_intersections()
                topic_key = "_".join(sorted(topics))

                #debug code to print topic_key
                if os.environ.get('ENV') == 'dev':
                    print("topic_key: " + str(topic_key))

                if topic_key in topic_pairs_intersections.keys():
                    #debug code to print topic_key
                    if os.environ.get('ENV') == 'dev':
                        print("topic_key: " + str(topic_key))

                    #debug code to print topic_pairs_intersections[topic_key]
                    if os.environ.get('ENV') == 'dev':
                        print("topic_pairs_intersections[topic_key]: " + str(topic_pairs_intersections[topic_key]))

                    #build prompt to write a sentence
                    prompt = "write a sentence that ends with a period and doesn't start wtih a quote and is written with flair about " + topic_key + ":"


                    #debug code to print prompt
                    if os.environ.get('ENV') == 'dev':
                        print("prompt: " + str(prompt))


                    openai.api_key = os.environ.get('OPENAI_API_KEY')

                    
                    messages = [{"role": "user", "content": prompt}]

                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=messages,
                        stop=["."],
                        temperature=1.0,
                    )
                    article_content = response['choices'][0]['message']['content']
                   
                    # Write the response to the page
                    self.wfile.write(bytes("<p>%s.</p>" % article_content, "utf-8"))

                    # Print the generated article
                    if os.environ.get('ENV') == 'dev':
                         print (article_content)

                    #cache the generated article in the insersections topics dictionary
                    topic_pairs_intersections[topic_key] = article_content
                    self.write_topic_pairs_intersections(topic_pairs_intersections)





                

                if topic_key in topic_pairs_intersections.keys():
                    #debug print topic_key found
                    if os.environ.get('ENV') == 'dev':
                        print("topic_key found")
                        #self.wfile.write(bytes("<p>%s</p>" % topic_key, "utf-8"))

                    #debug code to print topic_pairs_intersections
                    if os.environ.get('ENV') == 'dev':
                        print("topic_pairs_intersections: " + str(topic_pairs_intersections))

                    
                    intersection = topic_pairs_intersections[topic_key]
                    #self.wfile.write(bytes("<p>Intersection: %s</p>" % intersection, "utf-8"))
                else:
                    
                    #debug print topic_key not found
                    if os.environ.get('ENV') == 'dev':
                        print("topic_key not found")
                    
                    #split the topic_key into topics
                    topics = topic_key.split('_')
                    prompt = "write a sentence that ends with a period and doesn't start wtih a quote and is written with flair about " + topics[0]
                    rest_of_topics = topics[1:]
                    index = 1
                    for topic in rest_of_topics:
                        if index != rest_of_topics[len(rest_of_topics) -1]:
                            prompt += " and "
                        index += 1
                        prompt += topic
                       
                    
                    #debug code to print prompt
                    if os.environ.get('ENV') == 'dev':
                        print("prompt: " + str(prompt))


                    openai.api_key = os.environ.get('OPENAI_API_KEY')

                    
                    messages = [{"role": "user", "content": prompt}]

                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=messages,
                        stop=["."],
                        temperature=1.0,
                    )
                    article_content = response['choices'][0]['message']['content']
                   
                    # Write the response to the page
                    self.wfile.write(bytes("<p>%s.</p>" % article_content, "utf-8"))

                    # Print the generated article
                    if os.environ.get('ENV') == 'dev':
                         print (article_content)

                    #cache the generated article in the topic_pairs_intersections
                    topic_pairs_intersections[topic_key] = article_content 
                    self.write_topic_pairs_intersections(topic_pairs_intersections)
                        
            # If the request path is invalid, display a message
            else:
                self.wfile.write(b"Invalid request path")
        
        # End the HTML response
        self.wfile.write(bytes("</body></html>", "utf-8"))

    def write_topics(self, topics_dict):
     
        #write topics_dict to file with newlines between each topic  
        with open('topics.json', 'w') as outfile:
            json.dump(topics_dict, outfile, indent=2, separators=(',', ': '))

        #debug code to print topics_dict
        if os.environ.get('ENV') == 'dev':
            print("topics dict: " + str(topics_dict))
        
        #debug code to check file contents
        if os.environ.get('ENV') == 'dev':
            with open('topics.json') as json_file:
                topics_dict = json.load(json_file)
                print("topics_dict: " + str(topics_dict))


    def get_topic_pairs_intersections(self):
        #load topic_pairs_intersections from file cache topic_pairs_intersections.json or create if it doesn't exist
        
        #check if file exists
        if os.path.isfile('topic_pairs_intersections.json'):

            with open('topic_pairs_intersections.json') as json_file:
                topic_pairs_intersections = json.load(json_file)
                print("topic_pairs_intersections: " + str(topic_pairs_intersections))
                return topic_pairs_intersections
        else:
            topic_pairs_intersections = {}
            #create empty file
            with open('topic_pairs_intersections.json', 'w') as outfile:
                #write empty json to file
                json.dump(topic_pairs_intersections, outfile, indent=2, separators=(',', ': '))

            return topic_pairs_intersections

    def write_topic_pairs_intersections(self, topic_pairs_intersections):
        #write topic_pairs_intersections to file cache topic_pairs_intersections.json
        with open('topic_pairs_intersections.json', 'w') as outfile:
            json.dump(topic_pairs_intersections, outfile, indent=2, separators=(',', ': '))
    
    def read_topics(self):
        #load topics from file cache topics.json or create if it doesn't exist
        topics = {}
        #check if file exists
        if os.path.isfile('topics.json'):

            with open('topics.json') as json_file:
                topics = json.load(json_file)
                print("topics: " + str(topics))
                return topics
        else:
            topics = {}
            #create empty file
            with open('topics.json', 'w') as outfile:
                #write empty json to file
                json.dump(topics, outfile, indent=2, separators=(',', ': '))

            return topics
    


if __name__ == "__main__":       

    webServer = HTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")

    

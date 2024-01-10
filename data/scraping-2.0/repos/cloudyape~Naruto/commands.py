import html
import json
import re
import subprocess
import sys
import os
import socket
import time
import signal
import webbrowser
import threading
import io
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os

 
def get_dir_size(path='.'):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total

def is_port_open(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)

    try:
        result = sock.connect_ex(('127.0.0.1', port))
        return result == 0
    except Exception as e:
        print(f"Error checking port {port}: {e}")
        return False
    finally:
        sock.close()

class MyHandler(SimpleHTTPRequestHandler):

    def do_GET(self):
        requested_path = self.path

        if requested_path.endswith('/'):
            requested_path += 'flag.html'

        self.path = requested_path

        result = SimpleHTTPRequestHandler.do_GET(self)

        if "404 Not Found" in str(result):
            self.path = '/404.html'
            result = SimpleHTTPRequestHandler.do_GET(self)

        return result

def remove_dom_content_loaded(filename):
    with open(filename, 'r') as file:
        content = file.read()

    # Use regular expression to remove the specified block
    pattern = re.compile(r'document\.addEventListener\("DOMContentLoaded",\s*function\s*\(\)\s*{.*?}\);', re.DOTALL)
    content = re.sub(pattern, '', content)

    with open(filename, 'w') as file:
        file.write(content)

def add_base_url(html_file, base_url):
    with open(html_file, 'r') as f:
        soup = BeautifulSoup(f, 'html.parser')

    for tag in soup.find_all(['script', 'link']):
        if tag.has_attr('src') and not tag['src'].startswith(('http', '//')):
            tag['src'] = f"{base_url}/{tag['src']}"
            with open(html_file, 'w') as f:
                f.write(str(soup))
        elif tag.name == 'link' and tag.has_attr('href') and not tag['href'].startswith(('http', '//')):
            tag['href'] = f"{base_url}/{tag['href']}"
            with open(html_file, 'w') as f:
                f.write(str(soup))
        # Replace "http://" + window.location.host with base_url in text content
        fileText = open(html_file, 'r').read()
                
        with open(html_file, 'w') as f:
            text_node = fileText.replace("\"http://\"+window.location.host", "'" + base_url + "'")
            f.write(text_node)


def process_directory(directory_path, base_url):
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith(".html"):
                html_file_path = os.path.join(root, filename)
                add_base_url(html_file_path, base_url)
                print("**********************************************")
                print("added " + base_url + " to " + html_file_path)
                print("**********************************************")


def run_port(port, my_file_path, newStatus):
    status = get_dir_size(my_file_path)
    if is_port_open(int(port)):
        error(f"Port {port} is already in use.", "Choose a different port.")
    else:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            try:
                if newStatus != status:
                    print(f"Starting server on Port .... {port}")
                    server_command = [sys.executable, '-B' ,  'custom_server.py', port]
                    p = subprocess.Popen(
                        server_command, cwd=os.path.dirname(os.path.realpath(__file__)))

                    # Wait for the subprocess to finish
                    p.wait()
                else:
                    run_port(port, my_file_path, status)

            except Exception as e:
                print(e)

            except KeyboardInterrupt:
                print("Naruto at Rest")
                p.terminate()

            # Kill the subprocess after 1 second
            time.sleep(1)
            p.terminate()


def no_setup():
    print("Error: You need to set up the project... Solution: Use Command -> naruto setup new")

def error(problem, solution):
    print(f"Error: {problem}... Solution: {solution}")

def create_directory(directory):
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception as e:
        error("Failed to create directory", f"Error: {e}")

def main(file_path):
    setupFlag = 0
    newStatus = 0
    file_path = "./"
    while True:
        user_input = input("Welcome to naruto!......Enter a Command : ")
        
        index_html_path = "./flag.html"  # Replace with the actual path of flag.html

        if user_input.lower() == "naruto run":
            if not os.path.exists(index_html_path):
                no_setup()
            else:
                run_port('8000', file_path, 0)
        elif user_input.lower().startswith("naruto run port"):
            if not os.path.exists(index_html_path):
                no_setup()
            else:
                split_ip = user_input.lower().split()
                try:
                    run_port(split_ip[3], file_path, 0)
                except IndexError:
                    run_port("8000", file_path, 0)
        elif user_input.lower().startswith("naruto setup new"):
            try:
                if not os.path.exists(index_html_path):
                
                    print("**************************************************")
                    file = io.open("flag.html", "w", encoding='utf-8')
                    file.write(r'''

<!DOCTYPE html>
<html>
<head>
</head>
<body>

<script>
    // Flag to ensure loadAnotherHTML is called only once
    var loaded = false;
  
    // Function to load another HTML file
    function loadAnotherHTML() {
      if (!loaded) {
        fetch('src/components/app/app.component.html')
          .then(response => response.text())
          .then(htmlContent => {
            // Insert the loaded content into the body
            document.body.innerHTML = htmlContent;
  
            // Extract and execute scripts
            var scriptRegex = /<script\\b[^>]*>([\s\S]*?)<\/script>/gm;
            var matches;
            while ((matches = scriptRegex.exec(htmlContent)) !== null) {
              var scriptCode = matches[1];
              var srcAttribute = matches[0].match(/src=["'](.*?)["']/);
  
              if (srcAttribute) {
                // External script with src attribute
                var scriptSrc = srcAttribute[1];
                loadExternalScript(scriptSrc);
              } else {
                // Inline script
                try {
                  new Function(scriptCode)();
                } catch (error) {
                  console.error('Error executing script:', error);
                }
              }
            }
  
            // Set the flag to true to prevent further calls
            loaded = true;
          })
          .catch(error => {
            console.error('Error loading HTML:', error);
          });
      }
    }
  
    // Function to load an external script
    function loadExternalScript(src) {
      var scriptElement = document.createElement('script');
      scriptElement.src = src;
      document.head.appendChild(scriptElement);
    }
  
    // Call the function to load another HTML file
    loadAnotherHTML();
  </script>

</body>
</html>


                               
                               ''')
                    file = io.open("routing.json", "w", encoding='utf-8')
                    file.write('''
{"routes": [
    {"path": "/", "component": "app/app.component.html"}, 
    {"path": "/tagName", "component": "app/app.component.html"}
    ]
}                               
                               ''')
                    file = io.open("apikey.py", "w", encoding='utf-8')
                    file.write(r'''
import json
def handle_open_ai_key():
    # Handle API requests here
    return "Your Open AI Key Here"
''')
                    # Creating directories
                    directories = [
                        "backend",
                        "backend/app",
                        "backend/test",
                        "src",
                        "src/static",
                        "src/static/css",
                        "src/static/js",
                        "src/static/archive",
                        "src/components/app"
                    ]

                    for directory in directories:
                        try:
                            os.makedirs(os.path.join(file_path, directory), exist_ok=True)
                        except FileExistsError:
                            print(f"Directory {os.path.join(file_path, directory)} already exists. Skipping creation.")
                    file = io.open("backend/app/index.py", "w", encoding='utf-8')
                    file.write('''import json

def handle_api_request(path, payload=None):
    # Handle API requests here
    if path == "/api/data":
        return {'message': 'Hello from API!'}
    else:
        return {'error': 'Endpoint not found'}''')
                    file = io.open("src/static/js/main.js", "w", encoding='utf-8')
                    file.write(r'''

function loadTag(tagName) {
    document.addEventListener("DOMContentLoaded", function () {
        var tagNameTags = document.querySelectorAll(tagName);
        
        tagNameTags.forEach(function (tagNameTag, index) {
            var fileName = '../../../src/components/' + tagName + "/" + tagName + '.component.html'; // Adjust the filename as needed

            // Load content from the corresponding HTML file
            loadContent(fileName, function (response) {
                // Set the content inside the <tagName> tag
                console.log('blloook', response);
                tagNameTag.innerHTML = response;
                executeScriptsInElement(tagNameTag);
            });
        });

        function loadContent(url, callback) {
            // Add a unique query parameter to the URL to prevent caching
            const noCacheUrl = url + (url.includes('?') ? '&' : '?') + 'nocache=' + new Date().getTime();
        
            fetch(noCacheUrl, {
                cache: 'no-store'
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Error loading content from ${noCacheUrl}: ${response.status} ${response.statusText}`);
                }
                return response.text();
            })
            .then(callback)
            .catch(error => console.error('Error loading content:', error));
        }
        
        

        function executeScriptsInElement(element) {
            var scripts = element.getElementsByTagName('script');

            for (var i = 0; i < scripts.length; i++) {
                eval(scripts[i].innerHTML);
            }
        }
    });
}

function callApi(apiName, responseType, method, payload = null, headers = {}) {
    const options = {
        method: method,
        headers: headers
    };

    if (payload) {
        if (typeof payload === 'object') {
            options.body = JSON.stringify(payload);
            options.headers['Content-Type'] = 'application/json';
        } else {
            options.body = payload;
        }
    }

    return fetch(apiName, options)
        .then(response => {
            if (responseType === 'text') {
                return response.text();
            } else if (responseType === 'json') {
                return response.json();
            } else if (responseType === 'blob') {
                return response.blob();
            } else {
                // Handle other response types as needed
                throw new Error('Unsupported response type');
            }
        });
}


function updateHtml() {

    // Sample HTML content with variables
    var getFullHtml = returnHtml()

    // Regular expression to match the content inside ${{}}
    var regex = /\${(.*?)}/g;
    // Extract data from matched patterns
    var matches = getFullHtml.match(regex);

    if (matches) {
        // Loop through extracted variable names
        for (let i = 0; i < matches.length; i++) {
            // Extract the variable name from the match
            var variableName = matches[i].match(/\${(.*?)}/)[1];

            // Check if the variable with the same name is defined
            if (window.hasOwnProperty(variableName)) {
                // Print the value of the variable
                getFullHtml = getFullHtml.replace(matches[i], "<data class='" + /\${(.*?)}/.exec(matches[i])[1] + "'>" + window[variableName] + "</data>");
            } else {
                console.log(`Variable ${variableName} is not defined.`);
            }
        }
    }
    document.getElementsByTagName("html")[0].innerHTML = getFullHtml;
}

function updateVariable(variableName, variableValue) {
    // Sample HTML content with variables
    var getFullHtml = returnHtml();
    // Find the element with the specified class name
    var element = document.querySelectorAll('data.' + variableName);

    // Check if the element exists and has the expected class name
    element.forEach(element => {
        if (element && element.classList.contains(variableName)) {
            // Return the inner HTML of the element
            element.innerHTML = variableValue;
        } else {
            // Return a message indicating that the element was not found
            return 'Element with class "' + variableName + '" not found';
        }
    });
}

function returnHtml() {
    // Sample HTML content with variables
    var getFullHtml = document.getElementsByTagName("html")[0].innerHTML;
    return getFullHtml;
}

document.addEventListener("DOMContentLoaded", function () {
    updateHtml();
});                                                                                          
                               ''')
                    file = io.open("src/components/app/app.component.html", "w", encoding='utf-8')
                    file.write('''
<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.7.1/jquery.min.js"></script>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="../../../src/static/js/main.js"></script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
<link href="../../../src/components/app/app.component.css" rel="stylesheet">
<script src="../../../src/components/app/app.component.js"></script>
<script>
    // Function to dynamically add a stylesheet
    function addStyleSheet(url) {{
      var link = document.createElement('link');
      link.href = url;
      link.rel = 'stylesheet';
      document.head.appendChild(link);
    }}

    // Function to dynamically add a script
    function addScript(url) {{
        console.log(url);
      var script = document.createElement('script');
      script.src = url;
      document.head.appendChild(script);
    }}
      // Add the stylesheet dynamically
    addStyleSheet("../../.." + '/src/components/app/app.component.css?'+'nocache=' + new Date().getTime());
    // Add the script dynamically
    addScript("../../.." + '/src/components/app/app.component.js?'+'nocache=' + new Date().getTime());

</script>
<div class="app_component" id="app_component">
    <div class="text">
        <h1>Welcome to naruto... Hisss!!!</h1>
    </div>
</div>               
 ''')
                    file = io.open("src/components/app/app.component.css", "w", encoding='utf-8')
                    file.write('''
html, body {
    height: 100%; margin: 0; padding: 0;
}

* html #outer {/* ie6 and under only*/
    height:100%;
}

.wrapper {
    min-height: 100%;
    margin: -240px auto 0;
}

.content {padding-top: 240px;}

.footer {
    height: 240px; background: #F16924;
}

.app_component {
    background: #fe5e54;
    color: #fff;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: center;
    height: fit-content;
    width: 100%;
    height: 100%;
}

.app_component h1 {
    font-weight: 100 !important;
}

.logo {
    font-size: 114px;
    text-align: center;
}           
                               ''')
                    file = io.open("src/components/app/app.component.js", "w", encoding='utf-8')
                    file.write('''
                               
                               
                               
/***********************Write all JS above this ******************/
updateHtml()                               
                               ''')
                    file.close()
                    print(f"naruto Project Setup Complete")
                    print("**************************************************")
                    print("Naruto Passes the Internship")
                else:
                    error("Project already exists", "Solution: Log on to that project and run -> naruto run")
            except Exception as e:
                print(e)
                error("This is on us", f'Email us at admin@xanfinity.com\nError: {e}')
        elif user_input.lower() == "exit":
            print("naruto dead")
            break
        elif user_input.lower().startswith("naruto auto create"):
            from openai import OpenAI
            from apikey import handle_open_ai_key
            client = OpenAI(
                # This is the default and can be omitted
                api_key= handle_open_ai_key(),
            )
            split_comp = user_input.lower().split(" ")
            myfile = input("Enter Component Name Eg. (app): ")
            fileName = "src/components/" + myfile + "/" + myfile + ".component.html"
            getPrompt = input("Enter Prompt : ")
            fileName = open(fileName, 'r')
            htmlContent = fileName.read()
            soup = BeautifulSoup(htmlContent, 'html.parser')
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": getPrompt,
                    }
                ],
                model="gpt-3.5-turbo",
            )

            if chat_completion and chat_completion.choices:
                html_content = chat_completion.choices[0].message.content
                decoded_html_content = html.unescape(html_content)
                
                for tag in soup.find_all(text=True):
                    if split_comp[3] in tag:
                        parent = tag.parent
                        new_tag = str(tag).replace("@@" + split_comp[3] + "@@", decoded_html_content)
                        tag.replace_with(BeautifulSoup(new_tag, "html.parser"))

                with open(str(fileName.name), 'w') as NewFileName:
                    NewFileName.write(str(soup))
                    print("**************************************************")
                    print("Replaced " + split_comp[3] + " in component " + str(fileName.name))
                    print("**************************************************")

                        
        elif user_input.lower().startswith("naruto n c"):
            print("**************************************************")
            split_comp = user_input.lower().split(" ")
            try:
                os.mkdir("src/components/" +split_comp[3])
                file = open("src/components/" + split_comp[3] + "/" +split_comp[3]+".component.html", 'w')
                file.write(f'''

<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.7.1/jquery.min.js"></script>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="../../../src/static/js/main.js"></script>
<script src="../../../src/components/{split_comp[3]}/{split_comp[3]}.component.js"></script>
<link href="../../../src/components/{split_comp[3]}/{split_comp[3]}.component.css" rel="stylesheets">
<script>
        // Function to dynamically add a script
        function addScript(url) {{
            console.log(url);
        var script = document.createElement('script');
        script.src = url;
        document.head.appendChild(script);
        }}
        // Add the script dynamically
        addScript("../../.." + '/src/components/{split_comp[3]}/{split_comp[3]}.component.js?'+'nocache=' + new Date().getTime());
    
    </script>
<!-------------HTML BELOW-------------->
<div class="{split_comp[3]}_component" id="{split_comp[3]}_component">
    <div class="text">
        <h1>{split_comp[3]} component Work... Hisss!!!</h1>
    </div>
    <script>
        // Function to dynamically add a stylesheet
        function addStyleSheet(url) {{
        var link = document.createElement('link');
        link.href = url;
        link.rel = 'stylesheet';
        document.head.appendChild(link);
        }}

        // Add the stylesheet dynamically
        addStyleSheet("../../.." + '/src/components/{split_comp[3]}/{split_comp[3]}.component.css?'+'nocache=' + new Date().getTime());
    
    </script>
</div>
''')

                file.close()
                file = open("src/components/" + split_comp[3] + "/" +split_comp[3]+".component.css", 'w')
                file.close()
                file = open("src/components/" + split_comp[3] + "/" +split_comp[3]+".component.js", 'w')
                file.write('''
                
                
                
/***********************Write all JS above this ******************/
updateHtml()
                ''')
                file.close()
                print("Generated New Component " + split_comp[3])
            except Exception as e:
                print(e)
            print("**************************************************")
        elif user_input.lower().startswith("naruto deploy"):
            base_url_input = input("Enter Base URL : ")
            process_directory('src', base_url_input)
        else:
            print("Invalid command. Type 'naruto run' to start the server or 'exit' to quit.")

if __name__ == "__main__":
    main("")

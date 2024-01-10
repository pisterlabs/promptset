import openai


code = """
(ns example
  (:gen-class))

(defn factors [n]
  " Find the proper factors of a number "
  (into (sorted-set)
        (mapcat (fn [x] (if (= x 1) [x] [x (/ n x)]))
                (filter #(zero? (rem n %)) (range 1 (inc (Math/sqrt n)))) )))


(def find-pairs (into #{}
               (for [n (range  2 20000)
                  :let [f (factors n)     ; Factors of n
                        M (apply + f)     ; Sum of factors
                        g (factors M)     ; Factors of sum
                        N (apply + g)]    ; Sum of Factors of sum
                  :when (= n N)           ; (sum(proDivs(N)) = M and sum(propDivs(M)) = N
                  :when (not= M N)]       ; N not-equal M
                 (sorted-set n M))))      ; Found pair

;; Output Results
(doseq [q find-pairs]
  (println q))
"""

def languageDetect(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{prompt} bu kod hangi dile ait?"}
        ],
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6,
        stop=["Human:", "AI:"]
    )

    response = response['choices'][0]['message']['content']

    languages = [
    "Python", "C++", "Java", "C", "JavaScript", "Ruby", "PHP", "Swift", "Go", "Kotlin",
    "C#", "TypeScript", "Objective-C", "Scala", "Perl", "Lua", "Rust", "R", "Groovy", "Dart",
    "Haskell", "Shell", "Elixir", "Clojure", "Julia", "F#", "Visual Basic .NET", "Erlang", "Scheme",
    "PL/SQL", "COBOL", "Fortran", "Ada", "ABAP", "MATLAB", "LabVIEW", "Delphi", "PowerShell",
    "Lisp", "Prolog", "OCaml", "Pascal", "Apex", "MQL4", "MQL5", "D", "SAS", "ALGOL", "Smalltalk",
    "AWK", "Bash", "Forth", "Logo", "Nim", "Crystal", "Elm", "ActionScript", "COBOLScript", "VHDL",
    "Tcl", "Racket", "PostScript", "Haxe", "Verilog", "Simulink", "Processing", "AppleScript", "XQuery",
    "ColdFusion", "QML", "AMPL", "Squirrel", "Curl", "COBOL", "Blockly", "Ballerina", "Idris", "Reason",
    "Scratch", "Solidity", "Stan", "Red", "Purescript", "KRL", "UnrealScript", "JScript", "AngelScript",
    "Pike", "Io", "Magik", "Rascal", "Ring", "Hack", "AutoIt", "Harbour", "Nemerle", "Opa", "Vala", "Vyper",
    "Zig", "Assembly", "BASIC", "VBA", "SQL", "MUMPS", "PL/I", "JCL", "REXX", "CMS-2", "Jython", "IronPython",
    "RubyMotion", "Kivy", "Django", "Flask", "Pyramid", "Rails", "Sinatra", "Hanami", "Laravel", "Symfony",
    "CodeIgniter", "Phalcon", "Yii", "FuelPHP", "CakePHP", "Slim", "Zend Framework", "Express", "Meteor",
    "Sails.js", "Koa", "Hapi", "Spring", "Play", "Struts", "Mojolicious", "Catalyst", "Phoenix", "Ember",
    "Angular", "React", "Vue.js", "Backbone.js", "Aurelia", "Knockout", "Mithril", "Polymer", "Dojo",
    "Xamarin", "Ionic", "React Native", "Flutter", "NativeScript", "PhoneGap", "Cordova", "Titanium",
    "Qt", "GTK", "wxWidgets", "Swing", "JavaFX", "Tkinter", "PyQt", "Kivy", "Java AWT", "SFML",
    "SDL", "Allegro", "OpenGL", "WebGL", 'python']

    detected_language = None
    for language in languages:
        if language in response:
            # Tam eşleşme kontrolü
            if f" {language} " in response or response.startswith(language + " ") or response.endswith(" " + language) or response == language:
                detected_language = language
                break

    if detected_language is not None:
        return detected_language
    else:
        return "Bilinmeyen"



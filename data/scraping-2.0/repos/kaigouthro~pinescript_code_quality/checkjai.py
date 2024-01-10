import json
import os
import openai
import requests
import platform
import openai
import requests
from urllib3 import encode_multipart_formdata


# set these:

# os.environ['OPENAI_API_KEY'] = ""
# os.environ['TV_USERNAME']    = ""
# os.environ['TV_PASSWORD']    = ""


openai.api_key = os.environ["OPENAI_API_KEY"]

URLS = {
    "tvcoins": "https://www.tradingview.com/tvcoins/details/",
    "signin": "https://www.tradingview.com/accounts/signin/",
}


# this can be improved..

PINE_REFERENCE = (
    """

READ THIS FOR REFRERENCE THEN FIX THE BROKEN CODE..

**Python vs pinescript:**


- List Retrieval:
```python
my_list = [1, 2, 3, 4, 5]
element = my_list[2]  # Retrieve element at index 2
```


In PineScript, you would use the `array.get(index)` function to achieve the same result.
```pinescript
my_list = array.from(1, 2, 3, 4, 5)
element = my_list.get(2)  # Retrieve element at index 2
```

- Conditional Code Block:
```python
if condition:
    # Code block executed if condition is true
    ...
elif condition:
    # Code block executed if condition is true
else:
    # Code block executed if condition is false
    ...
```
In PineScript, the syntax is similar, but the `elif` statement is written as `else if`.


# Pine Mini-Reference for more information

## Pine Script™ Operators

> The following operators are available.

| Operator | Description                                                 |
| -------- | ----------------------------------------------------------- |
| +        | Adds two operands                                           |
| -        | Subtracts the second operand from the first                 |
| *        | Multiplies both operands                                    |
| /        | Divides first operand by second                             |
| %        | Modulus Operator and remainder of after an integer division |

## Pine Script Comparison Operaors

| Operator | Checks for                                                                 |
| -------- | -------------------------------------------------------------------------- |
| ==       | if values are equal then condition becomes true.                           |
| !=       | if values are not equal then condition becomes true.                       |
| >        | leftis greater than right, if yes then condition becomes true.             |
| <        | leftis less than right, if yes then condition becomes true.                |
| >=       | leftis greater than or equal to right, if yes then condition becomes true. |
| <=       | leftis less than or equal to right, if yes                                 |

## Pine Script Logical Operaors

| Operator | edsc             |
| -------- | ---------------- |
| and      | logical and      |
| or       | logical or       |
| not      | logical not      |
| ?:       | ternary operator |

## Pine Script Assignment Operaors

| Operator | Description               |
| -------- | ------------------------- |
| =        | assignment                |
| :=       | re-assignment             |
| +=       | addition assignment       |
| -=       | subtraction assignment    |
| *=       | multiplication assignment |
| /=       | division assignment       |
| %=       | modulo assignment         |

# Pine Script™ Keywords

The following keywords are reserved in Pine Script™ and cannot be used as variable names.

| Keyword | Description                                  |
| ------- | -------------------------------------------- |
| import  | Imports a function                           |
| export  | Exports a function                           |
| method  | Creates a method                             |
| type    | Creates a user defined type statement        |
| matrix  | namespace, see matrix                        |
| var     | Creates a variable                           |
| varip   | Creates a variable with intrabar persistence |

## Reserved Keywords

`Catch` ,`Class` ,`Do` ,`Ellipse` ,`In` ,`Is` ,`Polygon` ,`Range` ,`Return` ,`Struct` ,`Text` ,`Throw` ,`Try`

# storage methods (using string as type for example purposes)

| Storage Method | Description                              |
| -------------- | ---------------------------------------- |
| matrix<string> | Matrix are row and column structures     |
| array<string>  | Arrays are single dimensional structures |
| string[]       | Array legacy or declaration shorthand    |
| string         | type name is plain for single item       |


# Lists.

THERE ARE NO LISTS IN PINE SCRIPT!!!
use array instead.

- DO NOT use `[' ... `]` to create a list. Use `array.from( item1, item2,...)` to create an array or `array.new<type>(size, fillwith)` to create an array



# Pine Script™ Built-in Types

Thesse 10 types are built-in for variables, and can appear in storage types

| Types    | Description                                                                                           |
| -------- | ---------------------------------------------------------------------------------                     |
| string   | String of characters, use single quotes, in str.match(input, regex_string), double escape specials    |
| int      | Integer (whole number)                                                                                |
| float    | Float (number with decimal and optional _[Ee]_                                                        |
| bool     | Boolean (true/false)                                                                                  |
| color    | 3 options (color.name, #RRGGBBTT, rgba(r, g, b, a))                                                   |
| line     | line object (line.new(x1, y1, x2, y2, xloc, extend, style, width, color))                             |
| linefill | line fill object (linefill.new(l1, l1, coor))                                                         |
| box      | box object (box.new(left, top, right, bottom, .. etc.. )                                              |
| label    | label object (label.new(x, y, string, xloc, yloc, style, color, .. etc.. )                            |
| table    | table object (table.new(position, columns, rows, bgcolor, bordercolor, .. etc.. )                     |

## Pine Script™ User-defined Types

The following types are available for user-defined types.
A type can be defined with the type keyword.
A type is similar to a class in object-oriented languages,
but methods are declared afterwards and externally

| Type           | Description                                                                                    |
| -------------- | ---------------------------------------------------------------------------------------------- |
| type name      | Create a user-defined type with name, see UDT below                                            |
| name.new(...)  | Create a new object of the type                                                                |
| name.fieldname | calls the stored field item of the type either to reassign, or as an expression's return value |

## Storage Methods:

- Storage methods are used in type declarations.
- TYPE is a built-in type or a user-defined type, which can be any letter or underscore followed by any number of letters, numbers, and underscores.
- The type may have a class name prefix, which is a letter or underscore followed by any number of letters, numbers, and underscores, followed by a period.


### Storage methods can be:

- `matrix<type>`
- `array<type>`
- `type[]`
- `type`

where type is a built-in type or a user-defined type (string, int, float, bool, color, line, linefill, box, label, table).

## User Defined Types (UDTs):

- UDTs are types that are defined in the source code and used in function declarations.
- A UDT field is a name, which can be any letter or underscore followed by any number of letters, numbers, and underscores.

A UDT is like a struct, except that it is not a class in object-oriented languages, and only holds variables.

- Optional annotations:  `@type` tag for the UDT description and `@field` tag for the name and description of individual fields.
- Type declaration: The `export` keyword is optional, the `type` keyword is required, and the name of the UDT is specified.
- Fields: Each field consists of a storage method followed by a field name, and an optional default value on : string, boolean, int, float, color types (when color is hex #RRGGBBTT)
- The field default value can be a number, string, boolean, `na`, or a pine system variable. no arrays, no functions.

example:

```
//@type MyType description
//@field (int) myfieldname - description here
//@field (array<float>) myarrayname  - description goes here
type MyType
    int myfieldname = 0 // this is optional
    array<float> myarrayname // array cannot have default value

//@type MyTypeNesting description
//@field (array<MyType>) mynestedtypearray - description here
//@field (MyType) mynestedtype - description here
//@field (MyTypeNesting) mynestedtype2 - description here
type MyTypeNesting
    array<MyType> mynestedtypearray // can not have default value
    MyType        mynestedtype // also, UDT can not have default value
    MyTypeNesting mynestedtype2 // Can self-nest, but again, no default value.
```

the above is srtict syntax.


## Function Declaration:

- Optional annotations: `@function` tag for the function description, `@param` tag for parameter names and descriptions, and `@return` tag for the description of the return value.
- Function declaration: The `export` keyword is optional for library scripts, the `method` keyword is optional as the second keyword, followed by the function name and parameters in parentheses.
- Parameters: Comma-separated list of parameters, each specifying an optional series or simple keyword, optional storage method, parameter name, and optional default value.
- `=>` after the `(...)`denotes the start of the code block
- Code block: Can be either a single line of code or an indented block of code statements.
- does NOT support `return` keyword.
- does NOT use `function` keyword.
- does NOT support `{` or `}` in the code block.


## Annotations:

Annotations can be used to provide additional information about scripts, UDTs, fields, functions, parameters, and return values.

- For script declarations, the `@description` tag is used to provide a description.
- For UDTs, the `@type` tag is used for the UDT description and the `@field` tag is used for field names and descriptions.
- For functions, the `@function` tag is used for the function description, the `@param` tag is used for parameter names and descriptions, and the `@returns` tag is used for the return value description.
- Annotations must start with `//` followed by `@` and the corresponding tag.
- Annotations can include markdown formatting.

## Comments:

- Comments start with `//`.
- Comments can start a line or follow anything else.
- Comments run from the slashes to the end of the line and end the line.

## Storage Types:

- Storage types can be:
- `type`
- `type []`
- `matrix< type >`
- `array< type >`

- Storage types cannot be:
- `type [] []`
- `matrix< type > []`
- `array< type > []`
- `matrix< type > matrix< type >`
- `array< type > matrix< type >`
- `matrix< type > array< type >`
- `array< type > array< type >`

## Default Values:

- Default values can be a number, string, boolean, `na`, or a system variable.
- Default values cannot be a list of values, a function, or a UDT.

Here is some examples of correct pinescript code syntax:

```pinescript

//@version=5     // MANDATOR ON ALL SCRIPTS

// ONE OF THESE MANDAORY:
// if exporting: library("no_spaces_name")
// if no `strategy.___` calls:  indicator("Name of the indicator")
// if `strategy.___` calls:  strategy("Name of the strategy")

library("demo_pinescript")

// examples of each keyword:

// and
bool a = true and false

// array
int[] b = array.new<int> (input(3,"Size off Array"))

// color (rgb, new, black, blue, gray, green, lime, maroon, navy, olive, orange, purple, red, silver, teal, white, yellow)
// transp = 0-100
color c = color.red
color d = color.new(color.rgb(255,0,0,100), 50)

// export
export myFunction (int a, int b) =>
a + b

// false
bool e = false

// for
for i = 0 to 10
    // do something

// for...in
_array = array.from(1,2,3,4,5)
for [i,item] in _array
    val = _array.get(i)
    _array.set(i, val + 1)

// if
if (a > b)
    // do something
else if (a < b)
    // do something
else
    // do something

// import
import usernamefrom/libraryname/1 as thelib
import usernamefrom/libraryname/1

// matrix
mtx = matrix<int> f = matrix.new<int>(3,3)
mtx.set(f, 0, 0, 1)
a = mtx.get(f, 0, 0)

// method
method myMethod (int a, int b) => a + b
c = 10
d = 20
e = c.myMethod(d)

// not
bool f = not true

// or
bool g = true or false

// switch
switch a
    1 => 1
    2 =>
        2
    =>
        3

// type
type myType
    int myfield = 0
    string myother = "hello"
    imptdlib.imptdtype myimptdFField
    array<int> myarray

var myType myvar = myType.new(1,' yay ',imptdlib.imptdtype.new(1,2,3))
myvar.myfield += 2
myvar.myother := ' yay again'
myvar.myimptdFField.field1 := 5
myvar.myarray.push(myvar.myfield)

// var
var i = 10
var int j = na
j = int(na)

// varip
varip j = 10

// while
while (a < b)
    a += 1

```
"""
)




class TradingView:

    def __init__(self):
        """
        thanks trendoscope =)
        (borrowed somne code from trendoscope's tradingview access project:  https://github.com/trendoscope-algorithms/Tradingview-Access-Management )
        """
        DB = self.load_db()
        print("Getting sessionid from db")
        self.sessionid = DB.get("sessionid", "abcd")
        headers = {"cookie": f"sessionid={self.sessionid}"}
        test = requests.request("GET", URLS["tvcoins"], headers=headers)

        print(test.text)
        print(f"sessionid from db : {self.sessionid}")

        if test.status_code != 200:
            print("session id from db is invalid")
            password = os.environ["TV_PASSWORD"]
            username = os.environ["TV_USERNAME"]
            payload = {"username": username, "password": password, "remember": "on"}
            body, content_type = encode_multipart_formdata(payload)
            user_agent = f"TWAPI/3.0 ({platform.system()}; {platform.version()}; {platform.release()})"
            print(user_agent)
            login_headers = {
                "origin": "https://www.tradingview.com",
                "User-Agent": user_agent,
                "Content-Type": content_type,
                "referer": "https://www.tradingview.com",
            }
            login = requests.post(URLS["signin"], data=body, headers=login_headers)
            cookies = login.cookies.get_dict()
            self.sessionid = cookies["sessionid"]
            DB["sessionid"] = self.sessionid
            self.update_db(DB)
        self.to_fix = []

    def get_sessionid(self):
        DB = self.load_db()
        return DB.get("sessionid", "abcd")

    def load_db(self):
        if os.path.isfile("db.json"):
            with open("db.json", "r") as f:
                return json.load(f)
        return {}

    def update_db(self, db):
        with open("db.json", "w") as f:
            json.dump(db, f)


    def check_pine_server(self, source_code):
        """
        this is toal hax.
        it uses your current chart that was las open
        and does an `update` to the script, without saving it.
        but it returns the information from the compiler if there was an error, or if the script checks out.
        it does not save the script, only updaes i on the chart..
        """
        user_agent = f"TWAPI/3.0 ({platform.system()}; {platform.version()}; {platform.release()})"
        url = "https://pine-facade.tradingview.com/pine-facade/save/new_draft/?user_name={username}&allow_use_existing_draft=true"
        headers = {
            "cookie": f"sessionid={self.sessionid}",
            "origin": "https://www.tradingview.com",
            "User-Agent": user_agent,
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "referer": "https://www.tradingview.com",
        }
        body = {"source": source_code}
        response = requests.post(url, headers=headers, data=body)

        if response.status_code == 200:
            print("\nPOST request successful\n")
        else:
            print("\nPOST request failed\n")

        return json.loads(response.text)
    def check_loop(self):
        DB = self.load_db()
        to_check = DB.get("PINE", [])
        successful_responses = DB.get("Successful", [])
        unsuccessful_responses = DB.get("Failed", [])
        new_unsuccessful_responses = []  # To store the remaining unsuccessful items
        # do a qquick clearing of `PINEE` o removev an successful items from Successful` list
        for i in successful_responses:
            DB["PINE"] = [n for n in DB["PINE"] if n["instruction"] != i["instruction"]]
        for i in to_check:
            response = self.check_pine_server(i["completion"])
            if not response:
                continue
            if response.get("success"):
                successful_responses.append(
                    {"instruction": i['instruction'], "completion": i['completion']}
                )
                DB["PINE"] = [n for n in DB["PINE"] if n["instruction"] != i["instruction"]]
                continue

            error_reason = response.get("reason")
            to_repair = {
                "instruction": i["instruction"],
                "completion": i["completion"],
                "error": error_reason,
                "trycount": 0,
            }
            new_unsuccessful_responses.append(to_repair)  # Add unsuccessful item to the new list

        DB["Failed"] = new_unsuccessful_responses  # Update the failed items with the cleaned list
        DB["Successful"] = successful_responses

        self.update_db(DB)

        if new_unsuccessful_responses:
            self.repair_gpt()

        return successful_responses, unsuccessful_responses
    def repair_gpt(self):
        DB = self.load_db()

        unsuccessful_responses = DB.get("Failed", [])

        if "Unfixable" not in DB:
            DB["Unfixable"] = []
        if "Successful" not in DB:
            DB["Successful"] = []

        while len(unsuccessful_responses) > 0:
            repairable  = DB["Failed"].pop(0)
            instruction = repairable["instruction"]
            code        = repairable["completion"]
            error       = repairable["error"]
            prompt = "\n".join([
                " I am trying to produce a script using pinescript (Pine Script), version 5, and attempting to fulfil this instruction:",
                "```text",
                "the instruction i need my code to fulfill is:",
                instruction,
                "```",
                "# Error note: there are more errors possible, but the compiler only reports the 1st found. This is the error can see from the compiler:",
                f"error: {error}",
                "# Here is the code, it has numerous errors and non-pinescript syntax in it, it requires a total fix, please look a the reference maerial prior o responding..",
                "```",
                code,
                "```",
            ])

            messages = [
                {"role": "system", "content": PINE_REFERENCE},
                {"role": "system", "content": '''
    Ensure:
        - `//@version=5` is on a line by itself in the first line of the code
        - One of these:  `library`, `indicator`, or `strategy` script declaration
        - No python code, no python comments, no python syntax.  Only PineScript syntax and comments are allowed.
        - Function synax should never include `function` preceding the declration.

    Function declaration syntax should be:

        `<function_name>` `(` `<param_type>` `<param_name>` [OPIONAL `=` `<default_value>`] `)` `=>`
            `<function_body>`

        - never use `{` or `}`..
        - never use `return`

    EXAMPLE RESPONSE FORMAT (only include the contents INCLUDING `//BEGINCOMPLETION` to `//ENDCOMPLETION`, nothing outside those comments):

    `

    //BEGINCOMPLETION

    //@version=5
    library("Closest Value")

    // Function to calculate the value closest to the average of an array
    getClosestValue(array<float> myarray) =>
        // Calculate the average of the array
        average = myarray.avg()

        // Initialize variables
        float closestValue = na
        float smallestDifference = na

        // Iterate through the array
        for i = 0 to myarray.size() - 1
            // Calculate the absolute difference between the current value and the average
            difference = math.abs(myarray.get(i) - average)

            // Check if the difference is smaller than the previous smallest difference
            if na(smallestDifference) or difference < smallestDifference
                // Update the closest value and smallest difference
                closestValue := myarray.get(i)
                smallestDifference := difference
            else if difference == smallestDifference
                // If the difference is equal to the previous smallest difference, check if the current value is closer to the average
                if myarray.get(i) < closestValue
                    closestValue := myarray.get(i)
        closestValue

    // Test the function with an example array
    array<float> exampleArray = array.from(1.0, 2.0, 10.0, 20.0)
    closestValue = getClosestValue(exampleArray)

    // Plot the closest value
    plot(closestValue, title="Closest Value", color=color.blue)

    //ENDCOMPLETION

    `

all the code in the `BEGINCOMPLETION` and `ENDCOMPLETION` comments must be the completed code.
write code that is as concise as possible.  Never write notes about what you are doing.  Only write code that is needed to fix the code.

before you respond, analyze the enire script hat it is correct pinescript and not python.
remove and fix all errors in the script.
use single quotes only on strings.
include 1 plotting function of some kind, if a float value is outputted, plot it.
if a bool, use `plot( 0, "true if green", bool_var ? color.green : color.red)`
if a string, or an array, use `label.new(bar_index, close, str.tostring(array_or_value))` as the srt.tostring can convert any primitive as an arrayy or individual to a string.

'''},
    {"role": "assistant", "content": '''okay, i will only write the opening comment, the fixed code, and the closing comment, in this format:

    `
    //BEGINCOMPLETION

    //@version=5
    <script_declaration_type>('<title>')
    <working_code>

    //ENDCOMPLETION
    `
                '''},
                {"role": "user", "content": prompt}
            ]

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k", messages=messages, temperature=1.0
            )

            corrected = self.parse_response(response)
            response  = self.check_pine_server(corrected)

            if not response:
                continue

            success = response.get("success")

            print("\n===================\n")
            print(f"Compiled: {success}")
            print("\n===================\n")


            if success is True:
                # removve the item from DB["PINE"] where i  has the same instruction
                DB["PINE"] = [i for i in DB["PINE"] if i["instruction"] != instruction]
                DB["Successful"].append(
                    {"instruction": instruction, "completion": corrected}
                )
                self.update_db(DB)
                continue

            error_reason = response.get("reason")
            tries = repairable["trycount"]

            repairable["completion"] = corrected
            repairable["error"]      = error_reason
            repairable["trycount"]   = tries + 1

            print(f"\nFAIL\n=====\nInstruction: {instruction}")
            print(f"- Failed :\n {corrected}")
            print(f"- Error :\n  {error_reason}")
            print(f"- Tries :\n  {tries}")
            print("=========\n")


            if tries >= 2:
                print("Too many tries")
                DB["Unfixable"].append(repairable)
                continue

            unsuccessful_responses.append(repairable)
            DB["Failed"] = unsuccessful_responses
            self.update_db(DB)

        return

    @staticmethod
    def parse_response(response):
        responsetext = response["choices"][0]["message"]["content"]
        if "//BEGINCOMPLETION" not in responsetext:
            return responsetext
        if "//ENDCOMPLETION" not in responsetext:
            return responsetext
        return responsetext.split("//BEGINCOMPLETION")[1].split("//ENDCOMPLETION")[0]


TV = TradingView()
TV.check_loop()

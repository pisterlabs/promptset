import openai

openai.api_key = open("API.txt", "r").read()

# note error 1+acc, should be i+acc
PROGRAM = """
let sum xs =
    let rec f (xs, acc) =
        match xs with
        | [] -> acc
        | i::xs' -> f(xs', 1+acc)
    in
    f (xs, 0)
"""

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a useful assistant."},
        {"role": "user", "content": "Act as a senior pair programmer and programming expert."},
        {"role": "user", "content": "Explain code in the following program line-by-line and answer as concisely as possible. Find and explain any errors."},
        {"role": "user", "content": "{}".format(PROGRAM)},
    ]
)

generated_text = response.choices[0].message.content
print(generated_text)

# example answer (best):
#
# This is a function written in OCaml language that takes a list of integers as input and returns the sum of all the integers in the list. Here is the line-by-line explanation along with any errors:
# ```let sum xs =```
# - Define a function called `sum` that takes a list called `xs` as input.
# ```let rec f (xs, acc) =```
# - Define a recursive helper function called `f` that takes a tuple `(xs, acc)` as input, consisting of the remaining list `xs` and the current accumulated sum `acc`.
# ```match xs with```
# - Begin a pattern matching statement on the list `xs`.
# ```| [] -> acc```
# - If the list `xs` is empty, then return the accumulated sum `acc`.
# ```| i::xs' -> f(xs', 1+acc)```
# - If the list `xs` is not empty, bind its head element to variable `i` and the tail of the list to variable `xs'`, then return the recursive call to `f` with updated `(xs', 1+acc)` tuple, adding `i` to the accumulator.
# ```in```
# - Close the inner helper function definition.
# ```f(xs, 0)```
# - Call the `f` function with input tuple `(xs, 0)` where `0` is the initial value of accumulator.
# There are no obvious errors in this code. However, there is a logic error, since the function returns `1+acc` instead of `i+acc`. This can be easily fixed by changing the `1+acc` to `i+acc` in the recursive call.

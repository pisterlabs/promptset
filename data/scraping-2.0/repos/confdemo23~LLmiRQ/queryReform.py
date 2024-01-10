import argparse
import os

import openai

openai.api_key = ""


def main():
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('-b', type=str, required=True, help='Bug reports directory')
    parser.add_argument('-ne', type=str, required=True, help='Non-Existing Classes')
    parser.add_argument('-nb', type=str, required=True, help='Non-Buggy Classes')
    parser.add_argument('-o', type=str, required=True, help='Output directory')
    parser.add_argument('-p', type=str, required=True, help='Project name')
    # parser.add_argument('-a', action='store_true', help='The l flag (optional, no argument)')

    # Parse arguments
    args = parser.parse_args()

    # Access the arguments
    b_arg = args.b
    ne_arg = args.ne
    nb_arg = args.nb
    o_arg = args.o
    p_arg = args.p
    reformulate(b_arg, ne_arg, nb_arg, o_arg, p_arg)


def reformulate(br, ne_, nb_, output_, proj):
    document_dir = br
    o_file = open(output_ + proj + "_re.txt", 'w')
    file_list = os.listdir(ne_)
    sorted_file_list = sorted(file_list)
    expected_extension = '.txt'
    for filename in sorted_file_list:
        if filename.endswith(expected_extension):
            r1 = open(os.path.join(document_dir, filename), "r")
            r2 = open(os.path.join(ne_, filename), "r")
            r3 = open(os.path.join(nb_, filename), "r")
            line = r1.read()
            ne = r2.read()
            nb = r3.read()
            summary = openai.ChatCompletion.create(
                model="gpt-4",
                temperature=0,
                messages=[
                    {"role": "system",
                     "content": f"I previously asked you to analyze a bug report, and you provided some programming entities potentially related to the bug. "
                                f"However, some entities you mentioned either do not exist in the project or are not related to the bug:"
                                f"- Non-existent entities: {ne}"
                                f"- Non-buggy entities: {nb}"
                                f"Can you please re-analyze the bug report from " + proj + "and suggest alternative programming entities that could be related to the bug, based on your knowledge?"
                                                                                           f"Bug report: {line}"
                                                                                           f"Answer(just output the classes name, max number of class is 10):"}
                ]
            )
            s1 = summary.choices[0].message.content
            print("Reformulating query for " + filename.__str__())
            lines = s1.split('\n')  # Split the output into individual lines
            qu = ''
            for li in lines:
                qu += ' '
                qu += li
            o_file.write(filename.__str__().split('.')[0] + '\t' + qu + "\n")
            o_file.flush()
    o_file.close()


if __name__ == '__main__':
    main()

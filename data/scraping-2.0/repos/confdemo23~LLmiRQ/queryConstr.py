import argparse
import os

import openai

# setup open api key
openai.api_key = ""


def main():
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('-b', type=str, required=True, help='Bug reports directory')
    parser.add_argument('-t', type=str, required=True, help='The type of the bug report')
    parser.add_argument('-o', type=str, required=True, help='Output directory')
    parser.add_argument('-p', type=str, required=True, help='Project name')
    # parser.add_argument('-a', action='store_true', help='The l flag (optional, no argument)')

    # Parse arguments
    args = parser.parse_args()

    # Access the arguments
    b_arg = args.b
    t_arg = args.t
    o_arg = args.o
    p_arg = args.p
    construct(b_arg, t_arg, o_arg, p_arg)


def construct(br, t, output_, proj):
    document_dir = br
    o_file = open(output_ + proj + ".txt", 'w')
    file_list = os.listdir(document_dir)
    sorted_file_list = sorted(file_list)
    expected_extension = '.txt'
    if t == 'PE':
        for filename in sorted_file_list:
            if filename.endswith(expected_extension):
                with open(os.path.join(document_dir, filename.__str__()), "r") as f:
                    line = f.read()
                    summary = openai.ChatCompletion.create(
                        model="gpt-4",
                        temperature=0,
                        messages=[
                            {"role": "system",
                             "content": f"Question: Analyze the bug report and construct a query by identifying programming entities (e.g., classes, methods) that may be relevant to the bug’s root cause."
                                        f"Bug report: Deadlock in org.apache.camel.util.DefaultTimeoutMap After running a camel route with a camel Aggregator for a while, "
                                        f"I get a deadlock in  org.apache.camel.util.DefaultTimeoutMap. A full processdump is attached to this bug. I have also tried to recreate this as failing testcase, "
                                        f"but without any luck so far."
                                        f"Answer: DefaultTimeoutMap"
                                        f"Question: Analyze the bug report and construct a query by identifying programming entities (e.g., classes, methods) that may be relevant to the bug’s root cause."
                                        f"Bug report: {line}"
                                        f"Answer(just output the classes and methods name):"}
                        ]
                    )
                s1 = summary.choices[0].message.content
                print("Constructing query for " + filename.__str__())
                lines = s1.split('\n')  # Split the output into individual lines
                qu = ''
                for li in lines:
                    qu += ' '
                    qu += li
                o_file.write(filename.__str__().split('.')[0] + '\t' + qu + "\n")
                o_file.flush()
    elif t == 'ST':
        for filename in sorted_file_list:
            if filename.endswith(expected_extension):
                with open(os.path.join(document_dir, filename.__str__()), "r") as f:
                    line = f.read()
                    summary = openai.ChatCompletion.create(
                        model="gpt-4",
                        temperature=0,
                        messages=[
                            {"role": "system",
                             "content": f"Question: Analyze the provided stack traces and construct a query by identifying programming entities (e.g., classes, methods) relevant to the bug’s root cause."
                                        f"Bug report: ResequencerType.createProcessor could throw NPE as stream config does not get initialized."
                                        f"java.lang.NullPointerException"
                                        f"at org.apache.camel.model.ResequencerType.createStreamResequencer(ResequencerType.java:198)"
                                        f"at org.apache.camel.model.ResequencerType.createProcessor(ResequencerType.java:163)"
                                        f"at org.apache.camel.model.InterceptorRef.createProcessor(InterceptorRef.java:61)"
                                        f"at org.apache.camel.model.ProcessorType.addRoutes(ProcessorType.java:97)"
                                        f"at org.apache.camel.model.RouteType.addRoutes(RouteType.java:210)"
                                        f"at org.apache.camel.impl.DefaultCamelContext.startRouteDefinitions(DefaultCamelContext.java:462)"
                                        f"at org.apache.camel.impl.DefaultCamelContext.doStart(DefaultCamelContext.java:454)"
                                        f"at org.apache.camel.impl.ServiceSupport.start(ServiceSupport.java:47)"
                                        f"at org.apache.camel.ContextTestSupport.startCamelContext(ContextTestSupport.java:108)"
                                        f"at org.apache.camel.ContextTestSupport.setUp(ContextTestSupport.java:81)"
                                        f"at org.apache.camel.processor.ResequencerTest.setUp(ResequencerTest.java:48)"
                                        f"at junit.framework.TestCase.runBare(TestCase.java:128)"
                                        f"at junit.framework.TestResult.runProtected(TestResult.java:124)"
                                        f"..."
                                        f"Answer: ResequencerType ResequencerTest createProcessor"
                                        f"Question: Analyze the provided stack traces and construct a query by identifying programming entities (e.g., classes, methods) relevant to the bug’s root cause."
                                        f"Bug report: {line}"
                                        f"Answer(just output the classes name, no explanation):"}
                        ]
                    )
                s1 = summary.choices[0].message.content
                print("Constructing query for " + filename.__str__())
                lines = s1.split('\n')  # Split the output into individual lines
                qu = ''
                for li in lines:
                    qu += ' '
                    qu += li
                o_file.write(filename.__str__().split('.')[0] + '\t' + qu + "\n")
                o_file.flush()
    elif t == 'NL':
        for filename in sorted_file_list:
            if filename.endswith(expected_extension):
                with open(os.path.join(document_dir, filename.__str__()), "r") as f:
                    line = f.read()
                    summary = openai.ChatCompletion.create(
                        model="gpt-4",
                        temperature=0,
                        messages=[
                            {"role": "system",
                             "content": f"Question: Analyze the bug report and give me all potential programming entities (e.g., classes, methods) relevant to the bug’s root cause based on your knowledge."
                                        f"Bug report: JDBC component doesn't preserve headers."
                                        f"JDBC component doesn't preserve any of the headers that are sent into it."
                                        f"Answer: JdbcProducer JdbcEndpoint JdbcComponent JDBCQueryExecutor ResultSetExtractor"
                                        f"Question: Analyze the bug report from " + proj + "and give me all potential programming entities (e.g., classes, methods) relevant to the bug’s root cause based on your knowledge."
                                                                                           f"Bug report: {line}"
                                                                                           f"Answer(just output the classes name, max 10):"}
                        ]
                    )
                s1 = summary.choices[0].message.content
                print("Constructing query for " + filename.__str__())
                lines = s1.split('\n')  # Split the output into individual lines
                qu = ''
                for li in lines:
                    qu += ' '
                    qu += li
                o_file.write(filename.__str__().split('.')[0] + '\t' + qu + "\n")
                o_file.flush()
    o_file.close()


if __name__ == "__main__":
    main()

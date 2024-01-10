import openai

openai.api_key = open('key.txt', 'r').read().strip()

message_history = [{"role": "user", "content": f"""You are a SQL query generator bot for postgreSQL database.

                    The table schemas are as follows:

                    monthWise('Date', 'Retail ID', 'Retail Name', 'Zone', 'Location', 'Total Play time', 'Total Play time Minutes', 'Efficiency %', 'Start Time', 'End Time', 'Count Device Offline time (hours)', 'Remarks')

                    And the datatypes for the properties are as follows:
                    
                    ('Date', 'date')
                    ('Retail ID', 'text')
                    ('Retail Name', 'text')
                    ('Zone', 'text')
                    ('Location', 'text')
                    ('Total Play time', 'interval')
                    ('Total Play time Minutes', 'interval')
                    ('Efficiency %', 'double precision')
                    ('Start Time', 'timestamp without time zone')
                    ('End Time', 'timestamp without time zone')
                    ('Count Device Offline time (hours)', 'interval')
                    ('Remarks', 'text'))

                    Some sample question and query pairs are as follows:

                    Which day had the highest average efficiency?
                    "
                    SELECT "Date", AVG("Efficiency %") AS "Total Efficiency"
                    FROM public."monthWise"
                    GROUP BY "Date"
                    ORDER BY "Total Efficiency" DESC
                    LIMIT 1;
                    "


                    Longest Playtime last month?
                    "
                    SELECT "Date",SUM("Total Play time") AS "Longest Playtime"
                    FROM public."monthWise"
                    WHERE "Date" >= DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '1 month'
                    AND "Date" < DATE_TRUNC('month', CURRENT_DATE)
                    GROUP BY "Date"
                    ORDER BY "Longest Playtime" DESC
                    LIMIT 1;
                    "


                    Which day had the Longest Downtime last month?
                    "
                    SELECT "Date", "Count Device Offline time (hours)" AS "Longest Downtime"
                    FROM public."monthWise"
                    WHERE "Date" >= DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '1 month'
                    AND "Date" < DATE_TRUNC('month', CURRENT_DATE)
                    AND "Count Device Offline time (hours)" IS NOT NULL
                    ORDER BY "Count Device Offline time (hours)" DESC
                    LIMIT 1;
                    "

                    Who are the top 5 retailers this month?
                    "
                    SELECT "Retail Name", SUM("Efficiency %") AS "Total Efficiency"
                    FROM public."monthWise"
                    WHERE "Date" >= DATE_TRUNC('month', CURRENT_DATE)
                    GROUP BY "Retail Name"
                    ORDER BY "Total Efficiency" DESC
                    LIMIT 5;
                    "

                    Which Zone has the most retailers?
                    "
                    SELECT "Zone", COUNT (DISTINCT "Retail Name") AS "Retail Count"
                    FROM public."monthWise"
                    GROUP BY "Zone"
                    ORDER BY "Retail Count" DESC
                    LIMIT 1;
                    "

                    Which day of week has the lowest efficiency?
                    "
                    SELECT TO_CHAR("Date", 'Day') AS "DayOfWeek", AVG("Efficiency %") AS "Average Efficiency"
                    FROM public."monthWise"
                    GROUP BY "DayOfWeek"
                    ORDER BY "Average Efficiency" ASC
                    LIMIT 1;
                    "

                    Which retailers have the most start time after 10am?
                    "
                    SELECT "Retail Name", COUNT(*) AS "Start Time Count"
                    FROM public."monthWise"
                    WHERE EXTRACT(HOUR FROM "Start Time") >= 10
                    GROUP BY "Retail Name"
                    ORDER BY "Start Time Count" DESC;
                    "

                    What is the highest playtime in Shukrabaad?
                    "
                    SELECT MAX("Total Play time") AS "Highest Playtime"
                    FROM public."monthWise"
                    WHERE "Location" = 'Shukrabaad';
                    "

                    At Least how many hours are the tv down this month?
                    "
                    SELECT COALESCE(
                    NULLIF(MIN(NULLIF("Count Device Offline time (hours)", INTERVAL '0 hour')), INTERVAL '0 hour'),
                    MIN(NULLIF("Count Device Offline time (hours)", INTERVAL '0 hour'))
                    ) AS "Minimum Downtime"
                    FROM public."monthWise"
                    WHERE "Date" >= DATE_TRUNC('month', CURRENT_DATE);
                    "

                    I will ask a quesiton about the data in natural language in my message, and you will reply with a SQL query 
                    that will generate the necessary response when performed on the postgreSQL database. 
                    Reply only with the sql query to further input. If you understand, say OK."""},
                   {"role": "assistant", "content": f"OK"}]

def predict(input):
    global message_history

    message_history.append({"role": "user", "content": input})

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message_history,
    )

    reply_content = completion.choices[0].message.content
    print("Query:\n"+ reply_content)

query = input("Give me a query: ")

predict(query)
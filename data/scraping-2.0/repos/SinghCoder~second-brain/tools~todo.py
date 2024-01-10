import json
import sqlite3
from datetime import datetime

import parsedatetime as pdt
from langchain.tools import tool

todoconn = sqlite3.connect('secondbrain.db')
todoconn.cursor().execute('''CREATE TABLE IF NOT EXISTS todos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    what TEXT,
    due TEXT,
    parsed_when TIMESTAMP,
    completed BOOLEAN
)''')
todoconn.commit()
todoconn.close()

@tool
def add_todo_item(what: str, when: str) -> str:
    '''
    Adds a todo item to the todo list.
    what: What to do?
    when: When to do?
    '''
    try:
        cal = pdt.Calendar()
        datetime_obj, _ = cal.parseDT(when, datetime.now())
        parsed_when = datetime_obj.isoformat(timespec='minutes')

        ## Check if already exists
        todoconn = sqlite3.connect('secondbrain.db')
        cur = todoconn.cursor()
        cur.execute('SELECT * FROM todos WHERE what=?', (what,))
        if cur.fetchone() is not None:
            result = json.dumps({'what': what, 'when': when, 'status': 'already exists'})
            todoconn.commit()
            cur = todoconn.cursor()
            cur.execute('SELECT * FROM todos order by parsed_when asc')
            todos = cur.fetchall()
            with open('store/MyTodos.md', 'w') as f:
                for row in todos:
                    what = row[1]
                    due = row[2]
                    parsed_when = datetime.strptime(row[3], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H:%M')
                    completed = ' ' if row[4] == False else 'x'
                    f.write(f"- [{completed}] {what} by {due} ({parsed_when})\n")
            todoconn.commit()
            return result
        todoconn.commit()

        ## Add to database
        cur = todoconn.cursor()
        values = (what, when, datetime_obj, False)
        cur.execute('INSERT INTO todos (what, due, parsed_when, completed) VALUES (?, ?, ?, ?)', values).close()
        todoconn.commit()

        cur = todoconn.cursor()
        cur.execute('SELECT * FROM todos order by parsed_when asc')
        todos = cur.fetchall()
        with open('store/MyTodos.md', 'w') as f:
            for row in todos:
                what = row[1]
                due = row[2]
                parsed_when = datetime.strptime(row[3], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H:%M')
                completed = ' ' if row[4] == False else 'x'
                f.write(f"- [{completed}] {what} by {due} ({parsed_when})\n")
        todoconn.close()
        return json.dumps({'what': what,'when': f"{when} ({parsed_when})"})

    except Exception as e:
        return json.dumps({'error': str(e)})

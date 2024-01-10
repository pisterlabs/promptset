import sqlite3
import psycopg2
import time
import threading
import json
from Argument import Argument

class database:
    def __init__(self, argument, db_lock):
        self.argument = argument
        self.db_lock = db_lock
        self.start_connection()

    def __del__(self):
        self.stop_connection()

    def start_connection(self):
        if self.argument.read_conf('db','type')=='sqlite':
            db_file_path = self.argument.read_conf('db','sqlite_path')
            self.conn = sqlite3.connect(db_file_path, check_same_thread=False)
            self.cur = self.conn.cursor() 
        elif self.argument.read_conf('db','type')=='postgreSQL':
            db_host = self.argument.read_conf('db','pg_host')
            db_port = self.argument.read_conf('db','pg_port')
            db_name = self.argument.read_conf('db','pg_name')
            db_user = self.argument.read_conf('db','pg_user')
            db_password = self.argument.read_conf('db','pg_password')

            # connect to postgres with default db
            self.conn = psycopg2.connect(host=db_host, port=db_port, dbname='postgres', user=db_user, password=db_password)
            self.conn.autocommit = True
            self.cur = self.conn.cursor()
            db_exist = self.postgres_check_db_exist(db_name)
            if not db_exist:
                self.postgres_create_db(db_name)
            self.conn.close()

            # connect to postgres with target db
            self.conn = psycopg2.connect(host=db_host, port=db_port, dbname=db_name, user=db_user, password=db_password)
            self.conn.autocommit = True
            self.cur = self.conn.cursor()
            if not self.check_postgres_table_exist('user'):
                self.postgres_create_schema()

    def stop_connection(self):
        self.conn.close()

    def deal_sql_request(self, command, params=None) -> list:
        # TODO : 建議使用transaction重新改寫

        if self.argument.read_conf('db','type')=='postgreSQL':
            command = command.replace('?', '%s')

        try:
            self.db_lock.acquire()
            result = None
            if params:
                self.cur.execute(command, params)
            else:
                self.cur.execute(command)
            
            result = self.cur.fetchall()
            self.conn.commit()

        except psycopg2.ProgrammingError as err:
            if err.args[0] == 'no results to fetch': # no result to fetch     
                result = True
            else:
                print('ERR request failed!', command, err)
        except Exception as err:
            print('ERR request failed!', command, err)
        finally:
            self.db_lock.release()
            return result

    def check_postgres_table_exist(self, table_name):
        result = self.deal_sql_request('SELECT EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE table_name = ?)', (table_name,))
        return True if result[0][0] else False

    def postgres_check_db_exist(self, db_name):
        result = self.deal_sql_request('SELECT EXISTS (SELECT 1 FROM pg_database WHERE datname = ?)', (db_name,))
        return True if result[0][0] else False

    def postgres_create_db(self, db_name):
        self.deal_sql_request('CREATE DATABASE ' + db_name)
        return True

    def postgres_create_schema(self):
        self.deal_sql_request(  'CREATE TABLE "chat_session" ( \
                                    "sessionid"	bigserial  NOT NULL, \
                                    "messageid"	INTEGER NOT NULL, \
                                    "time"	bigint NOT NULL, \
                                    "analyze"	TEXT, \
                                    PRIMARY KEY("sessionid") \
                                )'
        )
        self.deal_sql_request(  'CREATE TABLE "keyword" ( \
                                    "id"	bigserial  UNIQUE, \
                                    "enable"	INTEGER NOT NULL DEFAULT 1, \
                                    "keyword"	TEXT NOT NULL UNIQUE, \
                                    "reply"	TEXT NOT NULL, \
                                    "note"	TEXT, \
                                    PRIMARY KEY("id") \
                                )'
        )
        self.deal_sql_request(  'CREATE TABLE "message" ( \
                                    "messageid"	bigserial  NOT NULL, \
                                    "userid" TEXT NOT NULL, \
                                    "time"	bigint NOT NULL, \
                                    "direction"	INTEGER NOT NULL, \
                                    "text"	TEXT NOT NULL, \
                                    PRIMARY KEY("messageid") \
                                )'
        )
        self.deal_sql_request(  'CREATE TABLE "message_reply" ( \
                                    "messageid"	INTEGER NOT NULL, \
                                    "reply_mode"	INTEGER NOT NULL, \
                                    "reply_rule"	INTEGER NOT NULL, \
                                    PRIMARY KEY("messageid") \
                                )'
        )
        self.deal_sql_request(  'CREATE TABLE "story" ( \
                                    "story_id"	bigserial NOT NULL, \
                                    "enable"	INTEGER NOT NULL, \
                                    "name"	TEXT NOT NULL UNIQUE, \
                                    PRIMARY KEY("story_id") \
                                )'
        )
        self.deal_sql_request(  'CREATE TABLE "story_sentence" ( \
                                    "sentence_id" bigserial NOT NULL, \
                                    "story_id"	INTEGER NOT NULL, \
                                    "parent_id"	INTEGER NOT NULL DEFAULT 0, \
                                    "type"	INTEGER NOT NULL, \
                                    "output_or_condiction"	TEXT NOT NULL, \
                                    FOREIGN KEY("story_id") REFERENCES "story"("story_id"), \
                                    PRIMARY KEY("sentence_id") \
                                )'
        )
        self.deal_sql_request(  'CREATE TABLE "user" ( \
                                    "uuid"	TEXT NOT NULL UNIQUE, \
                                    "platform"	TEXT NOT NULL, \
                                    "ban"	INTEGER NOT NULL DEFAULT 0, \
                                    "name"	TEXT, \
                                    "photo"	TEXT, \
                                    "last_update_time"	bigint, \
                                    "tag"    TEXT DEFAULT \'[\"all_user\"]\', \
                                    "tmp"	TEXT NOT NULL DEFAULT \'{}\', \
                                    PRIMARY KEY("uuid") \
                                )'
        )
        self.deal_sql_request(  'CREATE TABLE "openai_usage" ( \
                                    "call_id"	bigserial  UNIQUE, \
                                    "time"	bigint NOT NULL, \
                                    "model"	TEXT NOT NULL, \
                                    "tokens" INTEGER NOT NULL, \
                                    PRIMARY KEY("call_id") \
                                )'
        )

    ### Chat Operation

    def save_chat(self, userId, time ,direction, text): # 0:AI ; 1:Human
        self.deal_sql_request('INSERT INTO message (userid,time,direction,text) VALUES (?, ?, ?, ?)', (userId, time, direction, text))
        result = self.deal_sql_request('SELECT MAX(messageid) FROM message')
        # result = self.deal_sql_request('SELECT last_insert_rowid()')
        return result[0][0]
    
    def save_reply(self, messageId, reply_mode, reply_rule):
        if reply_rule==None: reply_rule='0'
        self.deal_sql_request('INSERT INTO message_reply (messageid,reply_mode,reply_rule) VALUES (?, ?, ?)', (messageId, reply_mode, reply_rule))

    def search_message(self, messageId):
        result = self.deal_sql_request('SELECT text FROM message WHERE messageid=?', (messageId,))
        return result

    def load_chat(self, userId, count=5):
        result = self.deal_sql_request('SELECT direction,text FROM (SELECT time,direction,text FROM message WHERE userid=? ORDER BY time DESC LIMIT ?) AS A ORDER BY time', (userId, count))
        return result

    def load_chat_detail(self, userId, count=5):
        result = self.deal_sql_request('SELECT messageid,time,direction,text FROM (SELECT messageid,time,direction,text FROM message WHERE userid=? ORDER BY time DESC LIMIT ?) AS A ORDER BY time', (userId, count))
        return result

    def load_recent_chat(self, count=100):
        result = self.deal_sql_request('SELECT time,userid,name,direction,text FROM message,"user" WHERE message.userid="user".uuid ORDER BY messageid DESC LIMIT ?', (count,))
        return result

    def load_chat_limited_time(self, userId, count=5, time_offset=180):
        time_limit = int((time.time()-time_offset)*1000)
        result = self.deal_sql_request('SELECT direction,text FROM (SELECT time,direction,text FROM message WHERE userid=? AND time>=? ORDER BY time DESC LIMIT ?) AS A ORDER BY time', (userId, time_limit, count))
        return result

    def load_chat_amount(self):
        result = self.deal_sql_request('SELECT COUNT(*) FROM message')
        return result[0][0]

    def load_chat_amount_each_month(self):
        if self.argument.read_conf('db','type')=='sqlite':
            result = self.deal_sql_request("SELECT strftime('%Y-%m-%d', time / 1000, 'unixepoch') as day, COUNT(*) FROM message GROUP BY day")
        else:
            result = self.deal_sql_request("SELECT to_char(to_timestamp(time / 1000), 'YYYY-MM-DD') as day, COUNT(*) FROM message GROUP BY day")
        result = {r[0]:r[1] for r in result}
        sorted_result = dict(sorted(result.items()))
        return sorted_result

    def load_last_reply_id(self, user_id):
        result = self.deal_sql_request('SELECT messageid FROM message WHERE userid=? AND direction=0 ORDER BY time DESC LIMIT 1', (user_id,))
        if result:
            return result[0][0]
        else:
            return None

    def check_reply_mode(self, messageId):
        result = self.deal_sql_request('SELECT reply_mode,reply_rule FROM message_reply WHERE messageid=?', (messageId,))
        if result:
            return result[0]
        else:
            return None

    ### talk analyze
    
    def load_chat_start_index_group_by_time_gap(self, time_gap=60): # sec
        result = self.deal_sql_request('SELECT messageid,time, (time/1000) as ts FROM message GROUP BY ts-ts%(?)', (time_gap,))
        return result

    def load_chats_by_start_index_limit_time(self, s_index, t_start, duration=60):
        result = self.deal_sql_request('SELECT direction,text FROM message WHERE messageid>=? AND time<?', (s_index, t_start+duration*1000))
        return result

    def add_chat_session(self, messageId, time):
        result = self.deal_sql_request('Insert INTO chat_session(messageid,time) SELECT ?,? WHERE NOT EXISTS (SELECT 1 FROM chat_session WHERE messageid=?)', (messageId, time, messageId))
        return result

    def load_chat_session(self):
        result = self.deal_sql_request('SELECT sessionid, messageid,time,"analyze" FROM chat_session')
        return result

    def load_chat_session_detail(self, chat_session_time_gap=60):
        chat_sessions = self.deal_sql_request('SELECT sessionid, messageid,time,"analyze" FROM chat_session ORDER BY time DESC LIMIT 100')
        for i, chat_session in enumerate(chat_sessions):
            user_id = self.get_userId_by_messageId(chat_session[1])  
            user_name = self.check_user(user_id)[0][3]
            chat_content = self.load_chats_by_start_index_limit_time(chat_session[1], chat_session[2], chat_session_time_gap)
            chat_content = [c[1] for c in chat_content]
            chat_sessions[i] = (chat_session[0], chat_session[1], user_name, chat_session[2], chat_content, chat_session[3])
        return chat_sessions

    def load_chat_session_no_analyze(self):
        result = self.deal_sql_request('SELECT sessionid, messageid,time FROM chat_session WHERE analyze IS NULL')
        return result

    def save_chat_analyze(self, sessionId, analyze):
        result = self.deal_sql_request('UPDATE chat_session SET "analyze"=? WHERE sessionid=?', (analyze, sessionId))
        return result

    ### keyword Operation
    
    def search_keyword(self, search_str):
        result = self.deal_sql_request('SELECT enable,reply,id FROM keyword WHERE instr(?,keyword)>0 ORDER BY length(keyword) DESC', (search_str,))
        for r in result:
            if r[0] == 1:   # keyword enable
                return (r[1],r[2])
        return None

    def load_keyword(self):
        result = self.deal_sql_request('SELECT Id,Enable,keyword,Reply,Note FROM keyword')
        return result

    def add_keyword(self, enable, keyword, reply, note):
        result = self.deal_sql_request('INSERT INTO keyword (enable,keyword,reply,note) VALUES (?,?,?,?)', (str(enable), keyword, reply, note))
        return result

    def delete_keyword(self, keyword_id):
        result = self.deal_sql_request('DELETE FROM keyword WHERE Id=?', (str(keyword_id),))
        return result

    ### story Operation

    def load_all_story(self):
        result = self.deal_sql_request('SELECT story.story_id, enable, sentence_id, output_or_condiction FROM story,story_sentence WHERE story.story_id=story_sentence.story_id AND story_sentence.type=?', (0,))
        return result

    def load_story_name(self):
        result = self.deal_sql_request('SELECT story_id, name FROM story')
        return result

    def load_next_sentence(self, sentence_id):
        result = self.deal_sql_request('SELECT sentence_id FROM story_sentence WHERE parent_id=?', (sentence_id,))
        return result

    def load_sentence(self, sentence_id):
        result = self.deal_sql_request('SELECT sentence_id, type, output_or_condiction FROM story_sentence WHERE sentence_id=?', (sentence_id,))
        return result[0]

    def load_sentences_from_story(self, story_id):
        result = self.deal_sql_request('SELECT sentence_id,parent_id,type,output_or_condiction FROM story_sentence WHERE story_id=?', (story_id,))
        return result

    def add_story_name(self, name):
        result = self.deal_sql_request('INSERT INTO story (enable, name) VALUES (?, ?)', (1, name))
        if result==None: return None
        result = self.deal_sql_request('SELECT MAX(story_id) FROM story')
        # result = self.deal_sql_request('SELECT last_insert_rowid()')
        return result[0][0]

    def add_story_sentence(self, story_id, parent_id, type, output_or_condiction):
        result = self.deal_sql_request('INSERT INTO story_sentence (story_id,parent_id,type,output_or_condiction) VALUES (?, ?, ?, ?)', (story_id, parent_id, type, output_or_condiction))
        if result==None: return None
        result = self.deal_sql_request('SELECT MAX(sentence_id) FROM story_sentence')
        # result = self.deal_sql_request('SELECT last_insert_rowid()')
        return result[0][0]

    def delete_storyname_id(self, story_id):
        result = self.deal_sql_request('DELETE FROM story WHERE story_id=?', (story_id,))
        return result

    def delete_storysentence_id(self, story_id):
        result = self.deal_sql_request('DELETE FROM story_sentence WHERE story_id=?', (story_id,))
        return result

    ### openAI
    def load_openai_usage(self):
        result = self.deal_sql_request('SELECT COUNT(1) FROM message_reply WHERE reply_mode=4')
        return result[0][0]
    
    def add_openai_tokens(self, model, tokens):
        result = self.deal_sql_request('INSERT INTO openai_usage (time,model,tokens) VALUES (?, ?, ?)', (int(time.time()), model, tokens))
        return result

    def load_openai_total_tokens(self):
        result = self.deal_sql_request('SELECT COALESCE(SUM(tokens), 0) FROM openai_usage')
        return result[0][0]

    def load_openai_tokens_detail(self):
        # for each model, calculate the total tokens
        result = self.deal_sql_request('SELECT model, COALESCE(SUM(tokens), 0) FROM openai_usage GROUP BY model')
        return result

    ### user Operation

    def load_user_amount(self):
        result = self.deal_sql_request('SELECT COUNT(DISTINCT userid) FROM message')
        return result[0][0]

    def load_all_user(self):
        result = self.deal_sql_request('SELECT uuid, platform, ban, name, photo, last_update_time, tmp, count(messageid), tag  FROM "user", message WHERE "user".uuid=message.userid GROUP BY uuid')
        return result

    def check_user(self, user_id):
        result = self.deal_sql_request('SELECT uuid,last_update_time,ban,name FROM "user" WHERE uuid=?', (user_id,))
        return result

    def add_new_user(self, user_id, platform, name, photo, last_update_time):
        last_update_time = int(float(last_update_time))
        result = self.deal_sql_request('INSERT INTO "user" (uuid, platform, name, photo, last_update_time) VALUES (?, ?, ?, ?, ?)', (user_id, platform, name, photo, last_update_time))
        return result

    def add_new_user_no_profile(self, user_id, platform):
        result = self.deal_sql_request('INSERT INTO "user" (uuid, platform) VALUES (?, ?)', (user_id, platform,))
        return result

    def update_user_profile(self, user_id, name, photo, last_update_time):
        result = self.deal_sql_request('UPDATE "user" SET name=?, photo=?, last_update_time=? WHERE uuid=?', (name, photo, last_update_time, user_id))
        return result

    def ban_user(self, user_id, ban):
        result = self.deal_sql_request('UPDATE "user" SET ban=? WHERE uuid=?', (ban, user_id,))
        return result

    def load_all_user_extra_data(self, user_id):
        result = self.deal_sql_request('SELECT tmp FROM "user" WHERE uuid=?', (user_id,))
        if not result:
            return None
        result = json.loads(result[0][0])
        return result

    def load_user_extra_data(self, user_id, d_name):
        d = self.load_all_user_extra_data(user_id)
        if d and d_name in d:
            return d[d_name]
        return None

    def add_user_extra_data(self, user_id, d_name, d_value):
        d = self.load_all_user_extra_data(user_id)
        d[d_name] = d_value
        result = self.deal_sql_request('UPDATE "user" SET tmp=? WHERE uuid=?', (json.dumps(d), user_id))
        return result

    def get_userId_by_messageId(self, messageId):
        result = self.deal_sql_request('SELECT userid FROM message WHERE messageid=?', (messageId,))
        return result[0][0]

    def load_all_user_tag(self):
        tmps = self.deal_sql_request('SELECT tag FROM "user"')
        return json.loads(tmps[0][0])

    def load_user_tag(self, user_id):
        tmps = self.deal_sql_request('SELECT tag FROM "user" WHERE uuid=?', (user_id,))
        return json.loads(tmps[0][0])

    def get_user_with_tag(self, tag) -> list:
        result = self.deal_sql_request('SELECT uuid FROM "user" WHERE STRPOS(tag, ?) > 0', (tag,))
        return result[0]

    def add_user_extra_tag(self, user_id, tag):
        tags = self.load_user_tag(user_id)
        tags.append(tag)
        tags = json.dumps(tags)
        result = self.deal_sql_request('UPDATE "user" SET tag=? WHERE uuid=?', (tags, user_id))
        return result


    ### System Logs

    def load_system_logs(self):
        # logs = [{'time':'2020-01-01','status':'success' ,'text':'test'}]
        logs = []
        with open('data/system_warn.log', 'r') as f:
            texts = f.readlines()[::-1]
            for text_row in texts:
                if text_row.startswith('*'):
                    logs.append({'time':'','status':'' ,'text':text_row[1:]})

        if len(logs)==0:
            logs = [{'time':'','status':'success' ,'text':'all good'}]
        elif len(logs)>10:
            logs = logs[:10]

        return logs

if __name__ == '__main__':
    db = database(Argument(),threading.Lock())
    data = db.load_chat_amount_each_month()
    print(data)
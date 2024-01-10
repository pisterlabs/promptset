from flask import (Blueprint, render_template,request,redirect,url_for,session,flash)
from flask_login import login_required, current_user
from twilio.twiml import re
from twilio.twiml.messaging_response import MessagingResponse
from twilio.twiml.voice_response import VoiceResponse
from werkzeug.security import (check_password_hash, generate_password_hash)
import openai
import threading
from datetime import datetime, timedelta
from .utils import *
from .models import User,Settings,Idea
from time import sleep
import json
import pytz
import random

tz = timezone(timedelta(hours=2))

with open('/etc/personaldb_config.json') as config_file:
    config = json.load(config_file)

openai.api_key = config.get('OPENAI_API_KEY')

routes = Blueprint('routes',__name__,template_folder='templates')

@routes.route("/<item_pk>", methods=['POST','GET'])
@routes.route("/", methods=['POST','GET'])
def home(item_pk=None):

    user = session.get('user',default="")
    # Left this here for reference for future migrations
    #ideas = Idea.find().all()
    #for i in ideas:
    #    i_dict = i.dict()
    #    if not hasattr(i_dict,'time'):
    #        i.time = int(round(datetime.now().timestamp()))
    #        i.save()



    return render_template('home.html',
                           session=session,
                           user=user,
                           page="home"
                           )

    
@routes.route("/settings",methods=['POST','GET'])
def settings():
    user = session.get('user',default="")
    if not user:
        flash('login required')
        return redirect(url_for('routes.home'))

    
    user = User.find(User.pk == user).first().dict()
    all_timezones = pytz.all_timezones
    cur_timezone = user['timezone']

    print("User settings",user['settings'])
    if request.method == 'POST':
        if "new_phone" in request.form:
            new_phone = request.form.get('new_phone')
            session['phone'] = new_phone
            session['todo'] = "change_phone"
            return redirect(url_for('auth.token'))

        elif "change-password" in request.form:
            new_pass = request.form.get('password')
            u = User.find(User.pk == user['pk']).first()
            u.password = generate_password_hash(str(new_pass),method='sha256')
            u.save()
            flash("Password changed!")
        elif "idea-stream" in request.form:
            ans = request.form.get('idea-stream')
            if user['settings']['idea_stream_public'] == "true":
                user['settings']['idea_stream_public'] = "false" 
                u = User.find(User.pk == user['pk']).first()
                u.settings.idea_stream_public = "false"
                u.save()
                flash('Ideas are now private')
            else:
                user['settings']['idea_stream_public'] = "true"
                u = User.find(User.pk == user['pk']).first()
                u.settings.idea_stream_public = "true"
                u.save()
                flash('Ideas are now public')
        elif "user_tz" in request.form:
            new_timezone = request.form.get('user_tz')
            user = User.find(User.pk == user['pk']).first()
            user.timezone = new_timezone
            user.save()
            flash("Timezone changed")
        return redirect(url_for('routes.settings'))

        
    
    return render_template('settings.html',
                           user=user,
                           timezones=all_timezones,
                           cur_timezone=cur_timezone,
                           page="settings"
                           )

@routes.route("/ideas",methods=['POST','GET'])
def ideas():
    user = session.get('user',default="")
    if not user:
        flash('login required')
        return redirect(url_for('routes.home'))

    if request.method == 'POST':
        msg = request.form['message']
        save_idea(user, msg)
        return redirect(url_for('routes.ideas'))

    ideas = user_all_ideas(user)
    return render_template('ideas.html',
                           user=user,
                           page="ideas",
                           ideas=ideas
                           )


@routes.route("/reminders",methods=['POST','GET'])
@routes.route("/reminders<item_pk>",methods=['POST','GET'])
def reminders(item_pk=None):
    user = session.get('user',default="")
    if not user:
        flash('login required')
        return redirect(url_for('routes.home'))

    if request.method == 'POST':
        if "reminder-new" in request.form:  
            msg = request.form['message']
            save_reminder_text(user, msg)
            return redirect(url_for('routes.reminders'))
        try:
            reminder = Reminder.find(Reminder.pk == item_pk).first()
        except:
            flash("couldn't find the reminder")
            return
        if "reminder-reocc" in request.form:
            if reminder.reoccurring == "true":
                reminder.reoccurring = "false"
            else:
                reminder.reoccurring = "true"
            reminder.save()
        elif "reminder-method" in request.form:
            if reminder.remind_method == "call":
                reminder.remind_method = "text"
            else:
                reminder.remind_method = "call"
            reminder.save()
           
        return redirect(url_for('routes.reminders'))
    return render_template('reminders.html',
                           user=user,
                           reminders=user_all_reminders(user),
                           page="reminders"
                           )

@routes.route("/quotes",methods=['POST','GET'])
def quotes():
    user = session.get('user',default="")
    if not user:
        flash('login required')
        return redirect(url_for('routes.home'))
    if request.method == 'POST':
        msg = request.form['quote']
        save_quote(user, msg)
        return redirect(url_for('routes.quotes'))
    return render_template('quotes.html',
                           user=user,
                           quotes=user_all_quotes(user),
                           page="quotes"
                           )

@routes.route("/links",methods=['POST','GET'])
def links():
    user = session.get('user',default="")
    if not user:
        flash('login required')
        return redirect(url_for('routes.home'))
    if request.method == 'POST':
        link = request.form['link']
        save_link(user, link)
        return redirect(url_for('routes.links'))
    return render_template('links.html',
                           user=user,
                           links=user_all_links(user),
                           page="links"
                           )

@routes.route("/notes",methods=['POST','GET'])
def notes():
    user = session.get('user',default="")
    if not user:
        flash('login required')
        return redirect(url_for('routes.home'))
    if request.method == 'POST':
        bagname = request.form['bag-name']
        save_notebag(user, bagname)
        flash("note bag created")
        return redirect(url_for('routes.notes'))
    return render_template('notes.html',
                           user=user,
                           notebags=user_all_notebags(user),
                           page="notes"
                           )

@routes.route("/feed",methods=['POST','GET'])
def feed():
    user = session.get('user',default="")
    if not user:
        flash('login required')
        return redirect(url_for('routes.home'))

    if request.method == 'POST':
        if "img" not in request.files:
            img = None
        else:
            img = request.files['img']
        url = save_post(user,img)
        if url is not None:
            flash("post uploaded")
        else:
            flash("failed to upload")
        sleep(2)
        posts = user_all_posts(user)
        return render_template('feed.html',
                           user=user,
                           posts=posts,
                           page="feed"
                           )


    posts = user_all_posts(user)
    return render_template('feed.html',
                           user=user,
                           posts=posts,
                           page="feed"
                           )

# -------------------------- EDITING/SAVING --------------------------
# --------------------------------------------------------------------

@routes.route("/move-<note_pk>-to-<bag_name>",methods=['POST'])
def move_to(note_pk,bag_name):
    flash("here")
    user = session.get('user',default="")
    if not user:
        flash('login required')
        return redirect(url_for('routes.home'))

    user = User.find(User.pk == user).first().dict()
      
    if move_note(user['pk'],note_pk,bag_name):
        flash("note moved")
    else:
        flash("error moving the note")
    return redirect(url_for('routes.home'))


@routes.route("/note-to-<bag_name>",methods=['POST'])
def save_note_to_bag(bag_name):
    user = session.get('user',default="")
    if not user:
        flash('login required')
        return redirect(url_for('routes.home'))

    user = User.find(User.pk == user).first().dict()

    message = request.form.get('note')
    if message:
        if save_note(user['pk'], bag_name, message):
            flash("note saved!")
        else:
            flash("error saving the note")
    else:
        flash("note can't be empty")
    return redirect(url_for('routes.notes'))
 

@routes.route("/edit-idea-<pk>", methods=['POST','GET'])
def edit_idea(pk):
    user = session.get('user',default="")
    if not user:
        flash('login required')
        return redirect(url_for('routes.home'))

    idea = Idea.find(Idea.pk == pk).first()
    cur_idea = idea.dict()
    if request.method == 'POST':
        new_idea = request.form.get('idea')
        if new_idea == "":
            Idea.delete(pk)
            flash("Idea deleted")
        else:
            idea.message = new_idea
            idea.save()
            flash("Idea edited")
        return redirect(url_for('routes.ideas'))

    return render_template('edit_idea.html',
                           user=user,
                           page="ideas",
                           cur_idea=cur_idea)

@routes.route("/edit-quote-<pk>", methods=['POST','GET'])
def edit_quote(pk):
    user = session.get('user',default="")
    if not user:
        flash('login required')
        return redirect(url_for('routes.home'))

    quote = Quote.find(Quote.pk == pk).first()
    cur_quote = quote.dict()
    if request.method == 'POST':
        new_quote = request.form.get('quote')
        if new_quote == "":
            Idea.delete(pk)
            flash("Quote deleted")
        else:
            quote.quote = new_quote 
            quote.save()
            flash("Quote edited")
        return redirect(url_for('routes.quotes'))

    return render_template('edit_quote.html',
                           user=user,
                           page="quotes",
                           cur_quote=cur_quote)



@routes.route("/edit-reminder-<pk>", methods=['POST','GET'])
def edit_reminder(pk):
    user = session.get('user',default="")
    if not user:
        flash("login required")
        return redirect(url_for('routes.home'))

    try: 
        reminder = Reminder.find(Reminder.pk == pk).first()
    except NotFoundError:
        flash("reminder not found")
        return redirect(url_for('routes.reminders'))

    rem_message_str = reminder.dict()['message']

    if request.method == 'POST':
        if 'time' in request.form:
            time = request.form.get('time')
            epoch_time = to_utc_epoch(user,time)
            if not epoch_time:
                flash("Time format not right. Time formatted: 'day/month/year_last_two_nums hour:minute'")
                return redirect(f"/edit-reminder-{pk}")
            reminder.time = epoch_time
            reminder.save()
            flash("Time changed")
        elif 'msg' in request.form:
            msg = request.form.get('msg')
            if msg == "":
                Reminder.delete(pk)
                flash("Reminder Deleted")
            else: 
                if reminder.message == msg:
                    flash("No change")
                else:
                    reminder.message = msg
                    reminder.save()
                    flash("Message changed")
        return redirect(url_for('routes.reminders'))
        
    return render_template('edit_reminder.html',
                           user=user,
                           message=rem_message_str,
                           reminder_pk=pk,
                           page="reminders",
                           reocc=reminder.reoccurring,
                           method=reminder.remind_method
                           )


@routes.route("/edit-link-<link>", methods=['POST','GET'])
def edit_link(link):
    user = session.get('user',default="")
    if not user:
        flash('login required')
        return redirect(url_for('routes.home'))
    if request.method == 'POST':
        new_link = request.form.get('link')
        user = User.find(User.pk == user).first()
        user.links.remove(link)
        if new_link == "":
            flash("Link deleted")
        else:
            user.links.append(new_link)
            flash("Link edited")

        user.save()
        return redirect(url_for('routes.links'))
    return render_template('edit_link.html',
                           user=user,
                           page="links",
                           link=link)


# -------------------------------------------------------------------------
#                                 DELETING 
# -------------------------------------------------------------------------

@routes.route("/delete-item/<item_type>/<item_pk>")
def delete_item(item_type, item_pk):
    user = session.get('user',default="")
    if not user:
        flash("Login required")
        return redirect(url_for('routes.home'))


    if item_type == "idea":
        Idea.delete(item_pk)
        flash("Idea deleted")
        return redirect(url_for('routes.ideas'))
    elif item_type == "reminder":
        Reminder.delete(item_pk)
        flash("Reminder deleted")
        return redirect(url_for('routes.reminders'))
    elif item_type == "quote":
        user = User.find(User.pk == user).first()
        for q in user.quotes:
            if q.pk == item_pk:
                user.quotes.remove(q)
                user.save()
                flash("Quote deleted")
        return redirect(url_for('routes.quotes'))
    elif item_type == "note":
        if delete_note(user,item_pk):
            flash("Note deleted")
        else:
            flash("couldn't delete note")
        return redirect(url_for('routes.notes'))
    elif item_type == "notebag":
        if delete_notebag(user,item_pk):
            flash("Notebag deleted")
        else:
            flash("Couldn't delete notebag")
            return redirect(url_for('routes.notes'))
    elif item_type == "link":
        user = User.find(User.pk == user).first()
        user.links.remove(item_pk)
        user.save()
        flash("Link deleted")
        return redirect(url_for('routes.links'))
    elif item_type == "post":
        delete_post(item_pk)
        return redirect(url_for("routes.feed"))
    return redirect(url_for('routes.home'))


@routes.route("/call-webhook", methods=['POST','GET'])
def call_webhook():
    response = VoiceResponse()
    phn = request.values.get('From')
    try:
        user = User.find(User.phone == phn).first().dict()
    except:
        user = dict()
        pass
    if user['username'] == 'r':
        usr = "rasmus"
    else:
        usr = user['username']

    response.say(f"hey,{user['username']}")
    response.record()
    response.hangup()
    # only not none if in last minute 
    #text_rec = latest_recording_text(user['pk'])
    #text(phn,text_rec)
    #print(phn)
    #print(text_rec)
    threading.Thread(target=process_speech, args=(user['pk'], phn)).start()
    return str(response)


@routes.route("/sms-webhook",methods=['POST'])
def sms_webhook():

    phone = request.values.get('From',None)
    body = request.values.get('Body',None)
    if body == None or phone == None: 
        print("NO MESSAGE OR PHONE")
        return "no message or phone",404
    try:
        user = User.find(User.phone == phone).first()
    except:
        print("USER NOT FOUND")
        return "user not found",404
    message = ""
    
    if body.startswith("h"):
        if body[2] == "r":
            message = '''"r [body]" -> add reminder,
                        [body] must include a message and date 
                        formatted like "12/2/2054 23:00" somewhere.
                        e.g. "r groceries 12/2/2054 12:00".
                        
                        "all r" -> get all reminders.
            '''
        if body[2] == "i":
            message = '''"i [body]" -> add idea,
                        [body] has the idea.
                        e.g. "i bake more" adds a "bake more" idea.

                        "all i with [body]" -> get all ideas, 
                        that have the keyword [body].
                        E.g. "all i with bake" returns every idea 
                        where you used the word "bake".
            '''
        if body[2] == "t":
            message = '''"t [body]" -> adds a new timer,
                        [body] must include a minute value.
                        E.g. "t 20min" will add timer for 20 minutes.

                        "t stop" -> stops the current timer.
            '''
        else:
            message = '''
                    "r [body]"->add reminder
                    "all r"   ->get all reminders
                    "i [body]"->save idea
                    "all i with [body]" -> all ideas with [body]
                    "t [body]"->start timer
                    "t stop"  ->stop timer

                    Use "h [x]" to get more info about x->(r,i or t).

                      
                    "
                    '''
    elif body.startswith("all r"):
        all_reminders = user_all_reminders(str(user.pk))
        if not all_reminders:
            return
        else:
            for r in all_reminders:
                message += str(r['time']) + " " + r['message'] + "\n"
    elif body.startswith("all i with"):
        key = body[11:]
        user_pk = str(user.pk)
        ideas_obj = Idea.find((Idea.user == user_pk) and 
                          (Idea.message % key)).all()
        ideas = format_ideas(ideas_obj)
        if ideas:
            for i in ideas:
                message += "- "+i['message']+"\n"
        else:
            return

    elif body.startswith("r "):
        t = re.search(r"\d+\/\d+\/\d+",body[2:])
        t2 = re.search(r"\d+\:\d+",body[2:])
        if t == None or t2 == None:
            message = 'Error. Date format not right.'
        else:
            time = t.group() + " " + t2.group()
            msg = ""
            first_add = True
            for x in body[2:].split():
                if x != t.group() and x != t2.group():
                    if first_add:
                        msg += x
                        first_add = False
                    else:
                        msg += " "+x
            if save_reminder(str(user.pk),msg,time):
                return
            else:
                message = "Error. Could not save the reminder"
    elif body.startswith("i "):
        if save_idea(str(user.pk),body[2:]):
            return
        else:
            message = "Error. Could not save the idea"

    elif body.startswith("t "):
        found = re.search("\d+",body[2:])
        stop = re.search("stop",body[2:])
        if stop != None:
            stop_timer(str(user.pk))
            message = "Timer stopped"
        else:  
            if found == None:
                message = "Error. Either give time in minutes or write stop"
            else:
                minutes = int(found.group())
                if start_timer(str(user.pk),minutes):
                    message = f"Timer for {minutes}min started"
                else:
                    message = f'Error. Another timer already going. Use "t stop" to stop it'
    elif body.startswith("rai "):
        to_parse = turn_text_to_reminder_format(body[4:])



        if not correct_reminder_format(to_parse):
            message = 'Error. Date format not right.'
        else:
            t = re.search(r"\d+\/\d+\/\d+",to_parse)
            t2 = re.search(r"\d+\:\d+",to_parse)
            time = t.group() + " " + t2.group()
            msg = ""
            first_add = True
            for x in to_parse.split():
                if x != t.group() and x != t2.group():
                    if first_add:
                        msg += x
                        first_add = False
                    else:
                        msg += " "+x
            if save_reminder(str(user.pk),msg,time):
                message = f"Reminder '{to_parse}' saved."
            else:
                message = "Error. Could not save the reminder"
    elif body.startswith("c "):
        response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You turn the given text into exact same text but with all the letters that are incorrectly 'a' instead of 'ä' or 'o' instead of 'ö', you replace them with the correct ones according to context and finnish language. You don't change the text or words in a text anyway other than replacing 'a' or 'o' letters if necessary. Keep letter capitalization as it is in the original text. Output only the text."},
            {"role": "user", "content": "Tanaan saan pitaisi parantua, joten voimme menna ulkoilemaan metsaan ja kerata sienia seka marjoja ampariin."},
            {"role": "assistant", "content": "Tänään sään pitäisi parantua, joten voimme mennä ulkoilemaan metsään ja kerätä sieniä sekä marjoja ämpäriin."},
            {"role": "user", "content": "ma en ymmarra, miks sa et parjaa tan homman kanssa."},
            {"role": "assistant", "content": "mä en ymmärrä, miks sä et pärjää tän homman kanssa."},
            {"role": "user","content": f"{body[2:]}"},
            ]
        )
        message = response["choices"][0]["message"]["content"]

    elif body.startswith("translate "):
        response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a bot that translates the given text into another language. You are given a text that in one language and if the input language is not given - you should guess it and note about it in the output. If the output language isn't given, you should translate the text into finnish language."},
            {"role": "user", "content": "translate 'wohnzimmer' to english"},
            {"role": "assistant", "content": "living room"},
            {"role": "user","content": f"{body[6:]}"}
            ]
        )
        message = response["choices"][0]["message"]["content"]


    else:
        message = 'Wrong keyword. Type "h" for help.'

    resp = MessagingResponse()

    resp.message(message)
    return str(resp)


@routes.route("/<username>-ideas",methods=['POST','GET'])
def idea_stream(username):
    
    try:
        user = User.find(User.username == username).first()
    except NotFoundError:
        return 'no user found'
    if user.settings.idea_stream_public == "false":
        return "user's idea stream is off"
    ideas = user_all_ideas(user.pk)
    return render_template('idea_stream.html',username=username,ideas=ideas)



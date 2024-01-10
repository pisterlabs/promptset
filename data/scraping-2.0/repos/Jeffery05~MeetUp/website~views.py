from flask import Blueprint, render_template, request, flash, jsonify
from flask_login import login_required, current_user
from . import db
from .models import Meetup, User 
import json
from datetime import datetime
import os
import cohere


key = os.environ.get('GOOGLE_MAPS_API_KEY', 'default_value')
cohere_key = os.environ.get('COHERE_API_KEY', 'default_value')
views = Blueprint('views', __name__) # define blueprint for our application


@views.route('/overview', methods=['GET', 'POST']) #decorator: whenever you go to the /overview URL, whatever in overview() will run
@login_required
def overview():
    return render_template("overview.html", user=current_user)

@views.route('/ideas', methods=['GET', 'POST']) #decorator: whenever you go to the /overview URL, whatever in overview() will run
@login_required
def ideas():
    co = cohere.Client(cohere_key)
    response=""
    titles=[]
    explanation=[]
    result_string=""
    if request.method == 'POST':
        duration = request.form.get('duration')
        party_size = request.form.get('size')
        location = request.form.get('location')
        description = request.form.get('description')
        response = co.generate(prompt='Give me ideas for activities under ' + duration + ' hours to do in ' + location + ' for a group of ' + party_size + ' people with the following features: ' + description, model='command-light', temperature=1.3)
        result_string = response.generations[0].text

        # Find the index of "1."
        index_of_1 = result_string.find("1.")
        index_of_2 = result_string.find("2.")
        index_of_3 = result_string.find("3.")
        index_of_4 = result_string.find("4.")

        if index_of_1 != -1 and index_of_2 != -1:
            # Find the index of ":"
            index_of_colon = result_string.find(":", index_of_1)
            if index_of_colon != -1:
                # Extract the substring starting from "1." to ":"
                titles.append(result_string[index_of_1:index_of_colon].strip())
                explanation.append(result_string[index_of_colon + 1:index_of_2].strip())
        if index_of_2 != -1 and index_of_3 != -1:
            # Find the index of ":"
            index_of_colon = result_string.find(":", index_of_2)
            if index_of_colon != -1:
                # Extract the substring starting from "2." to ":"
                titles.append(result_string[index_of_2:index_of_colon].strip())
                explanation.append(result_string[index_of_colon + 1:index_of_3].strip())
        if index_of_3 != -1 and index_of_4 != -1:
            # Find the index of ":"
            index_of_colon = result_string.find(":", index_of_3)
            if index_of_colon != -1:
                # Extract the substring starting from "2." to ":"
                titles.append(result_string[index_of_3:index_of_colon].strip())
                explanation.append(result_string[index_of_colon + 1:index_of_4].strip())
    return render_template("ideas.html", user=current_user, titles=titles, explanation=explanation, result_string=result_string)


            

@views.route('/create', methods=['GET', 'POST']) 
@login_required
def create():
    if request.method == 'POST': # if the user submits the form
        no_error = True
        
        meetup_date = request.form.get('date')
        date = datetime.strptime(meetup_date, '%Y-%m-%dT%H:%M')  
        meetup_end = request.form.get('endDate')
        date_end = datetime.strptime(meetup_end, '%Y-%m-%dT%H:%M')  
        title = request.form.get('title')
        fullAddress = request.form.get('fullAddress')
        location = request.form.get("location")
        locationCommonName = request.form.get("locationCommonName")
        lat = request.form.get("latitude")
        lng = request.form.get("longitude")
        if lat == "" or lng == "":
            flash('\"' + location + '\" is not a valid address. Please try again using autocomplete.', category='error')
            no_error = False
        description = request.form.get('description')
        invitations = request.form.get('invitations') 
        print("location: " + location)
        
        invitations = invitations.strip() # remove the spaces before and after the invitiations
        invitations = invitations.lower() # ensure the email inivitations are all lowercase
        print("invitations:" + str (invitations))  
        if invitations.find(current_user.email) == -1: # if the current user's email isn't included in the invite list, add them
            invitations = current_user.email + " " + invitations
        invitationsSpaced = " " + invitations + " " # pad the invite list with a space at the beginning and end
        new_meetup = Meetup(date_meetup=date, date_end = date_end, title=title, location=location, fullAddress=fullAddress, locationCommonName=locationCommonName, lat=lat, lng=lng, description=description, invitations=invitationsSpaced, confirmed = '', declined = '', owner=current_user.id, owner_firstname=current_user.first_name)
        
        # check if invites are registered, if so create a many-to=many relationship
        attendees = invitations.split(' ')
        current_user.meetups.append(new_meetup) # ensure the owner (current user) has the first many-to-many relationship
        for person in attendees:
            print("person = " + person)
            if person != current_user.email:
                user = User.query.filter_by(email=person).first()
                if user: # if there is a registered user with that email
                    user.meetups.append(new_meetup) # creating a many to many relationship
                else:
                    flash('There is no account associated with \"' + person + '\". Please ensure the email invitations are in the correct format.', category='error')
                    no_error = False
        if no_error:
            db.session.add(new_meetup)
            db.session.commit()
            flash('Meetup added!', category='success')
        else:
            flash('There was an error creating this meetup. Please try again.', category='error')
    return render_template("create.html", user=current_user, key=key)

@views.route('/view_meetups', methods=['GET', 'POST']) #decorator: whenever you go to the / URL, whatever in hom() will run
@login_required
def view_meetups():
    invites = "" # create a string of people who were invited but not confirmed
    confirmation = "" # create a string of people who have confirmed
    confirmSearch = " " + str(current_user.id) + " "
    firstInv = True
    firstConf = True
    meetupList = []
    for meetup in current_user.meetups: # go through all the current user's meetups
        if meetup.confirmed.find(confirmSearch) == -1:
            meetupList.append({'id': meetup.id, 'lat': meetup.lat, 'lng': meetup.lng, 'locationName': meetup.locationCommonName, 'address': meetup.fullAddress})
        invites = invites + "   " # add 3 spaces to show a new meetup
        confirmation = confirmation + "   "
        firstInv = True
        firstConf = True
        for user in meetup.user: # cycle through the users with a relationship with that meetup
            userSearch = " " + str(user.id) + " " 
            if meetup.confirmed.find(userSearch)  != -1: # if the user has confirmed
                if firstConf:   # if its the first confirmation for the meetup. add their name
                    confirmation  = confirmation + user.first_name
                    firstConf = False
                else: # if its not the first confirmation, add a comma
                    confirmation  = confirmation + ", " + user.first_name
            else: # if the user has not confirmed yet, add them to the invite list
                if firstInv: 
                    invites  = invites + user.first_name
                    firstInv = False
                else:
                    invites  = invites + ", " + user.first_name
    inviteList = invites.split("   ") # create an list of meetups, inside of which are the people invitied to it
    confirmationList = confirmation.split("   ") # create an list of meetups, inside of which are the people who confirmed to it
    
    return render_template("view_meetups.html", key=key, user=current_user, inviteList = inviteList, confirmSearch = confirmSearch, confirmationList = confirmationList, meetupList = meetupList)

@views.route('/confirmed', methods=['GET', 'POST'])
@login_required
def confirmed(): # same principal as view_meetups with the list of invites + confirmed for each meetup
    invites = ""
    confirmation = ""
    confirmSearch = " " + str(current_user.id) + " "
    firstInv = True
    firstConf = True
    meetupList = []
    for meetup in current_user.meetups:
        if meetup.confirmed.find(confirmSearch) != -1:
            meetupList.append({'id': meetup.id, 'lat': meetup.lat, 'lng': meetup.lng, 'locationName': meetup.locationCommonName, 'address': meetup.fullAddress})
        invites = invites + "   "
        confirmation = confirmation + "   "
        firstInv = True
        firstConf = True
        for user in meetup.user:
            userSearch = " " + str(user.id) + " "
            if meetup.confirmed.find(userSearch)  != -1:
                if firstConf:   
                    confirmation  = confirmation + user.first_name
                    firstConf = False
                else:
                    confirmation  = confirmation + ", " + user.first_name
            else:
                if firstInv:
                    invites  = invites + user.first_name
                    firstInv = False
                else:
                    invites  = invites + ", " + user.first_name
    inviteList = invites.split("   ")
    confirmationList = confirmation.split("   ")
    return render_template("confirmed.html", key=key, user=current_user, inviteList = inviteList, confirmSearch = confirmSearch, confirmationList = confirmationList, meetupList = meetupList)


@views.route('/confirm-meetup', methods=['POST'])
def confirm_meetup():
    meetup = json.loads(request.data)
    meetupId = meetup['meetupId']
    meetup = Meetup.query.get(meetupId)
    if meetup: # if the user has pressed confirm on a valid meetup
        meetup.confirmed= meetup.confirmed + " " + str(current_user.id) + " " # add them to the confirmed list
        db.session.commit()
    return jsonify({})

@views.route('/decline-meetup', methods=['POST'])
def decline_meetup():
    meetup = json.loads(request.data)
    meetupId = meetup['meetupId']
    meetup = Meetup.query.get(meetupId)
    confirmRemove = ' ' + str(current_user.id) + ' '
    if meetup: # if the user has declined a valid meetup
        meetup.confirmed = meetup.confirmed.replace(confirmRemove, '', 1) # get rid of them on the confirmed list
        current_user.meetups.remove(meetup) # remove their relationship on the user_meetup table
        meetup.invitations = meetup.invitations.replace(' ' + current_user.email + ' ', ' ', 1) # remove their email from the invite list
        if meetup.declined == "": # add them to the meetup declined column for that meetup
            meetup.declined = current_user.first_name
        else:
            meetup.declined = meetup.declined + ", " + current_user.first_name
        db.session.commit()
        flash("\"" + meetup.title + "\" meetup declined.", category = 'error')
    
    return jsonify({})

@views.route('/delete-meetup', methods=['POST'])
def delete_meetup():
    meetup = json.loads(request.data)
    meetupId = meetup['meetupId']
    meetup = Meetup.query.get(meetupId)
    if meetup: # if the user has deleted a valid meetup
        flash("\"" + meetup.title + "\" meetup deleted.", category = 'error')
        db.session.delete(meetup) # delete the meetup and all of its relationships
        db.session.commit()
    return jsonify({})

@views.route('/new-owner', methods=['POST'])
def new_owner():
    meetup = json.loads(request.data)
    meetupId = meetup['meetupId']
    meetup = Meetup.query.get(meetupId)
    owner = json.loads(request.data)
    ownerEmail = owner['newOwner']
    ownerEmail = ownerEmail.strip()
    ownerEmail = ownerEmail.lower() # ensure that the email is in lowercase and there are no spaces
    user = User.query.filter_by(email=ownerEmail).first()
    invited = False
    if user: # if the user entered a valid email to transfer ownership to
        for userMeetups in user.meetups: # cycle through the meetups that have a relationship with the potential new owner
            if userMeetups.id == meetupId: # if one of those meetups has the same id as the current one (meaning the new owner has been invited)
                invited = True
                break
        if invited: # if the new owner is invited, transfer the ownership
            meetup.owner = user.id
            meetup.owner_firstname = user.first_name
            db.session.commit()
            flash('Ownership of \"' + meetup.title + '\" has successfully been transferred to ' + user.first_name + ".", category = 'success')
        else:
            flash(user.email + " is not invited to the meetup. They must be invited before you can transfer ownership.", category = 'error')
    else:
        flash('There is no user registered with email \"' + ownerEmail + "\".", category = 'error')
    
    return jsonify({})

@views.route('/invite-users', methods=['POST'])
def invite_users():
    meetup = json.loads(request.data)
    meetupId = meetup['meetupId']
    meetup = Meetup.query.get(meetupId)
    invites = json.loads(request.data)
    fullInvites = invites['invites']
    fullInvites = fullInvites.strip()
    fullInvites = fullInvites.lower()
    newAttendees = fullInvites.split(' ')
    no_error = True
    for newPerson in newAttendees: # cycle through the new users that are invited
        user = User.query.filter_by(email=newPerson).first()
        if user: # if the invited user is registered
            if meetup.invitations.find(" " + newPerson + " ") == -1: # if the invited user is not already invited
                arr = meetup.declined.split(', ')
                if user.first_name in arr: # if the invited user already declined, re-invite them
                    arr.remove(user.first_name)
                    if arr: # if arr isn't empty, put meetup.declined back together
                        first = True
                        for names in arr:
                            if first:
                                meetup.declined = names
                                first = False
                            else:
                                meetup.declined = meetup.declined + ", " + names
                    else:
                        meetup.declined = ''
                user.meetups.append(meetup) # create a relationship between the new user and meetup
                meetup.invitations = meetup.invitations + newPerson + " " # add the new user to the invite list
            else:
                flash("\"" + user.first_name + "\" is already invited to this meetup. Please only add users that are not already invited. ", category='error')
                no_error = False
        else: 
            flash('There is no account associated with \"' + newPerson + '\". Please ensure the email invitations are in the correct format.', category='error')
            no_error = False
    if no_error: # if there were no errors adding the new users, commit the changes to the database
        db.session.commit()
        flash('Invitations sent!', category='success')
    else:
        flash('There was an error sending the invitations. Please try again.', category='error')
    return jsonify({})
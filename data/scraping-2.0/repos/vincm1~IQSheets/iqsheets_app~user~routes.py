"""Routes for user"""
from datetime import datetime
import stripe
from flask import Blueprint, Markup, render_template, redirect, request, flash, url_for
from flask_login import login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_mail import Message
from iqsheets_app import db
from iqsheets_app.models import User
from .forms import RegistrationForm, LoginForm, EditUserForm, ChangePasswordForm, ResetPasswordRequestForm, ResetPasswordForm
from .token import generate_confirmation_token, confirm_token
from .email import send_email
from werkzeug.utils import secure_filename
from iqsheets_app.utils.decorators import check_confirmed_mail, non_oauth_required
from iqsheets_app.openai import openai_chat
from iqsheets_app.core.forms import NewsletterForm

################
#### config ####
################

user_blueprint = Blueprint('user', __name__)

stripe.api_key = "sk_test_51MpD8VHjForJHjCtVZ317uTWseSh0XxZkuguQKo9Ei3WjaQdMDpo2AbKIYPWl2LXKPW3U3h6Lu71E94Gf1NvrHKE00xPsZzRZZ"

YOUR_DOMAIN = 'http://localhost:5000'

################
#### routes ####
################

@user_blueprint.route('/register', methods=['GET', 'POST'])
def register():
    """ Registering a local user """
    form = RegistrationForm()
    form_nl = NewsletterForm()
    
    if form.validate_on_submit() and request.method == "POST":
        user = User(email=form.email.data, password=form.password.data)

        db.session.add(user)
        db.session.commit()
        
        token = generate_confirmation_token(user.email)
        confirm_url = url_for('user.confirm_email', token=token, _external=True)
        html = render_template('user/email/activate.html', confirm_url=confirm_url)
        subject = "Bitte bestätige Deine Email für IQSheets!"
        send_email(user.email, subject, html)
        
        flash(f'Eine Bestätigungs-Email wurde an {user.email} geschickt.', 'success')
        return redirect(url_for('user.login'))
    
    return render_template('user/signup.html', form=form, form_nl=form_nl)

@user_blueprint.route('/login', methods=['GET', 'POST'])
def login():
    """Login User"""
    form = LoginForm()
    form_2 = ResetPasswordForm()
    form_nl = NewsletterForm()
    
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        # Check if user exist
        if user:
            # Check if user confirmed registration link
            if user.is_confirmed:
                # Check password of user
                if user.check_password(form.password.data):
                    if user.is_admin:
                        login_user(user, remember=True)
                        return redirect(url_for('dashboard.dashboard'))
                    else:
                        user.check_payment()
                        if user.stripe_customer_id and user.stripe_sub_id is not None:
                            login_user(user, remember=True)
                            return redirect(url_for('dashboard.dashboard'))
                        else:
                            return redirect(f"https://buy.stripe.com/test_aEU7t68NY7Dm6ukbIL?prefilled_email={user.email}")    
                else:
                    flash('Prüfe deine Anmeldedaten!', 'danger')
                    return redirect(url_for('user.login'))
            else:
                flash('Bitte bestätige deine Anmeldung per Link', 'danger')
        else:
            flash('Prüfe deine Anmeldedaten!', 'danger')
            return redirect(url_for('user.login'))
    return render_template('user/login.html', form=form, form_2=form_2, form_nl=form_nl)    

@user_blueprint.route('/logout')
@login_required
def logout():
    """Logout"""
    logout_user()
    return redirect(url_for('core.index'))

@user_blueprint.route('/confirm/<token>')
def confirm_email(token):
    """ Sending confirmation Email to user after signup """
    email = confirm_token(token)
    user = User.query.filter_by(email=email).first_or_404()
    print(user.email)
    form_nl = NewsletterForm()
    if user.email == email:
        user.is_confirmed = True
        user.confirmed_on = datetime.now()
        db.session.add(user)
        db.session.commit()
        flash('Account bestätigt', 'success')
        stripe_link = f"https://buy.stripe.com/test_aEU7t68NY7Dm6ukbIL?prefilled_email={user.email}"
        # return render_template('stripe/checkout.html', user_email=email, form_nl=form_nl)
    else:
        flash('Der Bestätigungslink ist abgelaufen oder invalide.', 'danger')
    return redirect(url_for('core.index'))

@user_blueprint.route('/unconfirmed')
@login_required
def unconfirmed():
    """ Checking if user confirmed email link """
    if current_user.is_confirmed:
        return redirect('user.login')
    flash('Bitte Account bestätigen', 'warning')
    return render_template('user/unconfirmed.html')

@user_blueprint.route('/resend')
@login_required
def resend_confirmation():
    """ Resending a confirmation link after signup """
    token = generate_confirmation_token(current_user.email)
    confirm_url = url_for('user.confirm_email', token=token, _external=True)
    html = render_template('user/email/activate.html', confirm_url=confirm_url)
    subject = "Bitte bestätige Deine Email für IQSheets!"
    send_email(current_user.email, subject, html)
    flash(f'Eine Bestätigungs-Email wurde an {current_user.email} geschickt.', 'success')
    return redirect(url_for('user.unconfirmed'))

@user_blueprint.route('/einstellungen/profil', methods=['GET','POST'])
@login_required
@check_confirmed_mail
def edit_user():
    """Edit user profile"""
    form = EditUserForm()
    if form.validate_on_submit() and request.method == 'POST':
        if form.firstname.data and form.firstname.data != current_user.firstname:
            current_user.firstname = form.firstname.data
        if form.lastname.data and form.lastname.data != current_user.lastname:
            current_user.lastname = form.lastname.data
        if form.lastname.data and form.job_description.data != current_user.job_description:
            current_user.job_description = form.job_description.data
        db.session.add(current_user)
        db.session.commit()
        flash("Profil erfolgreich bearbeitet", "success")
    
    return render_template('user/profil.html', form=form, active_page='edit_user')

@user_blueprint.route('/einstellungen/passwort', methods=['GET', 'POST'])
@login_required
@check_confirmed_mail
@non_oauth_required
def edit_password():
    """Edit user profile"""
    form = ChangePasswordForm()
    if form.validate_on_submit() and request.method == 'POST':
        if current_user.check_password(form.old_password.data):
            if current_user.check_password(form.password.data):
                flash("Neues Passwort identisch!", 'danger')
            else:
                current_user.password_hash = generate_password_hash(form.password.data)
                db.session.add(current_user)
                db.session.commit()
                flash("Passwort erfolgreich geändert!", "success")
        else:
            flash('Altes Passwort stimmt nicht!', 'danger')
    return render_template('user/change_password.html', form=form, active_page='edit_password')

@user_blueprint.route('/payments', methods=['GET'])
@login_required
@check_confirmed_mail
def user_payments():
    """ User payments routes """
    stripe_sub_id = current_user.stripe_sub_id
    sub = stripe.Subscription.retrieve(id=stripe_sub_id)
    stripe_cust_id = current_user.stripe_customer_id
    created_date, days_to_due = datetime.utcfromtimestamp(sub['created']).strftime('%d.%m.%Y'), sub['days_until_due']
    current_payment_start, current_payment_end = datetime.utcfromtimestamp(sub["current_period_start"]).strftime('%d.%m.%Y'), datetime.utcfromtimestamp(sub["current_period_end"]).strftime('%d.%m.%Y')
    
    return render_template('user/payments.html', sub=sub, stripe_cust_id=stripe_cust_id,
                           stripe_sub_id=stripe_sub_id, created_date=created_date, days_to_due=days_to_due,
                           current_payment_start=current_payment_start, current_payment_end=current_payment_end)

@user_blueprint.route('/passwort_zuruecksetzen', methods=['GET', 'POST'])
def reset_password_request():
    """ Sending a password request """
    form = ResetPasswordRequestForm()
    form_nl = NewsletterForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.is_confirmed:
            
            token = generate_confirmation_token(user.email)
            confirm_url = url_for('user.reset_password', token=token, _external=True)
            subject = f"Passwort Reset für {user.email} IQSheets!"
            html = render_template('user/email/reset_password.html', confirm_url=confirm_url, form_nl=form_nl)
            send_email(user.email, subject, html)
            flash('Prüfe deine Emails', 'success')
            redirect(url_for('user.login'))
        else:
            flash('Kein Profil unter dieser Emailadresse', 'warning')
            #return redirect(url_for('core.index'))
    return render_template('user/reset_password_request.html',
                           title='Passwort zurücksetzen', form=form, form_nl=form_nl)
    
@user_blueprint.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    """ Resetting password after clicking on tokenized mail link """
    email = confirm_token(token)   
    user = User.query.filter_by(email=email).first_or_404()
    
    if not user:
        return redirect(url_for('core.index'))
    
    form = ResetPasswordForm()
    
    if form.validate_on_submit():
        user.password_hash = generate_password_hash(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Passwort zurückgesetzt', 'success')
        return redirect(url_for('user.login'))
    return render_template('user/reset_password.html', form=form)


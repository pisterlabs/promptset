from flask import Flask, request, jsonify # pip install flask
from dotenv import load_dotenv  # Ajout de cette ligne
from classes.mesageses import Message
from middleware.verifToken import verify_token
from datetime import datetime
import jwt
import os
from database import db_singleton

import openai
from filterSpecialCaracters import filter_special_characters
my_engine = os.getenv("OPENAI_ENGINE")

load_dotenv()
secret_key = os.getenv("SECRET_KEY")


class message_controller():
    def MessageMethod(personnageConversation):
        # Create a database connection and cursor
        conn, cursor= db_singleton.get_cursor()

        verify_token(request.headers.get('Token'))
        if verify_token(request.headers.get('Token')) == 404:
            return jsonify({'error': 'Token invalide'}), 404
        elif verify_token(request.headers.get('Token')) == 505:
            return jsonify({'error': 'Token expiré'}), 505
        else:
            user = jwt.decode(request.headers.get('Token'), secret_key, algorithms=["HS256"])
            cursor.execute("SELECT id FROM users WHERE username = %s", (user['username'],))
            user_id = cursor.fetchone()
        if request.method == 'GET':
                try:
                    cursor.execute("SELECT p.id FROM personnages p INNER JOIN users us on p.user_id = us.id WHERE name = %s AND us.id = %s", (personnageConversation, user_id[0],))
                    personnage_id = cursor.fetchone()
                    
                    cursor.execute("""
            SELECT messages.id, messages.IsHuman, messages.conversation_id, messages.message, messages.sending_date, users.username, personnages.name 
            FROM messages
            INNER JOIN users ON messages.user_id = users.id
            INNER JOIN personnages ON messages.personnage_id = personnages.id
            WHERE messages.personnage_id = %s AND users.id = %s
            ORDER BY messages.sending_date
        """, (personnage_id[0], user_id[0],))
                    rows = cursor.fetchall()
                    messages = []

                    for row in rows:
                        message_temp = Message.from_map({
                            'IsHuman': row[1],
                            'conversation_id': row[2],
                            'message': row[3],
                            'sending_date': row[4],
                            'users_id': row[5],
                            'personnage_id': row[6]
                        })
                        messages.append(message_temp.to_map())
                    return jsonify(messages), 200
                except Exception as e:
                    return jsonify({'error': str(e)}), 500
                
        elif request.method == 'POST':
            data = request.get_json()
            message = Message.from_map(data)
            try:
                cursor.execute("SELECT id FROM personnages WHERE name = %s AND user_id = %s", (personnageConversation, user_id[0],))
                personnage_id = cursor.fetchone()
                if not personnage_id:
                    # Aucun résultat trouvé pour cet univers
                    return jsonify({'error': 'Le personnage n\'existe pas.'}), 404
                
                cursor.execute("SELECT id FROM conversations WHERE personnage_id = %s AND user_id = %s", (personnage_id[0],user_id[0],))
                conversation_id = cursor.fetchone()
                if not conversation_id:
                    # Aucun résultat trouvé pour cet univers
                    return jsonify({'error': 'La conversation n\'existe pas.'}), 404

		        # Obtenir la date et l'heure actuelles
                current_datetime = datetime.now()
            	# Formater la date et l'heure en tant que chaîne avec année, mois, jour, heure et minute
                formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
                
                cursor.execute("INSERT INTO messages (IsHuman, conversation_id, message, sending_date, user_id, personnage_id) VALUES (%s, %s, %s, %s, %s, %s)", (1, conversation_id[0], message.message, formatted_datetime, user_id[0], personnage_id[0],))
                conn.commit()

                # Générer avec OpenAI
                cursor.execute("""SELECT p.univers_id
                            FROM personnages p
                            INNER JOIN users us on p.user_id = us.id
                            WHERE p.name = %s AND us.id = %s
                       """, (personnageConversation, user_id[0]))
                univers_id = cursor.fetchall()
                if univers_id is None:
                    return jsonify({'error': 'L\'univers n\'existe pas'}), 404
                cursor.execute("""SELECT p.description, u.description, u.name
                                    FROM personnages p
                                    INNER JOIN	univers u on p.univers_id = u.id
                                    INNER JOIN users us on u.user_id = us.id
                                    WHERE us.id = %s AND p.name = %s AND u.id = %s
                                    """, (user_id[0], personnageConversation, univers_id[0][0]))
                row = cursor.fetchall()
                
                if row is None:
                    return jsonify({'error': 'Le personnage n\'existe pas'}), 404
                else:
                    for row in row:
                        personnageDescription = row[0]
                        universDescription = row[1]
                        universName = row[2]
                if personnageDescription is None:
                    return jsonify({'error': 'Le personnage n\'a pas de description'}), 404
                if universDescription is None:
                    return jsonify({'error': 'L\'univers n\'a pas de description'}), 404
                
                response = openai.Completion.create(
                    engine=my_engine,  # Choisir le moteur de génération de texte
                    prompt=f"Dans le cadre d'un jeu de rôle, l'IA devient le personnage de {personnageConversation} issu de l'univers de {universName} et répond à l'humain en français.  Voici la description de {personnageConversation}: {personnageDescription} \n Human: {message} \n AI:",
                    max_tokens=200,  # Limitez le nombre de tokens pour contrôler la longueur de la réponse
                    n=1,  # Nombre de réponses à générer
                    stop=None  # Vous pouvez spécifier des mots pour arrêter la génération
                )
                reponse = response.choices[0].text.strip()

                filtered_text = filter_special_characters(reponse)
                
                response_message = filtered_text

                """ message_response = Message.generate_message(message.message, personnageConversation, personnageDescription, universDescription) """

                cursor.execute("""INSERT INTO messages (IsHuman, conversation_id, message, sending_date, user_id, personnage_id)
                                VALUES (%s, %s, %s, %s, %s, %s)""", (0, conversation_id[0], response_message, formatted_datetime, user_id[0], personnage_id[0]))
                conn.commit()
                message.id = cursor.lastrowid

                return jsonify({'message': f"Le message suivant : {message.message} a bien etait enregistrer dans La BDD dans la conversation numéro : {conversation_id[0]} qui contient l'utilisateur numéro : {user_id[0]} et le personnage numéro : {personnage_id[0]} et la reponse du personnage est : {response_message}"}), 201
            

            except Exception as e:
                return jsonify({'error': str(e)}), 500
            finally:
                cursor.close()
                conn.close()


    def MessageMethodSpecifique(personnageConversation):
        conn, cursor = db_singleton.get_cursor()

        verify_token(request.headers.get('Token'))
        if verify_token(request.headers.get('Token')) == 404:
            return jsonify({'error': 'Token invalide'}), 404
        elif verify_token(request.headers.get('Token')) == 505:
            return jsonify({'error': 'Token expiré'}), 505
        else:
            user = jwt.decode(request.headers.get('Token'), secret_key, algorithms=["HS256"])
            cursor.execute("SELECT id FROM users WHERE username = %s", (user['username'],))
            user_id = cursor.fetchone()

        if request.method == 'PUT':
            
            try:
                cursor.execute("SELECT id FROM personnages WHERE name = %s AND user_id = %s", (personnageConversation, user_id[0],))
                personnage_id = cursor.fetchone()
                if not personnage_id:
                    # Aucun résultat trouvé pour cet univers
                    return jsonify({'error': 'Le personnage n\'existe pas.'}), 404
                
                cursor.execute("SELECT id FROM conversations WHERE personnage_id = %s AND user_id = %s", (personnage_id[0],user_id[0],))
                conversation_id = cursor.fetchone()
                if not conversation_id:
                    # Aucun résultat trouvé pour cet univers
                    return jsonify({'error': 'La conversation n\'existe pas.'}), 404
                cursor.execute("""SELECT m.message, m.id
                  FROM messages m
                  INNER JOIN personnages p ON m.personnage_id = p.id
                  INNER JOIN users us ON p.user_id = us.id
                  WHERE p.name = %s AND us.id = %s AND m.IsHuman = 0
                  ORDER BY m.sending_date DESC LIMIT 1""", (personnageConversation, user_id[0],))
                row = cursor.fetchone()

                if row is None:
                    return jsonify({'error': 'Le personnage n\'existe pas ou n\'a pas de message'}), 404

                last_message_ia = row[0]
                id_message = row[1]

                
                cursor.execute("""SELECT m.message
                                    FROM messages m
                                    INNER JOIN personnages p on m.personnage_id = p.id
                                    INNER JOIN users us on p.user_id = us.id
                                    WHERE p.name = %s AND us.id = %s AND m.IsHuman = 1
                                    ORDER BY m.sending_date DESC LIMIT 1""", (personnageConversation, user_id[0],))
                row = cursor.fetchone()
                if row is None:
                    return jsonify({'error': 'Le personnage n\'existe pas'}), 404
                else:
                    for row in row:
                        last_message_human = row
                if last_message_human is None:
                    return jsonify({'error': 'Le personnage n\'a pas de message'}), 404

		        # Obtenir la date et l'heure actuelles
                current_datetime = datetime.now()
            	# Formater la date et l'heure en tant que chaîne avec année, mois, jour, heure et minute
                formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

                # Générer avec OpenAI
                cursor.execute("""SELECT p.univers_id
                            FROM personnages p
                            INNER JOIN users us on p.user_id = us.id
                            WHERE p.name = %s AND us.id = %s
                       """, (personnageConversation, user_id[0]))
                univers_id = cursor.fetchall()
                if univers_id is None:
                    return jsonify({'error': 'L\'univers n\'existe pas'}), 404
                
                cursor.execute("""SELECT p.description, u.description, u.name
                                    FROM personnages p
                                    INNER JOIN	univers u on p.univers_id = u.id
                                    INNER JOIN users us on u.user_id = us.id
                                    WHERE us.id = %s AND p.name = %s AND u.id = %s
                                    """, (user_id[0], personnageConversation, univers_id[0][0]))
                row = cursor.fetchall()
                
                if row is None:
                    return jsonify({'error': 'Le personnage n\'existe pas'}), 404
                else:
                    for row in row:
                        personnageDescription = row[0]
                        universDescription = row[1]
                        universName = row[2]
                if personnageDescription is None:
                    return jsonify({'error': 'Le personnage n\'a pas de description'}), 404
                if universDescription is None:
                    return jsonify({'error': 'L\'univers n\'a pas de description'}), 404
                
                response = openai.Completion.create(
                    engine=my_engine,  # Choisir le moteur de génération de texte
                    prompt=f"Dans le cadre d'un jeu de rôle, l'IA devient le personnage de {personnageConversation} issu de l'univers de {universName} et répond à l'humain en français mais pas avec ce message: {last_message_ia}.  Voici la description de Gangplank: {personnageDescription} de l'univers de {universName} et voici la description de l'univers: {universDescription} Human: {last_message_human} AI:",
                    max_tokens=200,  # Limitez le nombre de tokens pour contrôler la longueur de la réponse
                    n=1,  # Nombre de réponses à générer
                    stop=None  # Vous pouvez spécifier des mots pour arrêter la génération
                )
                reponse = response.choices[0].text.strip()

                filtered_text = filter_special_characters(reponse)
                
                response_message = filtered_text

                """ message_response = Message.generate_message(message.message, personnageConversation, personnageDescription, universDescription) """

                

                cursor.execute("""UPDATE messages 
                  SET IsHuman = %s, 
                      conversation_id = %s, 
                      message = %s, 
                      sending_date = %s, 
                      user_id = %s, 
                      personnage_id = %s
                  WHERE id = %s""", 
                  (0, conversation_id[0], response_message, formatted_datetime, user_id[0], personnage_id[0],id_message ))
                

                conn.commit()
                ioajdoijzd = cursor.lastrowid

                return jsonify({'message': f"Le message suivant : {last_message_ia} a bien etait modifier dans La BDD dans la conversation numéro : {conversation_id[0]} qui contient l'utilisateur numéro : {user_id[0]} et le personnage numéro : {personnage_id[0]} et la nouvelle reponse du personnage est : {response_message}"}), 201

            except Exception as e:
                return jsonify({'error': str(e)}), 500
            finally:
                cursor.close()
                conn.close()


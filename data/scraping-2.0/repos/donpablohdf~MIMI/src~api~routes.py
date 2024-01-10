"""
This module takes care of starting the API Server, Loading the DB and Adding the endpoints
"""
from __future__ import print_function
import os
from dotenv import load_dotenv

from flask import Flask, request, jsonify, url_for, Blueprint
from api.models import db, Users, Actividades, Reservas, Comentarios
import jwt
from werkzeug.security import check_password_hash, generate_password_hash
from flask_jwt_extended import create_access_token
from flask_jwt_extended import jwt_required, get_jwt_identity
import uuid
import openai
load_dotenv()
openai.api_key = os.getenv("CHAT_GPT")

api = Blueprint("api", __name__)

app_path = os.getcwd()  # find path
str_delete = "src"
app_path = app_path.replace(str_delete, "")


@api.route("/chatgpt", methods=["POST", "GET"])
def handle_chatgpt():

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Ruta del Cares en León, España",
        temperature=0.9,
        max_tokens=120,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=[" AI:"]
    )
    return response, 200


@api.route("/usuarios_index", methods=["POST", "GET"])
def handle_usu_index():
    usu_ind = Users.get_guias_index()
    if usu_ind:
        all_usu_ind = [Users.serialize() for Users in usu_ind]
        return jsonify(all_usu_ind), 200
    return jsonify({"message": "Error al recuperar datos"}), 400


@api.route("/usuario/<int:usuario_id>", methods=["POST", "GET"])
def handle_user(usuario_id):
    user = Users.get_by_id(usuario_id)
    the_user = Users.serialize(user)
    return jsonify(the_user), 200


@api.route("/desactiva_user/<int:usuario_id>", methods=["POST", "GET"])
@jwt_required()
def handle_del(usuario_id):
    user = Users.desactiva_by_id(usuario_id)
    actividades = Actividades.get_by_guia(usuario_id)
    if actividades:
        for x in actividades:
            print(x.id)
            Actividades.desactiva_by_id(x.id)

    resevas_usr = Reservas.get_by_user(usuario_id)
    if resevas_usr:
        for t in resevas_usr:
            Reservas.desactiva_by_id(t.id)

    comentarios = Comentarios.get_by_usr(usuario_id)
    if comentarios:
        for c in comentarios:
            Comentarios.desactiva_by_id(c.id)

    return jsonify(user), 200


@api.route("/modifica_user/<int:usuario_id>", methods=["POST", "GET"])
@jwt_required()
def handle_mod(usuario_id):

    data = request.get_json()
    mod_user = Users.modifica_by_id(usuario_id, data)
    # print(mod_user)
    if mod_user:
        return jsonify(mod_user), 200
    else:
        return jsonify(mod_user), 401


@api.route("/foto_user/<int:usuario_id>", methods=["POST", "GET"])
@jwt_required()
def handle_foto(usuario_id):
    if request.method == "POST":
        f = request.files["archivo"]
        renom = uuid.uuid4()
        archivo = app_path + "public/imgs/users/" + \
            str(usuario_id) + "_" + str(renom)
        f.save(os.path.join(archivo))
        img_bbdd = "imgs/users/" + str(usuario_id) + "_" + str(renom)
        foto_user = Users.foto_by_id(usuario_id, img_bbdd)
        return jsonify(foto_user), 200
    else:
        return jsonify("No POST"), 400


@api.route("/new_user", methods=["POST"])
def handle_new():
    user = request.get_json()
    user_new = Users.new_user(user)
    return jsonify(user_new), 200


@api.route("/login", methods=["POST", "GET"])
def login_user():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Sin datos"}), 401
    user = Users.query.filter_by(email=data["email"]).first()
    if user:
        if user.activo == 1:
            if check_password_hash(user.password, data["password"]):
                SECRET = os.getenv("FLASK_APP_KEY")  # variable ENV
                token = jwt.encode(
                    {
                        "id": user.id,
                    },
                    SECRET,
                )
                access_token = create_access_token(token)
                return jsonify({"token": access_token, "userid": user.id}), 200
            return jsonify({"error": "Contraseña incorrecta"}), 401
        else:
            return jsonify({"error": "No existe el usuario"}), 401
    return jsonify({"error": "no_user"}), 401


@api.route("/new_pass", methods=["POST", "GET"])
@jwt_required()
def handle_pass():
    data = request.get_json()
    if data["email"]:
        pass_user = Users.pass_by_mail(data["email"])
        if pass_user:
            return jsonify(pass_user), 200
        else:
            return jsonify(pass_user), 400
    else:
        return jsonify("No email"), 400


# --------------------------------------- ACTIVIDADES---------------------


@api.route("/actividad/<int:actividad_id>", methods=["POST", "GET"])
def handle_acti(actividad_id):
    acti = Actividades.get_by_id(actividad_id)
    the_acti = Actividades.serialize(acti)
    return jsonify(the_acti), 200


@api.route("/actividad_guia/<int:guia_id>", methods=["POST", "GET"])
def handle_acti_guia(guia_id):
    act_guia = Actividades.get_by_guia(guia_id)
    if act_guia:
        all_act_guia = [Actividades.serialize() for Actividades in act_guia]
        return jsonify(all_act_guia), 200
    return jsonify({"message": "Error al recuperar datos"}), 400


@api.route("/actividad_user/<int:user_id>", methods=["POST", "GET"])
def handle_acti_user(user_id):
    act_user = Actividades.get_by_user(user_id)
    if act_user:
        all_act_user = [Actividades.serialize() for Actividades in act_user]
        return jsonify(all_act_user), 200
    return jsonify({"message": "Error al recuperar datos"}), 400


@api.route("/actividades_index", methods=["POST", "GET"])
def handle_acti_index():
    act_ind = Actividades.act_index()
    if act_ind:
        all_act_index = [Actividades.serialize() for Actividades in act_ind]
        return jsonify(all_act_index), 200
    return jsonify({"message": "Error al recuperar datos"}), 400


@api.route("/new_act/<int:guia_id>", methods=["POST", "GET"])
@jwt_required()
def new_act(guia_id):

    if request.method == "POST":
        if request.files:
            f = request.files["archivo"]
            renom = uuid.uuid4()
            archivo = (
                app_path + "public/imgs/actividades/" +
                str(guia_id) + "_" + str(renom)
            )
            f.save(os.path.join(archivo))
            img_bbdd = "imgs/actividades/" + str(guia_id) + "_" + str(renom)
        else:
            img_bbdd = ""

        data = {
            "nombre": request.form["nombre"],
            "descripcion": request.form["descripcion"],
            "precio": request.form["precio"],
            "fecha": request.form["fecha"],
            "id_guia": guia_id,
            "ciudad": request.form["ciudad"],
            "foto": img_bbdd,
        }
        new_act_guia = Actividades.new_act(guia_id, data)

        return jsonify(new_act_guia), 200
    else:
        return jsonify("No POST"), 400


@api.route("/modifica_act/<int:act_id>", methods=["POST", "GET"])
@jwt_required()
def act_mod(act_id):
    data = request.get_json()
    mod_act = Actividades.modifica_by_id(act_id, data)
    return jsonify(mod_act), 200


@api.route("/foto_act/<int:act_id>/<int:guia_id>", methods=["POST", "GET"])
@jwt_required()
def act_foto(act_id, guia_id):
    if request.method == "POST":
        f = request.files["ftAct"]
        renom = uuid.uuid4()
        archivo = (
            app_path + "public/imgs/actividades/" +
            str(guia_id) + "_" + str(renom)
        )
        f.save(os.path.join(archivo))
        img_bbdd = "imgs/actividades/" + str(guia_id) + "_" + str(renom)
        foto_act = Actividades.foto_by_id(act_id, img_bbdd)
        return jsonify(foto_act), 200
    else:
        return jsonify("No POST"), 400


@api.route("/desactiva_act/<int:act_id>", methods=["POST", "GET"])
@jwt_required()
def act_del(act_id):
    user = Actividades.desactiva_by_id(act_id)
    return jsonify(user), 200


@api.route("/search", methods=["POST", "GET"])
def search_act():
    search = Actividades.search()
    group_act = [Actividades.serialize() for Actividades in search]
    return jsonify(group_act), 200


# -----------------------------------RESERVAS-----------------------------------------------------------


@api.route("/reserva/<int:reserva_id>", methods=["POST", "GET"])
def handle_reser(reserva_id):
    reser = Reservas.get_by_id(reserva_id)
    the_reser = Reservas.serialize(reser)
    return jsonify(the_reser), 200


@api.route("/reserva_guia/<int:guia_id>", methods=["POST", "GET"])
def handle_reser_guia(guia_id):
    reser_guia = Reservas.get_by_guia(guia_id)
    if reser_guia:
        all_reser_guia = [Reservas.serialize() for Reservas in reser_guia]
        return jsonify(all_reser_guia), 200
    return jsonify({"message": "Error al recuperar datos"}), 400


@api.route("/reserva_user/<int:user_id>", methods=["POST", "GET"])
def handle_reser_user(user_id):
    reser_user = Reservas.get_by_user(user_id)
    if reser_user:
        all_reser_user = [Reservas.serialize() for Reservas in reser_user]
        return jsonify(all_reser_user), 200
    return jsonify({"message": "Error al recuperar datos"}), 400


@api.route("/reserva_est/<int:estado>", methods=["POST", "GET"])
def reser_estado(estado):
    reser_est = Reservas.res_estado(estado)
    if reser_est:
        all_reser_est = [Reservas.serialize() for Reservas in reser_est]
        return jsonify(all_reser_est), 200
    return jsonify({"message": "Error al recuperar datos"}), 400


@api.route("/reserva_canc/<int:id_reserva>", methods=["POST", "GET"])
@jwt_required()
def reser_canc(id_reserva):
    reser_c = Reservas.desactiva_by_id(id_reserva)
    return jsonify(reser_c), 200


@api.route("/reserva_new", methods=["POST", "GET"])
@jwt_required()
def res_nw():
    data = request.get_json()
    nw_res = Reservas.res_nueva(data)
    return jsonify(nw_res), 200


# -------------------COMENTARIOS------------------


@api.route("/comentarios/<int:comen_id>", methods=["POST", "GET"])
def handle_comen(comen_id):
    comentario = Comentarios.get_by_id(comen_id)
    the_comen = Comentarios.serialize(comentario)
    return jsonify(the_comen), 200


@api.route("/comentarios_act/<int:id_actividad>", methods=["POST", "GET"])
def comen_act(id_actividad):
    com_act = Comentarios.get_by_act(id_actividad)
    if com_act:
        all_com_act = [Comentarios.serialize() for Comentarios in com_act]
        return jsonify(all_com_act), 200
    return jsonify({"message": "Error al recuperar datos"}), 400


@api.route("/comen_new/<int:id_actividad>/<int:id_usuario>", methods=["POST", "GET"])
@jwt_required()
def comen_nw(id_actividad, id_usuario):
    data = request.get_json()
    nw_comen = Comentarios.com_nuevo(id_actividad, id_usuario, data)
    return jsonify(nw_comen), 200


@api.route("/desactiva_com/<int:comen_id>", methods=["POST", "GET"])
@jwt_required()
def comen_del(comen_id):
    com = Comentarios.desactiva_by_id(comen_id)
    return jsonify(com), 200

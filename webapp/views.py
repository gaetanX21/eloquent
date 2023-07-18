from flask import Blueprint, render_template, request, flash, redirect, url_for, jsonify
from flask_login import login_required, current_user
from .models import Note
from . import db
import json

views = Blueprint("views", __name__)


@views.route("/", methods=["GET", "POST"])
@login_required
def home():
    if request.method == "POST":
        note = request.form.get("note")

        if len(note) == 0:
            flash("Note is empty.", category="error")
        else:
            new_node = Note(data=note, user_id=current_user.id)
            db.session.add(new_node)
            db.session.commit()
            flash("Note added.", category="success")

    return render_template("home.html", user=current_user)


@views.route("/delete-note", methods=["POST"])
@login_required
def delete_note():
    data = json.loads(request.data)
    node_id = data["noteId"]
    note = Note.query.get(node_id)
    if note:
        if note.user_id == current_user.id:
            db.session.delete(note)
            db.session.commit()

    return jsonify({})


@views.route("/record")
@login_required
def record():
    return render_template("record.html", user=current_user)


@login_required
@views.route("/profile")
def profile():
    return render_template("profile.html", user=current_user)


@login_required
@views.route("/upload", methods=["POST"])
def upload():
    blob = request.form.get("audio")
    print("received blob")
    print(type(blob))
    print(blob)

    response = {"status": "success"}

    return jsonify(response)

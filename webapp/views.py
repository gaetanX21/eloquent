from flask import (
    Blueprint,
    render_template,
    request,
    flash,
    redirect,
    url_for,
    jsonify,
    make_response,
)
from flask_login import login_required, current_user
from .models import Recording, PerformanceReport
from . import db
import json
from . import myutils

views = Blueprint("views", __name__)


@views.route("/", methods=["GET", "POST"])
def home():
    print("/ route hit")
    return render_template("home.html", user=current_user)


@views.route("/delete-recording", methods=["POST"])
@login_required
def delete_recording():
    print("/delete-recording route hit")
    data = json.loads(request.data)
    recording_id = data["recordingId"]
    recording = Recording.query.get(recording_id)
    if recording:
        if recording.user_id == current_user.id:
            if recording.performance_report:
                db.session.delete(recording.performance_report)
            db.session.delete(recording)
            db.session.commit()
            flash("Recording deleted successfully", category="success")

    return jsonify({})


@views.route("/record")
@login_required
def record():
    print("/record route hit")
    return render_template("record.html", user=current_user)


@views.route("/profile")
@login_required
def profile():
    print("/profile route hit")
    return render_template("profile.html", user=current_user)


@views.route("/upload", methods=["POST"])
@login_required
def upload():
    print("/upload route hit")
    audio_file = request.files["audio"]

    # add audio file to database
    if audio_file:
        audio_blob = audio_file.read()

        performance_overview = myutils.run_performance_report(audio_blob)

        performance_report = PerformanceReport(
            duration=performance_overview["duration"],
            speech_duration=performance_overview["speech_duration"],
            ratio_speech_time=performance_overview["ratio_speech_time"],
            intensity_mean=performance_overview["intensity_mean"],
            intensity_std=performance_overview["intensity_std"],
            intensity_max=performance_overview["intensity_max"],
            pitch_mean=performance_overview["pitch_mean"],
            pitch_std=performance_overview["pitch_std"],
            pitch_min=performance_overview["pitch_min"],
            pitch_max=performance_overview["pitch_max"],
        )

        new_recording = Recording(
            audio_blob=audio_blob,
            user_id=current_user.id,
            name=audio_file.filename,
            performance_report=performance_report,
        )
        db.session.add(new_recording)
        db.session.commit()
        flash("Recording saved successfully", category="success")

        return "upload successful"
    else:
        flash("No recording found", category="error")

    return "upload unsuccessful"


@views.route("/play_audio/<int:recording_id>")
@login_required
def play_audio(recording_id):
    print("/play_audio route hit")
    recording = Recording.query.get_or_404(recording_id)
    audio_bytes = recording.audio_blob
    response = make_response(audio_bytes)
    response.headers.set("Content-Type", "audio/webm")
    response.headers.set("Content-Disposition", "inline")
    return response


@views.route("/saved")
@login_required
def saved():
    print("/saved route hit")
    return render_template("saved.html", user=current_user)


@views.route("/performance_overview")
@login_required
def performance_overview():
    print("/performance_overview route hit")
    recording_id = int(
        request.args.get("recording_id")
    )  # careful, arguments passed in URL are strings by default
    return render_template(
        "performance_overview.html", user=current_user, recording_id=recording_id
    )

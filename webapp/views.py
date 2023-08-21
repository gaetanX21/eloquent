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
    myutils.myprint("/ route hit", "yellow")
    return render_template("home.html", user=current_user)


@views.route("/delete-recording", methods=["POST"])
@login_required
def delete_recording():
    myutils.myprint("/delete-recording route hit", "yellow")
    data = json.loads(request.data)
    recording_id = data["recordingId"]
    recording = Recording.query.get(recording_id)
    if recording:
        if recording.user_id == current_user.id:
            if recording.performance_report:
                db.session.delete(recording.performance_report)
            db.session.delete(recording)
            db.session.commit()

    # if there are no more recordings, refresh page
    remaining_recordings = Recording.query.filter_by(user_id=current_user.id).all()
    n_remaining_recordings = len(remaining_recordings)
    myutils.myprint(n_remaining_recordings, "cyan")
    ret = jsonify({"n_remaining_recordings": n_remaining_recordings})
    print(ret)
    return ret


@views.route("/delete_all_recordings", methods=["POST"])
@login_required
def delete_all_recordings():
    myutils.myprint("/delete_all_recordings route hit", "yellow")
    recordings = Recording.query.filter_by(user_id=current_user.id).all()
    for recording in recordings:
        if recording.performance_report:
            db.session.delete(recording.performance_report)
        db.session.delete(recording)
    db.session.commit()

    # refresh page
    return redirect(url_for("views.saved"))


@views.route("/record")
@login_required
def record():
    myutils.myprint("/record route hit", "yellow")
    return render_template("record.html", user=current_user)


@views.route("/profile")
@login_required
def profile():
    myutils.myprint("/profile route hit", "yellow")
    return render_template("profile.html", user=current_user)


@views.route("/upload", methods=["POST"])
@login_required
def upload():
    myutils.myprint("/upload route hit", "yellow")
    audio_file = request.files["audio"]

    # add audio file to database
    if audio_file:
        audio_blob = audio_file.read()

        # get metadata
        numberOfChannels = int(request.form["numberOfChannels"])
        sampleRate = int(request.form["sampleRate"])
        sampleSize = int(request.form["sampleSize"])
        mimeType = request.form["mime"]
        locale = request.form["locale"]

        # we want to save to wav bytes if the file is not already in the wav format
        if mimeType != "audio/wav":
            audio_blob = myutils.convert_to_wav(
                audio_blob, numberOfChannels, sampleRate, sampleSize, mimeType
            )

        # now the audioBlob is in wav format! (nice!)
        try:
            performance_overview = myutils.run_performance_report(
                audio_blob, sampleRate, sampleSize, locale, numberOfChannels
            )
        except Exception as e:
            myutils.myprint(e, "red")
            return "upload unsuccessful"

        performance_report = PerformanceReport(
            overall_score=performance_overview["overall_score"],
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
            transcript=performance_overview["transcript"],
            words_per_minute=performance_overview["words_per_minute"],
        )

        new_recording = Recording(
            audio_blob=audio_blob,
            user_id=current_user.id,
            name=audio_file.filename,
            number_of_channels=numberOfChannels,
            sample_rate=sampleRate,
            sample_size=sampleSize,
            mime_type=mimeType,
            locale=locale,
            performance_report=performance_report,
        )
        db.session.add(new_recording)
        db.session.commit()
        # flash("Recording saved successfully", category="success")

        return "upload successful"
    else:
        flash("No recording found", category="error")

    return "upload unsuccessful"


@views.route("/play_audio/<int:recording_id>")
@login_required
def play_audio(recording_id):
    myutils.myprint("/play_audio route hit", "yellow")
    recording = Recording.query.get_or_404(recording_id)
    audio_bytes = recording.audio_blob
    response = make_response(audio_bytes)
    response.headers.set("Content-Type", "audio/webm")
    response.headers.set("Content-Disposition", "inline")
    return response


@views.route("/saved")
@login_required
def saved():
    myutils.myprint("/saved route hit", "yellow")
    return render_template("saved.html", user=current_user)


@views.route("/performance_overview")
@login_required
def performance_overview():
    myutils.myprint("/performance_overview route hit", "yellow")
    recording_id = int(
        request.args.get("recording_id")
    )  # careful, arguments passed in URL are strings by default
    return render_template(
        "performance_overview.html", user=current_user, recording_id=recording_id
    )

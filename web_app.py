import os
import threading
import uuid

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from main import process_video
import turboscribe_automation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JOB_DIR = os.path.join(BASE_DIR, "web_jobs")
FINISHED_VIDEO_DIR = os.path.join(BASE_DIR, "finished_videos")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
SUBTITLE_EXTENSIONS = {".txt", ".srt", ".vtt"}

app = Flask(__name__)
jobs = {}


def _allowed_file(filename, allowed_extensions):
    return os.path.splitext(filename)[1].lower() in allowed_extensions


def _set_job_progress(job_id, percent, message):
    jobs[job_id]["progress"] = percent
    jobs[job_id]["message"] = message


def _run_job(job_id, video_path, transcript_path, use_turboscribe):
    try:
        os.makedirs(FINISHED_VIDEO_DIR, exist_ok=True)
        source_name = os.path.splitext(os.path.basename(video_path))[0]
        safe_source_name = secure_filename(source_name) or "video"
        output_path = os.path.join(FINISHED_VIDEO_DIR, f"{safe_source_name}_legmon_{job_id[:8]}.mp4")

        if use_turboscribe and not transcript_path:
            _set_job_progress(job_id, 8, "Uploading to TurboScribe")
            transcript_path = turboscribe_automation.transcribe_to_srt(
                video_path,
                output_dir=os.path.join(JOB_DIR, job_id),
            )
            _set_job_progress(job_id, 28, "TurboScribe SRT downloaded")

        result_path = process_video(
            input_video_path=video_path,
            output_name=output_path,
            transcript_path=transcript_path,
            progress_callback=lambda percent, message: _set_job_progress(job_id, percent, message),
        )
        jobs[job_id].update({
            "status": "done",
            "progress": 100,
            "message": "Ready to download",
            "output_path": result_path,
            "saved_path": result_path,
        })
    except Exception as exc:
        jobs[job_id].update({
            "status": "error",
            "message": str(exc),
        })


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/jobs", methods=["POST"])
def create_job():
    video = request.files.get("video")
    subtitle = request.files.get("subtitle")
    use_turboscribe = request.form.get("use_turboscribe") == "on"

    if not video or not video.filename:
        return jsonify({"error": "Upload a raw video first."}), 400
    if not _allowed_file(video.filename, VIDEO_EXTENSIONS):
        return jsonify({"error": "Unsupported video format."}), 400
    if subtitle and subtitle.filename and not _allowed_file(subtitle.filename, SUBTITLE_EXTENSIONS):
        return jsonify({"error": "Unsupported subtitle format. Use .txt, .srt, or .vtt."}), 400

    job_id = uuid.uuid4().hex
    job_path = os.path.join(JOB_DIR, job_id)
    os.makedirs(job_path, exist_ok=True)

    video_filename = secure_filename(video.filename)
    video_path = os.path.join(job_path, video_filename)
    video.save(video_path)

    transcript_path = None
    if subtitle and subtitle.filename:
        subtitle_filename = secure_filename(subtitle.filename)
        transcript_path = os.path.join(job_path, subtitle_filename)
        subtitle.save(transcript_path)

    jobs[job_id] = {
        "status": "queued",
        "progress": 1,
        "message": "Queued",
        "output_path": None,
    }

    worker = threading.Thread(target=_run_job, args=(job_id, video_path, transcript_path, use_turboscribe), daemon=True)
    worker.start()
    return jsonify({"job_id": job_id})


@app.route("/api/turboscribe/login", methods=["POST"])
def open_turboscribe_login():
    worker = threading.Thread(target=turboscribe_automation.open_login_browser, daemon=True)
    worker.start()
    return jsonify({"status": "opening"})


@app.route("/api/turboscribe/status")
def turboscribe_status():
    try:
        turboscribe_automation.ensure_logged_in()
        return jsonify({"logged_in": True, "message": "TurboScribe profile is logged in."})
    except Exception as exc:
        return jsonify({"logged_in": False, "message": str(exc)})


@app.route("/api/jobs/<job_id>")
def get_job(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404
    return jsonify({
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
        "download_url": f"/download/{job_id}" if job["status"] == "done" else None,
        "saved_path": job.get("saved_path") if job["status"] == "done" else None,
    })


@app.route("/download/<job_id>")
def download(job_id):
    job = jobs.get(job_id)
    if not job or job.get("status") != "done" or not job.get("output_path"):
        return jsonify({"error": "File is not ready."}), 404
    return send_file(job["output_path"], as_attachment=True, download_name="legmon_export.mp4")


if __name__ == "__main__":
    os.makedirs(JOB_DIR, exist_ok=True)
    os.makedirs(FINISHED_VIDEO_DIR, exist_ok=True)
    app.run(host="127.0.0.1", port=7860, debug=False)

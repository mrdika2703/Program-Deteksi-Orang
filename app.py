from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session
import requests
import cv2
import numpy as np
import os
import csv
from detector import detect_and_track, get_full_log

ESP32_URL = "http://192.168.43.145/capture"  # Ganti sesuai IP ESP32-CAM

app = Flask(__name__)

app.secret_key = 'your_secret_key'  # Ganti dengan key aman

last_count, last_durations, total_people = 0, {}, 0

def get_esp32_frame():
    try:
        resp = requests.get(ESP32_URL, timeout=5)
        npimg = np.frombuffer(resp.content, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print(f"ERROR saat ambil gambar: {e}")
        return None

USERNAME = 'admin'
PASSWORD = 'admin123'

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username == USERNAME and password == PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("index"))
        else:
            return render_template("login.html", error="Username atau password salah")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("logged_in", None)
    return redirect(url_for("login"))

@app.route("/dashboard")
def index():
    global last_count, last_durations, total_people
    return render_template("index.html", count=last_count, durations=last_durations, total=total_people)

@app.route("/video_feed")
def video_feed():
    def gen_frames():
        global last_count, last_durations, total_people
        while True:
            frame = get_esp32_frame()
            if frame is None:
                continue
            processed, count, durations, total = detect_and_track(frame)
            last_count, last_durations, total_people = count, durations, total
            ret, buffer = cv2.imencode('.jpg', processed)
            if not ret:
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/get_status")
def get_status():
    global last_count, last_durations, total_people
    return jsonify({
        "count": last_count,
        "durations": {str(k): int(v) for k, v in last_durations.items()},
        "total": total_people
    })

@app.route("/history")
def history():
    return render_template("history.html")

@app.route("/get_history")
def get_history():
    page = int(request.args.get("page", 1))
    per_page = 20
    data = get_full_log()
    start = (page - 1) * per_page
    end = start + per_page
    paged_data = data[start:end]
    total_pages = (len(data) + per_page - 1) // per_page
    return jsonify({
        "data": paged_data,
        "total_pages": total_pages,
        "current_page": page
    })

@app.route("/grafik_data")
def grafik_data():
    from collections import Counter
    import datetime

    dates = []
    with open("data/log.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row[1]:
                dt = row[1].split()[0]  # Ambil tanggal saja
                dates.append(dt)

    count_by_date = Counter(dates)
    sorted_dates = sorted(count_by_date.items())
    labels = [item[0] for item in sorted_dates]
    values = [item[1] for item in sorted_dates]

    return jsonify({"labels": labels, "values": values})



if __name__ == "__main__":
    app.run(debug=True)

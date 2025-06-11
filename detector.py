import cv2
import time
import random
import csv
import os
from collections import defaultdict
from datetime import datetime
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
tracked_people = {}
next_id = 0
iou_threshold = 0.3
confidence_threshold = 0.7
max_disappeared_time = 2

log_path = "data/log.csv"
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists(log_path):
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Waktu_Masuk", "Waktu_Keluar", "Durasi", "Total_Orang"])

def format_duration(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes} menit {secs} detik"

def detect_and_track(frame):
    global tracked_people, next_id
    current_boxes = []
    results = model(frame, stream=True)

    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) != 0 or float(box.conf[0]) < confidence_threshold:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            current_boxes.append((x1, y1, x2, y2))

    updated_people = {}
    used_ids = set()
    now = time.time()

    for box in current_boxes:
        matched_id = None
        best_iou = 0
        for pid, (prev_box, start_time, color, last_seen) in tracked_people.items():
            iou = get_iou(box, prev_box)
            if iou > iou_threshold and iou > best_iou and pid not in used_ids:
                matched_id = pid
                best_iou = iou

        if matched_id is not None:
            _, start_time, color, _ = tracked_people[matched_id]
            updated_people[matched_id] = (box, start_time, color, now)
            used_ids.add(matched_id)
        else:
            color = tuple(random.randint(0, 255) for _ in range(3))
            updated_people[next_id] = (box, now, color, now)
            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([next_id, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)), "", ""])
            used_ids.add(next_id)
            next_id += 1

    for pid, (box, start_time, color, last_seen) in tracked_people.items():
        if pid not in updated_people and (now - last_seen) < max_disappeared_time:
            updated_people[pid] = (box, start_time, color, last_seen)
        elif pid not in updated_people:
            keluar = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
            durasi = format_duration(now - start_time)
            update_log(pid, keluar, durasi, next_id)

    tracked_people = updated_people

    for pid, (box, start_time, color, _) in tracked_people.items():
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        duration = now - start_time
        cv2.putText(frame, f"ID {pid} | {int(duration)}s", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    durations = {pid: now - data[1] for pid, data in tracked_people.items()}
    return frame, len(tracked_people), durations, next_id

def update_log(pid, keluar, durasi, total_people):
    lines = []
    with open(log_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row[0] == str(pid) and row[2] == "":
                row[2] = keluar
                row[3] = durasi
                row.append(str(total_people))  # Tambahkan total orang saat keluar
            lines.append(row)
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Waktu_Masuk", "Waktu_Keluar", "Durasi", "Total_Orang"])
        writer.writerows(lines)

def get_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def get_full_log():
    rows = []
    with open(log_path, "r") as f:
        reader = list(csv.reader(f))
        header = reader[0]
        data_rows = reader[1:]

        data_rows.reverse()  # Biar terbaru di atas

        # Hitung jumlah orang per tanggal
        count_per_day = defaultdict(int)
        seen_ids_per_day = defaultdict(set)

        for row in data_rows:
            date_str = row[1].split()[0] if row[1] else ""
            pid = row[0]
            if pid not in seen_ids_per_day[date_str]:
                seen_ids_per_day[date_str].add(pid)
                count_per_day[date_str] += 1

        for i, row in enumerate(data_rows, start=1):
            rows.append({
                "no": i,
                "id": row[0],
                "masuk": row[1],
                "keluar": row[2],
                "durasi": row[3],
                "total": row[4] if len(row) > 4 else "-"
            })

    return rows
#!/usr/bin/env python3
"""
Smart Door Lock  --  Raspberry Pi
Face recognition + Web server

Run:   python3 smart_door_lock.py [/dev/ttyACM0]
Web:   http://<pi-ip>:8080   login: admin / 1234

Install dependencies:
    pip install pyserial numpy insightface onnxruntime opencv-python --break-system-packages
    pip install picamera2 --break-system-packages
"""

import os, sys, time, signal, json, base64, threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime

import numpy as np
import serial

# ============================================================
#  CONFIGURATION
# ============================================================
SERIAL_PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"
SERIAL_BAUD = 9600
FACES_DIR   = "data/faces"
LOG_FILE    = "data/log.txt"
CAM_WIDTH   = 640
CAM_HEIGHT  = 480
TOLERANCE   = 0.50   # cosine similarity threshold (0.3 strict -- 0.7 lenient)
FRAMES      = 6      # frames to sample during recognition
MIN_VOTES   = 2      # minimum matching frames to grant access
WEB_PORT    = 7000
WEB_USER    = "admin"
WEB_PASS    = "1234"    # change this
# ============================================================

os.makedirs("data",     exist_ok=True)
os.makedirs(FACES_DIR,  exist_ok=True)

# ============================================================
#  LOGGING
# ============================================================
_log = []   # in-memory list of dicts: {ts, method, who, result}

def log_entry(method, who, result):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {"ts": ts, "method": method, "who": who, "result": result}
    _log.append(entry)
    if len(_log) > 200:
        _log.pop(0)
    with open(LOG_FILE, "a") as f:
        f.write(f"{ts} | {method} | {who} | {result}\n")
    print(f"[LOG] {ts} | {method} | {who} | {result}")

# Load existing log on startup
if os.path.exists(LOG_FILE):
    with open(LOG_FILE) as f:
        for line in f.readlines()[-200:]:
            parts = line.strip().split(" | ")
            if len(parts) == 4:
                _log.append({
                    "ts":     parts[0],
                    "method": parts[1],
                    "who":    parts[2],
                    "result": parts[3]
                })

# ============================================================
#  DOOR STATE
# ============================================================
_door_locked = True

def set_locked(v):
    global _door_locked
    _door_locked = v

# ============================================================
#  FACE RECOGNITION  --  InsightFace + picamera2
# ============================================================
FACE_OK = False
CAM_OK  = False

try:
    from insightface.app import FaceAnalysis
    import cv2
    _fa = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
    _fa.prepare(ctx_id=0, det_size=(320, 320))
    FACE_OK = True
    print("[FACE] InsightFace ready")
except Exception as e:
    print(f"[FACE] Disabled: {e}")

try:
    from picamera2 import Picamera2
    CAM_OK = True
    print("[CAM] picamera2 available")
except Exception as e:
    print(f"[CAM] Disabled: {e}")


def _open_camera():
    """Open camera, return Picamera2 instance."""
    cam = Picamera2()
    cfg = cam.create_still_configuration(
        main={"size": (CAM_WIDTH, CAM_HEIGHT), "format": "RGB888"}
    )
    cam.configure(cfg)
    cam.start()
    time.sleep(0.8)   # warm-up
    return cam


def _get_embedding(frame_rgb):
    """
    Return (embedding, face_count) for the given RGB frame.
    embedding is None if 0 or more than 1 face detected.
    """
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    faces = _fa.get(bgr)
    n = len(faces)
    if n != 1:
        return None, n
    return faces[0].normed_embedding, 1


def _cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def _load_known():
    """Return dict: {name: [embedding, ...]}"""
    known = {}
    for fname in os.listdir(FACES_DIR):
        if fname.endswith(".npy"):
            name = fname[:-4]
            known[name] = list(np.load(
                os.path.join(FACES_DIR, fname), allow_pickle=True
            ))
    return known


def face_recognize():
    """
    Capture FRAMES frames, vote on best match.
    Returns matched name string, or None.
    """
    if not (FACE_OK and CAM_OK):
        return None
    known = _load_known()
    if not known:
        return None

    cam = _open_camera()
    votes = {}
    try:
        for _ in range(FRAMES):
            frame = cam.capture_array()
            emb, n = _get_embedding(frame)
            if emb is None:
                time.sleep(0.15)
                continue
            for name, encs in known.items():
                score = max(_cosine(emb, e) for e in encs)
                if score >= TOLERANCE:
                    votes[name] = votes.get(name, 0) + 1
            time.sleep(0.15)
    finally:
        cam.stop()
        cam.close()

    if not votes:
        return None
    best = max(votes, key=votes.get)
    return best if votes[best] >= MIN_VOTES else None


def face_enroll(name, progress_cb, total_samples=10):
    """
    Capture total_samples face embeddings and save to FACES_DIR/<name>.npy
    progress_cb(msg) is called with serial-style MSG strings to show on LCD.
    This function is blocking -- run it in a thread.
    """
    if not (FACE_OK and CAM_OK):
        progress_cb("MSG:No camera|or face engine")
        return

    cam = _open_camera()
    collected = []
    attempts  = 0
    max_att   = total_samples * 6

    try:
        while len(collected) < total_samples and attempts < max_att:
            attempts += 1
            frame = cam.capture_array()
            emb, n = _get_embedding(frame)
            if n == 0:
                progress_cb("MSG:No face seen|move closer")
                time.sleep(0.3)
                continue
            if n > 1:
                progress_cb("MSG:Multiple faces|only 1 person")
                time.sleep(0.3)
                continue
            collected.append(emb)
            progress_cb(f"MSG:Sample {len(collected)}/{total_samples}|hold still")
            time.sleep(0.25)
    finally:
        cam.stop()
        cam.close()

    if len(collected) < (total_samples // 2):
        progress_cb("MSG:Enroll failed|try again")
        return

    save_path = os.path.join(FACES_DIR, f"{name}.npy")
    np.save(save_path, np.array(collected))
    progress_cb(f"MSG:Enrolled OK|{name[:12]}")
    log_entry("ENROLL", name, "OK")


# ============================================================
#  SERIAL COMMUNICATION
# ============================================================
_ser = None   # set in main

def send(msg):
    """Send a line to Arduino."""
    if _ser and _ser.is_open:
        line = msg.strip() + "\n"
        _ser.write(line.encode())
        print(f"[TX] {msg.strip()}")


def dispatch(line):
    """Handle a line received from Arduino."""
    print(f"[RX] {line}")

    if line == "FACE":
        # Run recognition in background thread so serial loop stays alive
        def _run():
            name = face_recognize()
            if name:
                log_entry("FACE", name, "OK")
                set_locked(False)
                send(f"GRANT:{name}")
            else:
                log_entry("FACE", "unknown", "DENY")
                send("DENY:Not recognised")
        threading.Thread(target=_run, daemon=True).start()

    elif line.startswith("ENROLL:"):
        name = line[7:].strip()
        if not name:
            send("MSG:Empty name|try again")
            return
        def _run():
            face_enroll(name, send)
        threading.Thread(target=_run, daemon=True).start()

    elif line.startswith("DELFACE:"):
        name = line[8:].strip()
        path = os.path.join(FACES_DIR, f"{name}.npy")
        if os.path.exists(path):
            os.remove(path)
            send(f"MSG:Deleted|{name[:12]}")
            log_entry("DELETE", name, "OK")
        else:
            send(f"MSG:Not found|{name[:12]}")

    elif line == "GETLOG":
        recent = _log[-5:]
        if not recent:
            send("MSG:Log is empty|")
            return
        for entry in reversed(recent):
            msg = f"MSG:{entry['result']} {entry['method']}|{entry['who'][:8]} {entry['ts'][11:16]}"
            send(msg)
            time.sleep(2.8)   # wait for LCD to display

    elif line.startswith("LOG:"):
        parts = line[4:].split(",", 2)
        if len(parts) == 3:
            log_entry(parts[0], parts[1], parts[2])

    elif line.startswith("STATUS:"):
        set_locked(line[7:] == "LOCKED")

    elif line == "PING":
        send("PONG")

    elif line.startswith("UID:"):
        print(f"[UID] {line[4:]}")   # just log it; Arduino already shows it on LCD


# ============================================================
#  WEB UI  --  clean light theme, no emojis
# ============================================================
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Smart Door Lock</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}
:root{
  --bg:#f1f5f9;--card:#fff;--border:#e2e8f0;--text:#0f172a;--muted:#64748b;
  --green:#15803d;--green-light:#dcfce7;--green-border:#86efac;
  --red:#b91c1c;  --red-light:#fee2e2;  --red-border:#fca5a5;
  --blue:#1d4ed8; --blue-light:#dbeafe; --blue-border:#93c5fd;
  --amber:#b45309;--amber-light:#fef3c7;--amber-border:#fcd34d;
  --radius:10px;--shadow:0 1px 3px rgba(0,0,0,.07),0 1px 2px rgba(0,0,0,.05);
}
body{background:var(--bg);color:var(--text);min-height:100vh}

/* Header */
header{
  background:#fff;border-bottom:1px solid var(--border);
  padding:13px 24px;display:flex;align-items:center;
  justify-content:space-between;position:sticky;top:0;z-index:50;
  box-shadow:0 1px 4px rgba(0,0,0,.06)
}
header h1{font-size:1em;font-weight:700;letter-spacing:-.01em}
.hdr-right{display:flex;align-items:center;gap:10px;font-size:.82em}
.conn{width:8px;height:8px;border-radius:50%;background:#22c55e;
  box-shadow:0 0 0 2px #bbf7d0;flex-shrink:0}
.conn.off{background:#ef4444;box-shadow:0 0 0 2px #fecaca}
.badge{padding:3px 10px;border-radius:20px;font-weight:700;font-size:.78em}
.badge.locked{background:var(--red-light);color:var(--red)}
.badge.unlocked{background:var(--green-light);color:var(--green)}

/* Layout */
.page{max-width:960px;margin:0 auto;padding:20px;
  display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:620px){.page{grid-template-columns:1fr;padding:12px}}
.span2{grid-column:1/-1}

/* Cards */
.card{background:var(--card);border:1px solid var(--border);
  border-radius:var(--radius);box-shadow:var(--shadow);overflow:hidden}
.card-head{padding:12px 16px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between}
.card-head h2{font-size:.78em;font-weight:700;text-transform:uppercase;
  letter-spacing:.07em;color:var(--muted)}
.card-body{padding:16px}

/* Status ring */
.status-wrap{text-align:center;padding:16px 0 12px}
.ring{width:88px;height:88px;border-radius:50%;margin:0 auto 12px;
  display:flex;align-items:center;justify-content:center;
  font-size:1.6em;font-weight:900;border:4px solid;letter-spacing:.05em;
  transition:all .4s}
.ring.locked{border-color:var(--red);background:var(--red-light);color:var(--red)}
.ring.unlocked{border-color:var(--green);background:var(--green-light);color:var(--green)}
.status-label{font-size:1.4em;font-weight:800;letter-spacing:.04em;margin-bottom:3px}
.status-label.locked{color:var(--red)}
.status-label.unlocked{color:var(--green)}
.status-hint{font-size:.8em;color:var(--muted)}
.status-line{height:3px;border-radius:2px;margin-top:14px;transition:background .4s}
.status-line.locked{background:var(--red)}
.status-line.unlocked{background:var(--green)}

/* Buttons */
button{border:none;border-radius:8px;padding:9px 14px;font-size:.86em;
  font-weight:600;cursor:pointer;display:flex;align-items:center;
  justify-content:center;gap:6px;transition:all .12s;width:100%;
  letter-spacing:.01em}
button:hover:not(:disabled){filter:brightness(.95)}
button:active:not(:disabled){transform:scale(.96)}
button:disabled{opacity:.45;cursor:not-allowed}
.btn-green{background:var(--green-light);color:var(--green);border:1px solid var(--green-border)}
.btn-red{background:var(--red-light);color:var(--red);border:1px solid var(--red-border)}
.btn-blue{background:var(--blue-light);color:var(--blue);border:1px solid var(--blue-border)}
.btn-amber{background:var(--amber-light);color:var(--amber);border:1px solid var(--amber-border)}
.btn-gray{background:#f8fafc;color:var(--muted);border:1px solid var(--border)}
.btn-sm{padding:5px 12px;font-size:.78em;width:auto}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.g1{display:grid;gap:8px}

/* Inputs */
input,select{
  width:100%;padding:8px 12px;border:1px solid var(--border);
  border-radius:7px;font-size:.86em;color:var(--text);background:#fff;outline:none;
  transition:border-color .15s,box-shadow .15s
}
input:focus,select:focus{
  border-color:var(--blue);box-shadow:0 0 0 3px rgba(29,78,216,.1)
}
.irow{display:flex;gap:8px;margin-bottom:8px}
.irow input{flex:1}.irow button{width:auto;white-space:nowrap;padding:9px 14px}

/* Toggle */
.tog{position:relative;width:40px;height:22px;flex-shrink:0}
.tog input{opacity:0;width:0;height:0}
.tslide{position:absolute;inset:0;border-radius:22px;
  background:#cbd5e1;cursor:pointer;transition:.25s}
.tslide::before{content:'';position:absolute;
  height:16px;width:16px;left:3px;bottom:3px;
  background:#fff;border-radius:50%;transition:.25s;
  box-shadow:0 1px 3px rgba(0,0,0,.2)}
input:checked+.tslide{background:var(--green)}
input:checked+.tslide::before{transform:translateX(18px)}

/* Info rows */
.irow2{display:flex;align-items:center;justify-content:space-between;
  padding:9px 0;border-bottom:1px solid var(--border)}
.irow2:last-child{border-bottom:none}
.irow2 .label{font-size:.86em;font-weight:500}
.irow2 .sub{font-size:.76em;color:var(--muted)}

/* Pills */
.pill{display:inline-block;padding:2px 8px;border-radius:20px;
  font-size:.72em;font-weight:700;letter-spacing:.02em}
.pill-ok{background:var(--green-light);color:var(--green)}
.pill-err{background:var(--red-light);color:var(--red)}
.pill-info{background:var(--blue-light);color:var(--blue)}

/* Face tags */
.faces{display:flex;flex-wrap:wrap;gap:6px;min-height:30px;align-items:center;
  padding:4px 0}
.ftag{background:#f8fafc;border:1px solid var(--border);padding:3px 10px;
  border-radius:20px;font-size:.8em;display:flex;align-items:center;
  gap:5px;font-weight:500}
.fdel{cursor:pointer;color:var(--red);font-weight:700;line-height:1;font-size:1em}
.fdel:hover{color:#991b1b}
.empty{font-size:.8em;color:var(--muted);font-style:italic}

/* Progress bar */
.pbar-wrap{margin-top:10px;display:none}
.pbar-track{background:#f1f5f9;border-radius:4px;height:6px;overflow:hidden}
.pbar-fill{height:100%;background:var(--blue);width:0%;
  transition:width .4s;border-radius:4px}
.pbar-text{font-size:.76em;color:var(--muted);text-align:center;margin-top:4px}

/* Log */
.log-scroll{max-height:300px;overflow-y:auto}
.log-row{display:flex;align-items:center;gap:10px;padding:8px 0;
  border-bottom:1px solid var(--border);font-size:.81em}
.log-row:last-child{border-bottom:none}
.dot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
.d-ok{background:var(--green)}.d-deny{background:var(--red)}
.log-main{flex:1;min-width:0}
.log-who{font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.log-meta{color:var(--muted);font-size:.9em}
.log-time{color:var(--muted);font-size:.8em;text-align:right;white-space:nowrap}

/* Timed unlock */
.timed-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-top:8px}
.timer-cnt{font-family:monospace;font-size:1.3em;font-weight:700;
  text-align:center;letter-spacing:.08em;color:var(--blue);
  padding:4px 0;display:none}

/* Range slider */
input[type=range]{
  padding:0;height:6px;accent-color:var(--blue);cursor:pointer
}

/* Divider */
.divider{border:none;border-top:1px solid var(--border);margin:12px 0}

/* Spinner */
.spin{display:inline-block;width:12px;height:12px;
  border:2px solid currentColor;border-right-color:transparent;
  border-radius:50%;animation:sp .6s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}

/* Toast */
#toast{
  position:fixed;bottom:20px;left:50%;
  transform:translateX(-50%) translateY(14px);
  background:#1e293b;color:#f8fafc;
  padding:9px 20px;border-radius:8px;font-size:.85em;
  opacity:0;transition:all .22s;pointer-events:none;z-index:200;
  box-shadow:0 4px 12px rgba(0,0,0,.25);white-space:nowrap
}
#toast.show{opacity:1;transform:translateX(-50%) translateY(0)}

/* Tabs */
.tabs{display:flex;gap:3px;margin-bottom:12px}
.tab{padding:5px 12px;border-radius:6px;font-size:.8em;font-weight:600;
  cursor:pointer;color:var(--muted);background:none;
  border:1px solid transparent;width:auto;transition:.12s}
.tab.active{background:var(--blue-light);color:var(--blue);border-color:var(--blue-border)}
.tabpane{display:none}.tabpane.active{display:block}
</style>
</head>
<body>

<header>
  <h1>Smart Door Lock</h1>
  <div class="hdr-right">
    <div class="conn" id="conn" title="Connection status"></div>
    <span class="badge locked" id="hdr-badge">LOCKED</span>
  </div>
</header>

<div class="page">

  <!-- STATUS -->
  <div class="card">
    <div class="card-head">
      <h2>Door Status</h2>
      <span id="last-seen" style="font-size:.74em;color:var(--muted)">--</span>
    </div>
    <div class="card-body">
      <div class="status-wrap">
        <div class="ring locked" id="ring">LOCKED</div>
        <div class="status-label locked" id="s-label">LOCKED</div>
        <div class="status-hint" id="s-hint">Door is secured</div>
      </div>
      <div class="status-line locked" id="s-line"></div>
    </div>
  </div>

  <!-- CONTROLS -->
  <div class="card">
    <div class="card-head"><h2>Controls</h2></div>
    <div class="card-body">
      <div class="g2" style="margin-bottom:8px">
        <button class="btn-green" onclick="cmd('unlock')">Unlock</button>
        <button class="btn-red"   onclick="cmd('lock')">Lock</button>
      </div>
      <div class="g2" style="margin-bottom:14px">
        <button class="btn-blue"  onclick="doFace()">Face Auth</button>
        <button class="btn-gray"  onclick="doRefresh()">Refresh</button>
      </div>

      <hr class="divider">
      <div style="font-size:.76em;font-weight:700;text-transform:uppercase;
        letter-spacing:.06em;color:var(--muted);margin-bottom:8px">Timed Unlock</div>
      <div class="timed-grid">
        <button class="btn-gray" onclick="timedUnlock(5)">5s</button>
        <button class="btn-gray" onclick="timedUnlock(15)">15s</button>
        <button class="btn-gray" onclick="timedUnlock(30)">30s</button>
        <button class="btn-gray" onclick="timedUnlock(60)">1m</button>
      </div>
      <div class="timer-cnt" id="timer-cnt">00:00</div>

      <hr class="divider">
      <div class="irow2" style="border-bottom:none;padding-bottom:0">
        <div><div class="label">Auto-refresh</div>
          <div class="sub">Update every 4 seconds</div></div>
        <label class="tog">
          <input type="checkbox" id="auto-ref" checked onchange="toggleAR()">
          <span class="tslide"></span>
        </label>
      </div>
    </div>
  </div>

  <!-- FACE RECOGNITION -->
  <div class="card">
    <div class="card-head"><h2>Face Recognition</h2></div>
    <div class="card-body">
      <div class="tabs">
        <button class="tab active" onclick="switchTab(event,'enroll')">Enroll</button>
        <button class="tab"        onclick="switchTab(event,'manage')">Manage</button>
      </div>

      <div class="tabpane active" id="tab-enroll">
        <div class="irow">
          <input id="ename" placeholder="Name (letters only)"
            maxlength="20" onkeydown="if(event.key==='Enter')doEnroll()">
          <button class="btn-blue" onclick="doEnroll()" id="enroll-btn">Enroll</button>
        </div>
        <div style="font-size:.76em;color:var(--muted)">
          Look directly at the camera. Capturing 10 samples, takes ~15 seconds.
        </div>
        <div class="pbar-wrap" id="pbar-wrap">
          <div class="pbar-track"><div class="pbar-fill" id="pbar"></div></div>
          <div class="pbar-text" id="pbar-text">Starting...</div>
        </div>
      </div>

      <div class="tabpane" id="tab-manage">
        <div style="font-size:.76em;font-weight:700;text-transform:uppercase;
          letter-spacing:.06em;color:var(--muted);margin-bottom:8px">
          Enrolled faces</div>
        <div class="faces" id="faces"><span class="empty">None enrolled</span></div>
        <hr class="divider">
        <div style="font-size:.76em;font-weight:700;text-transform:uppercase;
          letter-spacing:.06em;color:var(--muted);margin-bottom:6px">
          Match tolerance: <span id="tol-val" style="color:var(--blue)">0.50</span></div>
        <input type="range" id="tol-slider" min="30" max="75" value="50"
          onchange="setTol(this.value)">
        <div style="display:flex;justify-content:space-between;
          font-size:.72em;color:var(--muted);margin-top:2px">
          <span>Strict (0.30)</span><span>Lenient (0.75)</span>
        </div>
      </div>
    </div>
  </div>

  <!-- SYSTEM INFO -->
  <div class="card">
    <div class="card-head"><h2>System</h2></div>
    <div class="card-body">
      <div class="irow2">
        <div><div class="label">Face Engine</div>
          <div class="sub" id="face-sub">InsightFace</div></div>
        <span class="pill pill-ok" id="face-pill">OK</span>
      </div>
      <div class="irow2">
        <div><div class="label">Camera</div>
          <div class="sub" id="cam-sub">picamera2</div></div>
        <span class="pill pill-ok" id="cam-pill">OK</span>
      </div>
      <div class="irow2">
        <div><div class="label">Arduino Serial</div>
          <div class="sub" id="serial-sub">--</div></div>
        <span class="pill pill-ok" id="serial-pill">Online</span>
      </div>
      <div class="irow2">
        <div><div class="label">Enrolled Faces</div>
          <div class="sub">Total registered</div></div>
        <span style="font-weight:700" id="face-count">0</span>
      </div>
      <div class="irow2">
        <div><div class="label">Log Entries</div>
          <div class="sub">Total recorded</div></div>
        <span style="font-weight:700" id="log-total">0</span>
      </div>
    </div>
  </div>

  <!-- ACCESS LOG -->
  <div class="card span2">
    <div class="card-head">
      <div style="display:flex;align-items:center;gap:8px">
        <h2>Access Log</h2>
        <span style="background:#f1f5f9;border:1px solid var(--border);
          padding:1px 8px;border-radius:10px;font-size:.72em;
          font-weight:700;color:var(--muted)" id="log-count">0</span>
      </div>
      <div style="display:flex;gap:6px;align-items:center">
        <select id="log-filter" onchange="doRefresh()"
          style="width:auto;padding:4px 8px;font-size:.78em">
          <option value="all">All entries</option>
          <option value="OK">Granted only</option>
          <option value="DENY">Denied only</option>
          <option value="RFID">RFID</option>
          <option value="PIN">PIN</option>
          <option value="FACE">Face</option>
          <option value="WEB">Web</option>
        </select>
        <button class="btn-gray btn-sm" onclick="doClearLog()">Clear</button>
      </div>
    </div>
    <div class="card-body" style="padding:0 16px">
      <div class="log-scroll" id="log">
        <div style="padding:18px;text-align:center;color:var(--muted);font-size:.84em">
          No log entries yet</div>
      </div>
    </div>
  </div>

</div><!-- /page -->

<div id="toast"></div>

<script>
let autoRef = true;
let arTimer;
let timerInterval;
let timerSec = 0;

// Toast notification
function toast(msg, ms=2500) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), ms);
}

// API helper
async function api(path, body=null) {
  try {
    const opts = body
      ? { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body) }
      : {};
    const r = await fetch(path, opts);
    document.getElementById('conn').className = 'conn';
    return r.json();
  } catch(e) {
    document.getElementById('conn').className = 'conn off';
    toast('Connection error');
    return {};
  }
}

// Commands
async function cmd(action) {
  const d = await api('/api/cmd', {action});
  if (d.msg) toast(d.msg);
  doRefresh();
}

async function doFace() {
  toast('Face auth started -- check LCD', 3000);
  const d = await api('/api/cmd', {action:'face'});
  if (d.msg) toast(d.msg);
  setTimeout(doRefresh, 5000);
}

// Timed unlock
function timedUnlock(secs) {
  api('/api/cmd', {action:'unlock'});
  toast('Unlocked for ' + secs + ' seconds');
  timerSec = secs;
  const el = document.getElementById('timer-cnt');
  el.style.display = 'block';
  clearInterval(timerInterval);
  timerInterval = setInterval(() => {
    if (timerSec <= 0) {
      clearInterval(timerInterval);
      el.style.display = 'none';
      api('/api/cmd', {action:'lock'});
      doRefresh();
      return;
    }
    const m = String(Math.floor(timerSec / 60)).padStart(2, '0');
    const s = String(timerSec % 60).padStart(2, '0');
    el.textContent = m + ':' + s;
    timerSec--;
  }, 1000);
}

// Tabs
function switchTab(e, name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tabpane').forEach(t => t.classList.remove('active'));
  e.target.classList.add('active');
  document.getElementById('tab-' + name).classList.add('active');
}

// Face enroll
async function doEnroll() {
  const name = document.getElementById('ename').value.trim();
  if (!name) { toast('Enter a name first'); return; }

  const btn  = document.getElementById('enroll-btn');
  const wrap = document.getElementById('pbar-wrap');
  const bar  = document.getElementById('pbar');
  const txt  = document.getElementById('pbar-text');

  btn.innerHTML = '<span class="spin"></span> Enrolling...';
  btn.disabled  = true;
  wrap.style.display = 'block';
  bar.style.width = '5%';
  txt.textContent = 'Opening camera...';

  // Fake progress while backend works
  let pct = 5;
  const ticker = setInterval(() => {
    pct = Math.min(pct + 6, 92);
    bar.style.width = pct + '%';
    txt.textContent = 'Capturing samples... ' + Math.round(pct) + '%';
  }, 1600);

  const d = await api('/api/enroll', {name});

  clearInterval(ticker);
  bar.style.width = '100%';
  txt.textContent = d.msg || 'Done';
  setTimeout(() => { wrap.style.display = 'none'; }, 2500);

  btn.innerHTML = 'Enroll';
  btn.disabled  = false;
  document.getElementById('ename').value = '';
  toast(d.msg || 'Enrolled');
  doRefresh();
}

async function doDeleteFace(name) {
  if (!confirm('Delete face data for: ' + name + '?')) return;
  const d = await api('/api/delface', {name});
  toast(d.msg || 'Deleted');
  doRefresh();
}

async function doClearLog() {
  if (!confirm('Clear all log entries?')) return;
  await api('/api/clearlog', {});
  toast('Log cleared');
  doRefresh();
}

function setTol(v) {
  const val = (v / 100).toFixed(2);
  document.getElementById('tol-val').textContent = val;
  api('/api/settol', {tolerance: parseFloat(val)});
}

// Auto-refresh toggle
function toggleAR() {
  autoRef = document.getElementById('auto-ref').checked;
  if (autoRef) schedAR(); else clearTimeout(arTimer);
}
function schedAR() {
  clearTimeout(arTimer);
  if (autoRef) arTimer = setTimeout(doRefresh, 4000);
}

// Main refresh
async function doRefresh() {
  const d = await api('/api/status');
  if (!d || !('locked' in d)) { schedAR(); return; }

  // Header
  const hb = document.getElementById('hdr-badge');
  hb.textContent  = d.locked ? 'LOCKED' : 'UNLOCKED';
  hb.className    = 'badge ' + (d.locked ? 'locked' : 'unlocked');
  document.getElementById('last-seen').textContent =
    'Updated ' + new Date().toLocaleTimeString();

  // Status card
  const cls   = d.locked ? 'locked' : 'unlocked';
  const label = d.locked ? 'LOCKED' : 'UNLOCKED';
  const hint  = d.locked ? 'Door is secured' : 'Door is open';
  document.getElementById('ring').className     = 'ring ' + cls;
  document.getElementById('ring').textContent   = label;
  document.getElementById('s-label').className  = 'status-label ' + cls;
  document.getElementById('s-label').textContent = label;
  document.getElementById('s-hint').textContent  = hint;
  document.getElementById('s-line').className    = 'status-line ' + cls;

  // System
  const fp = document.getElementById('face-pill');
  fp.textContent = d.face_ok ? 'Online' : 'Offline';
  fp.className   = 'pill ' + (d.face_ok ? 'pill-ok' : 'pill-err');
  document.getElementById('face-sub').textContent =
    d.face_ok ? 'InsightFace + ONNX' : 'Not installed';

  const cp = document.getElementById('cam-pill');
  cp.textContent = d.cam_ok ? 'Online' : 'Offline';
  cp.className   = 'pill ' + (d.cam_ok ? 'pill-ok' : 'pill-err');
  document.getElementById('cam-sub').textContent =
    d.cam_ok ? 'picamera2 ready' : 'Not available';

  document.getElementById('serial-sub').textContent = d.serial_port;
  document.getElementById('face-count').textContent = d.faces ? d.faces.length : 0;
  document.getElementById('log-total').textContent  = d.log   ? d.log.length   : 0;

  // Tolerance slider sync
  if (d.tolerance != null) {
    document.getElementById('tol-slider').value = Math.round(d.tolerance * 100);
    document.getElementById('tol-val').textContent  = d.tolerance.toFixed(2);
  }

  // Faces
  const fe = document.getElementById('faces');
  if (d.faces && d.faces.length) {
    fe.innerHTML = d.faces.map(f =>
      `<div class="ftag">${f}
       <span class="fdel" onclick="doDeleteFace('${f}')" title="Remove">&times;</span>
       </div>`
    ).join('');
  } else {
    fe.innerHTML = '<span class="empty">None enrolled</span>';
  }

  // Log with filter
  const filter = document.getElementById('log-filter').value;
  let entries  = d.log || [];
  if (filter !== 'all') {
    entries = entries.filter(e => {
      if (filter === 'OK')   return e.result === 'OK' || e.result === 'UNLOCK';
      if (filter === 'DENY') return e.result === 'DENY';
      return e.method === filter || e.method.startsWith(filter);
    });
  }

  document.getElementById('log-count').textContent = entries.length;
  const le = document.getElementById('log');
  if (entries.length) {
    le.innerHTML = entries.map(e => {
      const ok  = e.result === 'OK' || e.result === 'UNLOCK';
      const dot = ok ? 'd-ok' : 'd-deny';
      const pill = ok ? 'pill-ok' : 'pill-err';
      return `<div class="log-row">
        <div class="dot ${dot}"></div>
        <div class="log-main">
          <div class="log-who">${e.who}
            <span class="pill ${pill}">${e.result}</span>
          </div>
          <div class="log-meta">${e.method}</div>
        </div>
        <div class="log-time">
          ${e.ts.slice(11,16)}<br>
          <span style="color:#94a3b8;font-size:.9em">${e.ts.slice(5,10)}</span>
        </div>
      </div>`;
    }).join('');
  } else {
    le.innerHTML =
      '<div style="padding:18px;text-align:center;color:var(--muted);font-size:.84em">No entries</div>';
  }

  schedAR();
}

doRefresh();
</script>
</body>
</html>"""


# ============================================================
#  WEB REQUEST HANDLER
# ============================================================
class Handler(BaseHTTPRequestHandler):

    def log_message(self, *args):
        pass   # silence default access log

    def _check_auth(self):
        auth = self.headers.get("Authorization", "")
        if not auth.startswith("Basic "):
            return False
        try:
            decoded = base64.b64decode(auth[6:]).decode()
            return decoded == f"{WEB_USER}:{WEB_PASS}"
        except Exception:
            return False

    def _send_auth_challenge(self):
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="Door Lock"')
        self.end_headers()

    def _send_json(self, data, code=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self._check_auth():
            self._send_auth_challenge()
            return
        body = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        global TOLERANCE
        if not self._check_auth():
            self._send_auth_challenge()
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            body = {}

        path = self.path

        if path == "/api/status":
            faces = [f[:-4] for f in os.listdir(FACES_DIR) if f.endswith(".npy")]
            self._send_json({
                "locked":      _door_locked,
                "faces":       sorted(faces),
                "log":         list(reversed(_log[-50:])),
                "face_ok":     FACE_OK,
                "cam_ok":      CAM_OK,
                "serial_port": SERIAL_PORT,
                "tolerance":   TOLERANCE
            })

        elif path == "/api/cmd":
            action = body.get("action", "")
            if action == "unlock":
                send("CMD:UNLOCK")
                set_locked(False)
                log_entry("WEB", "remote", "UNLOCK")
                self._send_json({"msg": "Door unlocked"})
            elif action == "lock":
                send("CMD:LOCK")
                set_locked(True)
                log_entry("WEB", "remote", "LOCK")
                self._send_json({"msg": "Door locked"})
            elif action == "face":
                def _run():
                    name = face_recognize()
                    if name:
                        log_entry("WEB-FACE", name, "OK")
                        set_locked(False)
                        send(f"GRANT:{name}")
                    else:
                        log_entry("WEB-FACE", "unknown", "DENY")
                        send("DENY:Not recognised")
                threading.Thread(target=_run, daemon=True).start()
                self._send_json({"msg": "Face auth started -- check LCD"})
            elif action == "status":
                send("CMD:STATUS")
                self._send_json({"msg": "Status request sent"})
            else:
                self._send_json({"msg": "Unknown action"}, 400)

        elif path == "/api/enroll":
            name = body.get("name", "").strip()
            if not name:
                self._send_json({"msg": "No name provided"}, 400)
                return
            if not (FACE_OK and CAM_OK):
                self._send_json({"msg": "Face engine or camera not available"})
                return
            # Run synchronously here -- the HTTP client waits for the result.
            # This is fine because face_enroll calls progress_cb to update the LCD
            # via serial in real time, and the web UI shows a fake progress bar.
            result_holder = []
            def _cb(msg):
                send(msg)              # update LCD via serial
                result_holder.append(msg)
            face_enroll(name, _cb)
            last = result_holder[-1] if result_holder else "Done"
            msg_clean = last.replace("MSG:", "").replace("|", " -- ")
            self._send_json({"msg": msg_clean})

        elif path == "/api/delface":
            name = body.get("name", "").strip()
            path2 = os.path.join(FACES_DIR, f"{name}.npy")
            if os.path.exists(path2):
                os.remove(path2)
                log_entry("DELETE", name, "OK")
                self._send_json({"msg": f"Deleted {name}"})
            else:
                self._send_json({"msg": f"Not found: {name}"})

        elif path == "/api/clearlog":
            _log.clear()
            open(LOG_FILE, "w").close()
            self._send_json({"msg": "Log cleared"})

        elif path == "/api/settol":
            try:
                val = float(body.get("tolerance", TOLERANCE))
            except (TypeError, ValueError):
                val = TOLERANCE

            TOLERANCE = max(0.30, min(0.75, val))
            self._send_json({"msg": f"Tolerance set to {TOLERANCE:.2f}"})
    
        else:
            self._send_json({"msg": "Not found"}, 404)


def start_web_server():
    server = HTTPServer(("0.0.0.0", WEB_PORT), Handler)
    print(f"[WEB] Listening on http://0.0.0.0:{WEB_PORT}  ({WEB_USER}/{WEB_PASS})")
    server.serve_forever()


# ============================================================
#  MAIN
# ============================================================
def main():
    global _ser

    print(f"[SERIAL] Opening {SERIAL_PORT} at {SERIAL_BAUD}...")
    try:
        _ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
    except Exception as e:
        print(f"[SERIAL] ERROR: {e}")
        print("         Check the port. Try:  ls /dev/ttyACM*  or  ls /dev/ttyUSB*")
        sys.exit(1)

    time.sleep(2)   # wait for Arduino reset after serial open
    print("[SERIAL] Ready")

    threading.Thread(target=start_web_server, daemon=True).start()

    def _shutdown(sig, frame):
        print("\n[EXIT] Shutting down...")
        if _ser and _ser.is_open:
            _ser.close()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print("[READY] Waiting for Arduino messages...")
    while True:
        try:
            raw = _ser.readline()
            if raw:
                line = raw.decode(errors="ignore").strip()
                if line:
                    dispatch(line)
        except serial.SerialException as e:
            print(f"[SERIAL] Lost connection: {e}")
            time.sleep(2)
        time.sleep(0.02)


if __name__ == "__main__":
    main()

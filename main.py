import cv2
import time
import os
import smtplib
import threading
from email.message import EmailMessage
import sqlite3
from datetime import datetime

# Import your custom computer vision modules
from eye_movement import process_eye_movement
from head_pose import process_head_pose, reset_head_pose_state # Import reset function
from mobile_detection import process_mobile_detection

# --- Configuration ---
# IMPORTANT: Replace with your actual email and app password for sending alerts
# For security, avoid hardcoding sensitive info directly in production.
# Consider environment variables or a config file for real deployments.
EMAIL_ADDRESS = "your_email@gmail.com"  # <<< Your Gmail address
EMAIL_PASSWORD = "your_app_password"     # <<< Your Gmail App Password
ALERT_RECIPIENT = "recipient_email@example.com" # <<< Email address to receive alerts

# Database and Logging
DB_NAME = "surveillance_logs.db"
LOG_DIR = "log"
os.makedirs(LOG_DIR, exist_ok=True) # Ensure log directory exists

# Webcam setup
cap = cv2.VideoCapture(0) # 0 for default webcam
# Set a standard resolution that's generally supported and good for CV tasks
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30) # Request 30 frames per second

# Check if webcam opened successfully
if not cap.isOpened():
    print("[ERROR] Cannot access webcam. Please ensure it's not in use by another application and drivers are installed.")
    print("       Exiting application.")
    exit()

# --- Calibration and Violation Timers ---
# Head Pose Calibration
calibrated_angles = None
calibration_samples = []
calibration_done = False
start_time = None # Will be set when calibration starts or is initiated
CALIBRATION_TIME = 10  # Seconds for calibration (keep head straight)

# Violation Timers (to prevent immediate, flickering alerts)
head_misalignment_start_time = None
eye_misalignment_start_time = None
mobile_detection_start_time = None

# Current states (updated by CV functions, used for UI display and logic)
head_direction = "N/A"
gaze_direction = "N/A"
mobile_detected = False

# Email cooldown (global variable, correctly accessed using 'global')
last_email_time = 0
EMAIL_COOLDOWN = 5 # seconds between emails of the same type (per type of alert, or overall?)
                   # Current implementation is per type via the send_email_alert_async call.

# Violation duration threshold
VIOLATION_THRESHOLD_SECONDS = 3 # How long an anomaly must persist to trigger an alert

# --- Define Valid States (Crucial: These MUST match outputs from your CV modules) ---
# Ensure these lists accurately reflect the string outputs from your `eye_movement.py`
# and `head_pose.py` functions for non-face-detected and specific abnormal states.

# Based on the provided `head_pose.py` logic:
VALID_HEAD_STATES = [
    "Looking Left", "Looking Right", "Looking Up", "Looking Down", "Looking Tilted",
    "Looking Up and Left", "Looking Up and Right", "Looking Down and Left", "Looking Down and Right"
]
# For eye_movement.py, if it outputs "Looking Left", "Looking Right", "Looking Up",
# "Looking Down", "Looking Center", "Face Not Detected", etc.
# IMPORTANT: Adjust this based on what your process_eye_movement function *actually* outputs.
VALID_EYE_STATES = [
    "Looking Left", "Looking Right", "Looking Up", "Looking Down", "Looking Center"
    # Add other specific eye states if your eye_movement.py generates them, e.g., "Eyes Closed"
]


# --- Database Integration Functions ---
def init_db():
    """Initializes the SQLite database and creates the alerts table if it doesn't exist."""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                description TEXT,
                screenshot_path TEXT,
                emailed BOOLEAN NOT NULL DEFAULT 0
            )
        ''')
        conn.commit()
        print(f"[DB] Database '{DB_NAME}' initialized successfully.")
    except sqlite3.Error as e:
        print(f"[DB ERROR] Error initializing database: {e}")
    finally:
        if conn:
            conn.close()

def log_alert_to_db(alert_type, description, screenshot_path, emailed=False):
    """Logs an alert to the SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO alerts (timestamp, alert_type, description, screenshot_path, emailed) VALUES (?, ?, ?, ?, ?)",
                       (timestamp, alert_type, description, screenshot_path, emailed))
        conn.commit()
        print(f"[DB] Logged alert: Type='{alert_type}', Desc='{description}'")
    except sqlite3.Error as e:
        print(f"[DB ERROR] Error logging alert to database: {e}")
    finally:
        if conn:
            conn.close()

# Initialize the database when the script starts
init_db()

# --- Asynchronous Email Sending ---
def send_email_alert_async(filename, alert_type, description):
    """Sends an email alert asynchronously to avoid freezing the main video feed."""
    # Note: last_email_time is managed globally for a simple cooldown across all email types.
    # For more granular control (e.g., separate cooldowns per alert_type),
    # last_email_time could be a dictionary: `last_email_time = {"eye": 0, "head": 0, "mobile": 0}`
    # and updated like `last_email_time[alert_type] = now`.
    global last_email_time # Access the global cooldown timer

    def send_email_task():
        # Correctly access the global variable `last_email_time`
        global last_email_time # <<< FIX: Changed from nonlocal to global

        now = time.time()

        # Check cooldown
        if now - last_email_time < EMAIL_COOLDOWN:
            print(f"[EMAIL SKIPPED] Cooldown active for {alert_type}. Next email allowed in {EMAIL_COOLDOWN - (now - last_email_time):.1f}s.")
            return

        # If not on cooldown, update the last email time *before* sending
        # This prevents multiple threads from trying to send at the same time if they pass the initial check
        last_email_time = now

        try:
            subject = f"[ALERT] {alert_type}"
            body = f"{description}. See attached image: {os.path.basename(filename)}"

            msg = EmailMessage()
            msg["From"] = EMAIL_ADDRESS
            msg["To"] = ALERT_RECIPIENT
            msg["Subject"] = subject
            msg.set_content(body)

            # Add attachment
            if os.path.exists(filename):
                with open(filename, "rb") as img_file:
                    msg.add_attachment(img_file.read(), maintype="image", subtype="png", filename=os.path.basename(filename))
            else:
                print(f"[EMAIL WARNING] Screenshot file not found: {filename}")
                msg.add_attachment(f"Screenshot file not found: {os.path.basename(filename)}", maintype="text", subtype="plain")


            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                smtp.send_message(msg)

            print(f"[EMAIL SENT] {subject}")
            log_alert_to_db(alert_type, description, filename, emailed=True)

        except Exception as e:
            print(f"[EMAIL ERROR] Failed to send email for {alert_type}: {e}")
            log_alert_to_db(alert_type, description, filename, emailed=False) # Log even if email fails

    email_thread = threading.Thread(target=send_email_task)
    email_thread.daemon = True # Allows the main program to exit even if thread is running
    email_thread.start()

# --- UI Drawing Functions for OpenCV Window ---
def draw_status_panel(frame):
    """Draws a status panel in the top-right corner of the OpenCV frame."""
    height, width = frame.shape[:2]

    panel_width = 400
    panel_height = 180
    panel_x = width - panel_width - 10
    panel_y = 10

    # Create semi-transparent overlay for the panel background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame) # Blend overlay with frame

    text_x = panel_x + 10
    cv2.putText(frame, "SURVEILLANCE STATUS", (text_x, panel_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.line(frame, (text_x, panel_y + 30), (panel_x + panel_width - 20, panel_y + 30), (255, 255, 255), 1)

    # Status indicators
    status_y = panel_y + 50
    # Adjusted check for eye/head tracking based on new outputs
    # Assume 'N/A' or 'Face Not Detected' means tracking is not active/successful
    eye_tracking_active = (gaze_direction != 'N/A' and gaze_direction != 'Face Not Detected')
    head_tracking_active = (head_direction != 'N/A' and head_direction != 'Face Not Detected' and head_direction != 'Calibrating')

    cv2.putText(frame, f"Eye Tracking: {'ACTIVE' if eye_tracking_active else 'INACTIVE'}",
                (text_x, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if eye_tracking_active else (0, 0, 255), 1)

    status_y += 25
    cv2.putText(frame, f"Head Tracking: {'ACTIVE' if head_tracking_active else 'INACTIVE'}",
                (text_x, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if head_tracking_active else (0, 0, 255), 1)

    status_y += 25
    # Overall violation check
    # A violation occurs if:
    # 1. Gaze is in VALID_EYE_STATES but not "Looking Center" or "Face Not Detected"
    # 2. Head is in VALID_HEAD_STATES but not "Looking at Screen", "Calibrating", or "Face Not Detected"
    # 3. Mobile device is detected
    is_violation = (gaze_direction in VALID_EYE_STATES and gaze_direction not in ["Looking Center", "Face Not Detected"]) or \
                   (head_direction in VALID_HEAD_STATES and head_direction not in ["Looking at Screen", "Calibrating", "Face Not Detected"]) or \
                   mobile_detected
    violation_color = (0, 0, 255) if is_violation else (0, 255, 0)
    cv2.putText(frame, f"Overall Violation: {'DETECTED' if is_violation else 'NONE'}",
                (text_x, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, violation_color, 1)


    # Current anomaly details
    status_y += 30
    cv2.putText(frame, "CURRENT ANOMALIES:", (text_x, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    status_y += 20
    # Display specific anomalies
    if gaze_direction not in ["Looking Center", "N/A", "Face Not Detected"]:
        cv2.putText(frame, f"- Eye: {gaze_direction}", (text_x, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)
        status_y += 15

    if head_direction not in ["Looking at Screen", "N/A", "Calibrating", "Face Not Detected"]:
        cv2.putText(frame, f"- Head: {head_direction}", (text_x, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)
        status_y += 15

    if mobile_detected:
        cv2.putText(frame, f"- Mobile: DETECTED", (text_x, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)
        status_y += 15

    # Calibration status message
    if not calibration_done:
        if start_time is not None and (time.time() - start_time) <= CALIBRATION_TIME:
            remaining_time = max(0, CALIBRATION_TIME - (time.time() - start_time))
            cv2.putText(frame, f"Calibrating: {remaining_time:.1f}s remaining",
                        (text_x, panel_y + panel_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        else: # Calibration needed or failed
            cv2.putText(frame, "Calibration Needed (Press 'r')",
                        (text_x, panel_y + panel_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1) # Orange for warning


# --- Main Video Processing Loop ---
print("[INFO] Starting Enhanced Surveillance System (Press 'q' to quit, 'r' to recalibrate)")

# Initial calibration prompt or setup
# If you want to force calibration on startup, set start_time here.
# If you want to wait for 'r' key press, leave start_time as None initially.
# Let's make it start calibration automatically on startup for user convenience.
start_time = time.time()
print(f"[INFO] Calibration started. Look straight at the camera for {CALIBRATION_TIME} seconds.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame. Exiting.")
        break

    current_time = time.time() # Current time for this frame

    # 1. Process Eye Movement
    frame, gaze_direction_current = process_eye_movement(frame)
    gaze_direction = gaze_direction_current # Update global for UI display
    cv2.putText(frame, f"GAZE: {gaze_direction}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # 2. Head Pose Detection with calibration logic
    head_direction_current = "N/A" # Default for this frame
    if not calibration_done and start_time is not None and (current_time - start_time) <= CALIBRATION_TIME:
        # During calibration phase
        remaining_time = CALIBRATION_TIME - (current_time - start_time)
        progress = (current_time - start_time) / CALIBRATION_TIME

        # Draw calibration progress bar on the frame
        bar_width = 400
        bar_height = 20
        bar_x = (frame.shape[1] - bar_width) // 2
        bar_y = 200

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
        cv2.rectangle(frame, (bar_x + 2, bar_y + 2),
                      (bar_x + int((bar_width - 4) * progress), bar_y + bar_height - 2), (0, 255, 0), -1)

        cv2.putText(frame, "CALIBRATING - Keep your head straight and look at screen",
                    (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Time remaining: {remaining_time:.1f}s",
                    (bar_x + 120, bar_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        # Call process_head_pose for calibration: it returns (frame, "Calibrating", angles_tuple)
        frame, _, current_angles = process_head_pose(frame, None)
        head_direction_current = "Calibrating" # Explicitly set status for UI

        if current_angles is not None:
            calibration_samples.append(current_angles)
            # A rough estimate assuming 30 FPS. Can be more precise if you actually count frames.
            cv2.putText(frame, f"Samples collected: {len(calibration_samples)}/{int(cap.get(cv2.CAP_PROP_FPS) * CALIBRATION_TIME)}",
                        (bar_x + 100, bar_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Check if calibration is complete (time elapsed AND sufficient samples)
        # Using a minimum of 80% of expected frames at the current FPS
        min_samples_required = int(cap.get(cv2.CAP_PROP_FPS) * CALIBRATION_TIME * 0.8)
        if len(calibration_samples) >= min_samples_required and (current_time - start_time) >= CALIBRATION_TIME:
            avg_pitch = sum(a[0] for a in calibration_samples) / len(calibration_samples)
            avg_yaw   = sum(a[1] for a in calibration_samples) / len(calibration_samples)
            avg_roll  = sum(a[2] for a in calibration_samples) / len(calibration_samples)
            calibrated_angles = (avg_pitch, avg_yaw, avg_roll)
            calibration_done = True
            print(f"[INFO] Calibration complete: Pitch={avg_pitch:.1f}°, Yaw={avg_yaw:.1f}°, Roll={avg_roll:.1f}°")
            # Reset the head pose state machine after calibration is done to avoid old states from calibration phase
            reset_head_pose_state()
            # Clear calibration samples
            calibration_samples = [] # Reset for potential future recalibrations
            start_time = None # Clear start_time to indicate calibration finished

    else:
        # Normal operation after calibration, or if calibration failed (e.g., no face detected)
        if not calibration_done and (start_time is not None and (current_time - start_time) > CALIBRATION_TIME):
            # If calibration window passed but it's not marked done
            print("[WARNING] Calibration window expired or failed to collect enough samples. Please recalibrate (press 'r').")
            head_direction_current = "Calibration Failed" # Display status
            calibrated_angles = None # Ensure it's explicitly reset
            start_time = None # Clear start time so it doesn't try to continue calibration
            calibration_samples = [] # Clear any partial samples

        # Call process_head_pose for monitoring: it returns (frame, head_direction_string, angles_tuple)
        frame, head_direction_current, _ = process_head_pose(frame, calibrated_angles if calibration_done else None)

        cv2.putText(frame, f"HEAD: {head_direction_current}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    head_direction = head_direction_current # Update global for UI display

    # 3. Mobile Detection
    frame, mobile_detected_current = process_mobile_detection(frame)
    mobile_detected = mobile_detected_current # Update global for UI display
    mobile_color = (0, 0, 255) if mobile_detected else (0, 255, 0)
    cv2.putText(frame, f"MOBILE: {'DETECTED' if mobile_detected else 'CLEAR'}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, mobile_color, 2)

    # Draw the status panel on the frame
    draw_status_panel(frame)

    # --- Violation Detection Logic ---
    # Eye Violation Check
    if gaze_direction in VALID_EYE_STATES and gaze_direction != "Looking Center" and gaze_direction != "Face Not Detected":
        if eye_misalignment_start_time is None:
            eye_misalignment_start_time = current_time
        elif current_time - eye_misalignment_start_time >= VIOLATION_THRESHOLD_SECONDS:
            filename = os.path.join(LOG_DIR, f"eye_{gaze_direction.replace(' ', '_')}_{int(current_time)}.png")
            cv2.imwrite(filename, frame)
            description = f"Eye violation detected: {gaze_direction}"
            send_email_alert_async(filename, "Eye Violation", description)
            print(f"[ALERT] {description}")
            eye_misalignment_start_time = None # Reset timer after alert
    else:
        eye_misalignment_start_time = None # Reset timer if not in violation

    # Head Violation Check
    if head_direction in VALID_HEAD_STATES and \
       head_direction != "Looking at Screen" and \
       head_direction != "Calibrating" and \
       head_direction != "Face Not Detected":
        if head_misalignment_start_time is None:
            head_misalignment_start_time = current_time
        elif current_time - head_misalignment_start_time >= VIOLATION_THRESHOLD_SECONDS:
            filename = os.path.join(LOG_DIR, f"head_{head_direction.replace(' ', '_')}_{int(current_time)}.png")
            cv2.imwrite(filename, frame)
            description = f"Head violation detected: {head_direction}"
            send_email_alert_async(filename, "Head Violation", description)
            print(f"[ALERT] {description}")
            head_misalignment_start_time = None
    else:
        head_misalignment_start_time = None

    # Mobile Detection Check
    if mobile_detected:
        if mobile_detection_start_time is None:
            mobile_detection_start_time = current_time
        elif current_time - mobile_detection_start_time >= VIOLATION_THRESHOLD_SECONDS:
            filename = os.path.join(LOG_DIR, f"mobile_detected_{int(current_time)}.png")
            cv2.imwrite(filename, frame)
            description = "Mobile device detected"
            send_email_alert_async(filename, "Mobile Detection", description)
            print(f"[ALERT] {description}")
            mobile_detection_start_time = None
    else:
        mobile_detection_start_time = None

    # Display the processed frame
    cv2.imshow("Enhanced Surveillance System - Live Feed", frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break # Quit
    elif key == ord('r'):
        # Reset calibration if 'r' is pressed
        calibration_done = False
        calibrated_angles = None
        calibration_samples = []
        start_time = time.time() # Restart calibration timer
        # Reset the head pose state machine when recalibrating to clear old smoothing history
        reset_head_pose_state()
        print(f"[INFO] Recalibrating head pose. Look straight at the camera for {CALIBRATION_TIME} seconds.")

print("[INFO] Shutting down surveillance system...")
cap.release()
cv2.destroyAllWindows()
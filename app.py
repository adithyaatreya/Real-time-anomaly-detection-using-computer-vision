import streamlit as st
import cv2
import time
import os
import smtplib
import threading
import numpy as np
from email.message import EmailMessage
import sqlite3
from datetime import datetime
import subprocess # NEW: For running external processes

# --- Configuration ---
EMAIL_ADDRESS = "tron199805@gmail.com"  # <<< Your Gmail address
EMAIL_PASSWORD = "ktaf advw kdni ilhj"     # <<< Your Gmail App Password
ALERT_RECIPIENT = "romani6042@example.com" # <<< Email address to receive alerts

DB_NAME = "surveillance_logs.db"
LOG_DIR = "log"
os.makedirs(LOG_DIR, exist_ok=True) # Ensure log directory exists

# Cooldown period for emails (seconds)
EMAIL_COOLDOWN = 10

# Violation duration thresholds (seconds)
VIOLATION_DURATION = 3

# Define valid states from your modules (These MUST match outputs from your CV modules)
VALID_HEAD_STATES = [
    "Looking Left", "Looking Right", "Looking Up", "Looking Down", "Looking Tilted",
    "Looking Up and Left", "Looking Up and Right", "Looking Down and Left", "Looking Down and Right"
]
VALID_EYE_STATES = [
    "Looking Left", "Looking Right", "Looking Up", "Looking Down", "Looking Center"
]


# --- Session State Initialization (CRITICAL for Streamlit) ---
print("\n--- Initializing Streamlit Session State ---")
if "surveillance_process" not in st.session_state:
    st.session_state["surveillance_process"] = None # To store the subprocess object for main.py
    print(" 'surveillance_process' initialized.")
else:
    print(" 'surveillance_process' already exists.")

# The rest of the session state items are technically not needed if main.py is running separately
# because main.py won't update these in app.py's session.
# However, keeping them for now in case you re-integrate logic.
if "calibration_data" not in st.session_state:
    st.session_state["calibration_data"] = {
        "done": False,
        "calibrated_angles": None,
        "samples": [],
        "start_time": None,
        "calibration_time": 10
    }
    print(" 'calibration_data' initialized.")
else:
    print(" 'calibration_data' already exists.")

if "violation_timers" not in st.session_state:
    st.session_state["violation_timers"] = {
        "head_misalignment_start_time": None,
        "eye_misalignment_start_time": None,
        "mobile_detection_start_time": None
    }
    print(" 'violation_timers' initialized.")
else:
    print(" 'violation_timers' already exists.")

if "last_violation_display" not in st.session_state:
    st.session_state["last_violation_display"] = None
    print(" 'last_violation_display' initialized.")
else:
    print(" 'last_violation_display' already exists.")

if "gaze_direction" not in st.session_state:
    st.session_state["gaze_direction"] = "N/A"
    print(" 'gaze_direction' initialized.")
else:
    print(" 'gaze_direction' already exists.")

if "head_direction" not in st.session_state:
    st.session_state["head_direction"] = "N/A"
    print(" 'head_direction' initialized.")
else:
    print(" 'head_direction' already exists.")

if "mobile_detected" not in st.session_state:
    st.session_state["mobile_detected"] = False
    print(" 'mobile_detected' initialized.")
else:
    print(" 'mobile_detected' already exists.")

if "calibration_progress" not in st.session_state:
    st.session_state["calibration_progress"] = None
    print(" 'calibration_progress' initialized.")
else:
    print(" 'calibration_progress' already exists.")

if "calibration_status" not in st.session_state:
    st.session_state["calibration_status"] = {'status': 'inactive'}
    print(" 'calibration_status' initialized.")
else:
    print(" 'calibration_status' already exists.")

if 'last_email_time' not in st.session_state:
    st.session_state['last_email_time'] = 0
    print(" 'last_email_time' initialized.")
else:
    print(" 'last_email_time' already exists.")
print("--- Session State Initialization Complete ---\n")


# --- Database Functions ---
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
        st.error(f"Error initializing database: {e}", icon="❌")
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
        print(f"[DB ERROR] Error logging alert: {e}")
    finally:
        if conn:
            conn.close()

# Initialize the database when the Streamlit app starts
init_db()

# --- Asynchronous Email Sending (for test emails from app.py) ---
def send_email_alert_async(filename, alert_type, description):
    """Sends an email alert asynchronously to avoid blocking the Streamlit UI."""
    now = time.time()
    if now - st.session_state['last_email_time'] < EMAIL_COOLDOWN:
        print(f"[EMAIL SKIPPED] Cooldown active for {alert_type}. Next email allowed in {EMAIL_COOLDOWN - (now - st.session_state['last_email_time']):.1f}s.")
        return

    st.session_state['last_email_time'] = now

    def send_email_task():
        try:
            subject = f"[ALERT] {alert_type}"
            body = f"{description}. See attached image: {os.path.basename(filename)}"
            msg = EmailMessage()
            msg["From"] = EMAIL_ADDRESS
            msg["To"] = ALERT_RECIPIENT
            msg["Subject"] = subject
            msg.set_content(body)

            if os.path.exists(filename):
                with open(filename, "rb") as img_file:
                    msg.add_attachment(img_file.read(), maintype="image", subtype="png", filename=os.path.basename(filename))
            else:
                print(f"[EMAIL WARNING] Screenshot file not found for email: {filename}")
                msg.add_attachment(f"Screenshot file not found: {os.path.basename(filename)}. Alert: {description}", maintype="text", subtype="plain")

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                smtp.send_message(msg)

            print(f"[EMAIL SENT] {subject}")
            log_alert_to_db(alert_type, description, filename, emailed=True)

        except Exception as e:
            print(f"[EMAIL ERROR] Failed to send email for {alert_type}: {e}")
            log_alert_to_db(alert_type, description, filename, emailed=False)

    email_thread = threading.Thread(target=send_email_task)
    email_thread.daemon = True
    email_thread.start()


# --- Streamlit UI Layout ---

st.set_page_config(layout="wide", page_title="ProctorSense Dashboard", initial_sidebar_state="collapsed", page_icon="👁️")

st.title("👁️ ProctorSense AI Surveillance Dashboard")

# Custom CSS for styling the Streamlit app
st.markdown(
    """
    <style>
    body {
        font-family: 'Inter', sans-serif;
        background-color: #1a202c; /* Dark background */
        color: #e2e8f0; /* Light text */
    }
    .stApp {
        background-color: #1a202c;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.25rem; /* text-xl */
        font-weight: 600; /* font-semibold */
    }
    .status-box {
        background-color: #2D3748; /* Darker gray for boxes */
        padding: 1rem;
        border-radius: 0.75rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #4A5568;
    }
    .status-label {
        font-size: 1rem;
        color: #A0AEC0;
    }
    .status-value {
        font-size: 1.25rem;
        font-weight: bold;
    }
    .status-green { color: #48BB78; } /* Green */
    .status-red { color: #E53E3E; }   /* Red */
    .status-yellow { color: #ECC94B; } /* Yellow */
    .status-blue { color: #4299E1; } /* Blue */
    .stButton>button {
        background-color: #4A5568;
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.6rem 1.2rem;
        font-size: 1rem;
        transition: background-color 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .stButton>button:hover {
        background-color: #6A7A90;
    }
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background-color: #2D3748;
        color: #e2e8f0;
        border: 1px solid #4A5568;
        border-radius: 0.5rem;
    }
    .stExpander {
        background-color: #2D3748;
        border-radius: 0.75rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #4A5568;
    }
    .stExpander div[role="button"] p {
        font-size: 1.1rem;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Navigation with tabs for different sections of the dashboard
tab1, tab2, tab3 = st.tabs(["Live Monitoring", "Alert History", "Settings"])

with tab1:
    st.header("Live Monitoring Stream (via external script)")
    st.warning("⚠️ **WARNING:** This tab will launch an external `main.py` script. The video feed will appear in a **separate OpenCV window**, not here in the browser. Real-time status updates in this Streamlit UI will NOT reflect the live data from `main.py`.", icon="⚠️")
    st.warning("To stop the `main.py` script, use the 'Stop Surveillance' button, or close its OpenCV window and press 'q' in its terminal. The 'Stop' button here performs a rude termination.", icon="⚠️")


    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("External Surveillance Control")

        if st.button("Start Surveillance (main.py)", help="Launches main.py in a separate process."):
            if st.session_state["surveillance_process"] is None or st.session_state["surveillance_process"].poll() is not None:
                try:
                    # Launch main.py as a separate process.
                    # shell=True might be needed on Windows if python is not in PATH
                    # stdout=subprocess.PIPE, stderr=subprocess.PIPE allows capturing output
                    # bufsize=1, universal_newlines=True is for real-time text output
                    st.session_state["surveillance_process"] = subprocess.Popen(
                        ["python", "main.py"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, # Capture stderr into stdout for easier debugging
                        text=True, # Decode stdout/stderr as text
                        bufsize=1, # Line-buffered
                        # shell=True # Uncomment this if 'python' command isn't found directly on Windows
                    )
                    st.success("`main.py` launched successfully! Check your desktop for the OpenCV window.", icon="✅")
                    print("[Streamlit] main.py process started.")
                except FileNotFoundError:
                    st.error("Error: Python executable or main.py not found. Ensure 'python' is in your PATH and 'main.py' is in the current directory.", icon="❌")
                except Exception as e:
                    st.error(f"Error launching `main.py`: {e}", icon="❌")
            else:
                st.info("`main.py` is already running.", icon="ℹ️")

        if st.button("Stop Surveillance (main.py)", help="Attempts to terminate the main.py process."):
            if st.session_state["surveillance_process"] is not None and st.session_state["surveillance_process"].poll() is None:
                st.session_state["surveillance_process"].terminate() # Send SIGTERM
                time.sleep(1) # Give it a moment to terminate
                if st.session_state["surveillance_process"].poll() is None:
                    st.session_state["surveillance_process"].kill() # Force kill if still running (SIGKILL)
                    st.error("`main.py` forcefully terminated.", icon="🛑")
                else:
                    st.info("`main.py` process stopped.", icon="✅")
                st.session_state["surveillance_process"] = None
                print("[Streamlit] main.py process terminated.")
            else:
                st.info("`main.py` is not running.", icon="ℹ️")

        st.markdown("---")
        st.subheader("`main.py` Console Output (for debugging)")
        output_placeholder = st.empty() # Placeholder to update output dynamically
        if st.session_state["surveillance_process"] is not None:
            # Read stdout of the subprocess in a non-blocking way
            # This is a very basic way to show output. For continuous, large logs,
            # you might need a more robust logging solution.
            output_lines = []
            while True:
                line = st.session_state["surveillance_process"].stdout.readline()
                if line:
                    output_lines.append(line.strip())
                    output_placeholder.text_area("Live Log", value="\n".join(output_lines[-20:]), height=300, disabled=True)
                else:
                    break
            # Check if process has exited
            if st.session_state["surveillance_process"].poll() is not None:
                st.error("`main.py` process has exited. Check logs for details.", icon="⚠️")
                st.session_state["surveillance_process"] = None # Clear the process state


    with col2:
        st.subheader("Real-time Status (from Streamlit UI)")
        st.info("These metrics will NOT update live because `main.py` is running in a separate process and not communicating back to this Streamlit app.", icon="ℹ️")

        # Display static or default values for demonstration.
        # These will not reflect the actual state of main.py
        st.markdown('<div class="status-box">', unsafe_allow_html=True)
        st.markdown('<p class="status-label">Gaze Direction (Static):</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="status-value status-blue">{st.session_state["gaze_direction"]}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="status-box">', unsafe_allow_html=True)
        st.markdown('<p class="status-label">Head Pose (Static):</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="status-value status-blue">{st.session_state["head_direction"]}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="status-box">', unsafe_allow_html=True)
        st.markdown('<p class="status-label">Mobile Detection (Static):</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="status-value status-blue">{"DETECTED" if st.session_state["mobile_detected"] else "CLEAR"}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("Calibration Status (Static)")
        st.info("This status also won't update live. Calibration for `main.py` happens independently within its own process.", icon="ℹ️")
        st.markdown('<div class="status-box">', unsafe_allow_html=True)
        st.markdown('<p class="status-label">Calibration Status:</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="status-value status-blue">Handled by `main.py`</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


with tab2:
    st.header("Alert History")
    st.write("Review past alerts logged by the system.")

    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, alert_type, description, screenshot_path, emailed FROM alerts ORDER BY timestamp DESC")
        alerts = cursor.fetchall()

        if alerts:
            import pandas as pd
            df = pd.DataFrame(alerts, columns=['Timestamp', 'Alert Type', 'Description', 'Screenshot Path', 'Emailed'])
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Alert History as CSV",
                data=csv,
                file_name="proctorsense_alerts.csv",
                mime="text/csv",
            )

            st.markdown("---")
            st.subheader("Detailed Alert View (with Screenshots)")
            for i, alert in enumerate(alerts):
                with st.expander(f"Alert {i+1}: **{alert[1]}** at {alert[0]}"):
                    st.write(f"**Type:** {alert[1]}")
                    st.write(f"**Description:** {alert[2]}")
                    st.write(f"**Emailed:** {'Yes' if alert[4] else 'No'}")
                    if alert[3] and os.path.exists(alert[3]):
                        try:
                            st.image(alert[3], caption="Violation Screenshot", use_column_width=True)
                        except Exception as e:
                            st.warning(f"Could not load screenshot: {e}. File might be corrupted or inaccessible.", icon="⚠️")
                    else:
                        st.info("No screenshot available or file not found for this alert.", icon="ℹ️")
        else:
            st.info("No alerts logged yet.", icon="ℹ️")
    except sqlite3.Error as e:
        st.error(f"Error fetching alerts from database: {e}", icon="❌")
    finally:
        if conn:
            conn.close()

with tab3:
    st.header("Settings")
    st.write("Configure system parameters and email alerts.")

    st.subheader("Email Settings")
    st.info("Make sure to use a Gmail App Password if you have 2FA enabled for your Gmail account, not your regular password. See Google's documentation for 'App passwords'.")
    st.text_input("Sender Email Address (Gmail)", value=EMAIL_ADDRESS, key="email_sender", disabled=True)
    st.text_input("Sender Email App Password", value=EMAIL_PASSWORD, type="password", key="email_password", disabled=True)
    st.text_input("Recipient Email Address", value=ALERT_RECIPIENT, key="email_recipient", disabled=True)
    st.caption("For security, these values are hardcoded in `app.py` for deployment. Edit `EMAIL_ADDRESS`, `EMAIL_PASSWORD`, `ALERT_RECIPIENT` directly in `app.py`.")

    if st.button("Send Test Email (from Streamlit App)"):
        if EMAIL_ADDRESS == "your_email@gmail.com" or EMAIL_PASSWORD == "your_app_password" or ALERT_RECIPIENT == "recipient_email@example.com":
            st.warning("Please update email settings in `app.py` first before sending a test email.")
        else:
            dummy_image_path = os.path.join(LOG_DIR, f"test_email_screenshot_{int(time.time())}.png")
            dummy_img = np.zeros((200, 300, 3), dtype=np.uint8)
            cv2.putText(dummy_img, "TEST IMAGE", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(dummy_image_path, dummy_img)
            send_email_alert_async(dummy_image_path, "Test Alert", "This is a test email from ProctorSense.")
            st.success("Test email initiated! Check your recipient inbox shortly.")

    st.subheader("Violation Thresholds")
    st.info("These settings apply to the Streamlit app's internal logic, but since `main.py` is running externally, its own thresholds will be used.", icon="ℹ️")
    new_violation_duration = st.number_input(
        "Violation Persistence Duration (seconds)",
        min_value=1, max_value=30, value=VIOLATION_DURATION, step=1,
        key="violation_duration_setting", disabled=True
    )
    new_email_cooldown = st.number_input(
        "Email Cooldown Period (seconds)",
        min_value=1, max_value=300, value=EMAIL_COOLDOWN, step=1,
        key="email_cooldown_setting", disabled=True
    )

    st.subheader("Danger Zone")
    if st.button("Clear All Alert History (Database)", help="This action is irreversible and will delete all stored alerts."):
        if st.session_state.get('confirm_clear_db', False):
            conn = None
            try:
                conn = sqlite3.connect(DB_NAME)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM alerts")
                conn.commit()
                st.success("All alert history cleared successfully!", icon="✅")
                st.session_state['confirm_clear_db'] = False
                st.experimental_rerun()
            except sqlite3.Error as e:
                st.error(f"Error clearing database: {e}", icon="❌")
            finally:
                if conn:
                    conn.close()
        else:
            st.session_state['confirm_clear_db'] = True
            st.warning("Are you sure you want to clear ALL alert history? This cannot be undone. Click 'Clear All Alert History' again to confirm.", icon="⚠️")
    if st.session_state.get('confirm_clear_db', False):
        if st.button("Cancel Clear"):
            st.session_state['confirm_clear_db'] = False
            st.info("Clear history cancelled.")
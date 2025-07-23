import cv2
import mediapipe as mp # Import MediaPipe
import numpy as np
import math
from collections import deque

# --- MediaPipe Initialization ---
# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,             # Detects up to 1 face
    refine_landmarks=True,       # Refines landmark detection around eyes and lips for better accuracy
    min_detection_confidence=0.5, # Minimum confidence for face detection to succeed
    min_tracking_confidence=0.5  # Minimum confidence for landmarks to be tracked
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- 3D Model Points (generic facial model for solvePnP) ---
# These points define a generic 3D face model, corresponding to specific facial features.
# The Z-coordinate is negative because in solvePnP, points behind the camera are typically negative.
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float64)

# --- MediaPipe Landmark Indices Mapping ---
# These are the *specific* MediaPipe Face Mesh landmark indices that
# best correspond to the `model_points` above.
# You can find the full list of 468 landmarks and their IDs in MediaPipe documentation.
# This mapping is crucial for accurate pose estimation.
mp_landmark_indices = [
    1,    # Nose tip (MediaPipe index for the center of the nose)
    152,  # Chin (MediaPipe index for the bottom of the chin)
    33,   # Left eye left corner (MediaPipe index for the inner corner of the left eye)
    263,  # Right eye right corner (MediaPipe index for the inner corner of the right eye)
    61,   # Left mouth corner (MediaPipe index for the left corner of the mouth)
    291   # Right mouth corner (MediaPipe index for the right corner of the mouth)
]

# --- Camera Calibration Parameters ---
def get_camera_matrix(frame_width, frame_height):
    """Dynamically calculates the camera matrix based on frame dimensions."""
    focal_length = frame_width # Common approximation for webcams
    center = (frame_width / 2, frame_height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    return camera_matrix

dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion (can be calibrated for better accuracy)

# --- Pose Thresholds (tuned for sensitivity) ---
PITCH_THRESHOLD = 12   # Degrees up/down
YAW_THRESHOLD = 20     # Degrees left/right
ROLL_THRESHOLD = 8     # Degrees tilt

# --- Smoothing Filters for Stability ---
yaw_history = deque(maxlen=8)
pitch_history = deque(maxlen=8)
roll_history = deque(maxlen=8)

# --- State Persistence for Flickering Reduction ---
# Global variables to maintain state across function calls
previous_head_state = "Looking at Screen" # Renamed to avoid conflict if other modules have 'previous_state'
head_state_counter = 0                   # Renamed
STATE_PERSISTENCE_THRESHOLD = 3 # How many frames a state must persist before updating

def get_head_pose_angles(image_points, camera_matrix):
    """
    Calculates head pose angles (pitch, yaw, roll) using OpenCV's solvePnP.
    Args:
        image_points (np.array): 2D coordinates of facial landmarks in the image.
        camera_matrix (np.array): Camera intrinsic parameters.
    Returns:
        tuple: (pitch, yaw, roll) in degrees, rotation_vector, translation_vector.
               Returns (None, None, None) if calculation fails.
    """
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE # Iterative method for best accuracy
    )

    if not success:
        return None, None, None

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Extract Euler angles (Pitch, Yaw, Roll)
    # This uses a standard conversion from rotation matrix to Euler angles.
    sy = math.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
    singular = sy < 1e-6 # Check for gimbal lock

    if not singular:
        pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        pitch = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = 0 # Assume roll is zero if gimbal lock occurs

    return (np.degrees(pitch), np.degrees(yaw), np.degrees(roll)), rotation_vector, translation_vector

def smooth_angle(angle_history_deque, new_angle):
    """
    Applies a simple moving average smoothing with outlier rejection to angle data.
    Uses a deque (double-ended queue) for efficient history management.
    """
    angle_history_deque.append(new_angle)

    if len(angle_history_deque) < 3:
        return new_angle # Not enough data for meaningful smoothing

    angles_array = np.array(list(angle_history_deque))
    median_angle = np.median(angles_array)

    # Filter out values that are significantly different from the median
    # (e.g., more than 30 degrees away, indicating a potential outlier/jump)
    filtered_angles = [a for a in angles_array if abs(a - median_angle) < 30]

    if len(filtered_angles) > 0:
        return np.mean(filtered_angles) # Return the mean of filtered valid angles
    else:
        return new_angle # Fallback to new angle if all previous are outliers

def classify_head_direction(pitch, yaw, roll, calibrated_angles):
    """
    Classifies head direction based on deviations from calibrated angles.
    Applies multi-level classification (e.g., "Up and Left").
    """
    if calibrated_angles is None:
        return "Calibrating" # Should not happen if calibration is done properly

    cal_pitch, cal_yaw, cal_roll = calibrated_angles

    # Calculate deviations from the calibrated "straight" position
    pitch_dev = pitch - cal_pitch
    yaw_dev = yaw - cal_yaw
    roll_dev = roll - cal_roll

    directions = [] # List to store detected directions

    # Yaw (left-right movement)
    if abs(yaw_dev) > YAW_THRESHOLD:
        if yaw_dev < -YAW_THRESHOLD: # Negative yaw deviation usually means looking left
            directions.append("Left")
        elif yaw_dev > YAW_THRESHOLD: # Positive yaw deviation usually means looking right
            directions.append("Right")

    # Pitch (up-down movement)
    if abs(pitch_dev) > PITCH_THRESHOLD:
        if pitch_dev > PITCH_THRESHOLD: # Positive pitch deviation usually means looking up
            directions.append("Up")
        elif pitch_dev < -PITCH_THRESHOLD: # Negative pitch deviation usually means looking down
            directions.append("Down")

    # Roll (head tilt)
    if abs(roll_dev) > ROLL_THRESHOLD:
        directions.append("Tilted") # Just "Tilted" for simplicity, could be "Left Tilted" / "Right Tilted"

    # Combine detected directions into a single string
    if not directions:
        return "Looking at Screen" # No significant deviation
    elif len(directions) == 1:
        return f"Looking {directions[0]}"
    else:
        return f"Looking {' and '.join(directions)}" # e.g., "Looking Up and Left"

def process_head_pose(frame, calibrated_angles=None):
    """
    Main function to process head pose in a video frame.
    Args:
        frame (np.array): The current video frame (BGR).
        calibrated_angles (tuple, optional): (pitch, yaw, roll) of the calibrated straight pose.
                                             If None, it's calibration phase.
    Returns:
        tuple: (processed_frame, head_direction_string) during monitoring,
               or (processed_frame, (pitch, yaw, roll)) during calibration.
    """
    global previous_head_state, head_state_counter # Access global state variables

    h, w, c = frame.shape
    camera_matrix = get_camera_matrix(w, h)
    head_direction = "Face Not Detected" # Default state

    # Convert BGR to RGB for MediaPipe processing
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        # We process only the first detected face for simplicity (max_num_faces=1)
        face_landmarks = results.multi_face_landmarks[0]

        # Extract 2D image points from MediaPipe landmarks
        # Scale normalized landmarks [0,1] to pixel coordinates [0,W] or [0,H]
        image_points = np.array([
            (face_landmarks.landmark[idx].x * w, face_landmarks.landmark[idx].y * h)
            for idx in mp_landmark_indices
        ], dtype=np.float64)

        # Get head pose angles
        angles_data = get_head_pose_angles(image_points, camera_matrix)
        if angles_data[0] is None:
            return frame, "Calculation Failed", None # Return None for angles if calculation fails

        angles, rotation_vector, translation_vector = angles_data

        # Apply smoothing to angles
        pitch = smooth_angle(pitch_history, angles[0])
        yaw = smooth_angle(yaw_history, angles[1])
        roll = smooth_angle(roll_history, angles[2])

        # --- Drawing and Visualization ---
        # Draw all MediaPipe facial landmarks (optional, but good for debugging)
        # Tesselation
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
        # Contours (eyes, mouth, face outline)
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
        )
        # Iris
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
        )

        # Draw a bounding box around the detected face (approximated from landmarks)
        # Find min/max x and y from landmarks to define a bounding box
        lm_x = [lm.x * w for lm in face_landmarks.landmark]
        lm_y = [lm.y * h for lm in face_landmarks.landmark]
        x_min, x_max = int(min(lm_x)), int(max(lm_x))
        y_min, y_max = int(min(lm_y)), int(max(lm_y))
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)


        # If in calibration phase, return the raw angles for calibration
        if calibrated_angles is None:
            # Display raw angles during calibration
            cv2.putText(frame, f"Raw: P:{pitch:.1f} Y:{yaw:.1f} R:{roll:.1f}",
                        (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            # When calibrating, we return the frame and the raw angles tuple
            return frame, "Calibrating", (pitch, yaw, roll) # Return (frame, status_string, angles_tuple)

        # If monitoring, classify head direction and apply state persistence
        current_head_direction = classify_head_direction(pitch, yaw, roll, calibrated_angles)

        # Apply state persistence logic
        if current_head_direction == previous_head_state:
            head_state_counter += 1
        else:
            head_state_counter = 0 # Reset counter if state changes

        # Update final_direction based on state persistence or immediate change
        if head_state_counter >= STATE_PERSISTENCE_THRESHOLD or previous_head_state == "Looking at Screen":
            final_head_direction = current_head_direction
            previous_head_state = current_head_direction
        else:
            final_head_direction = previous_head_state # Stick to previous state if not persistent enough

        # Draw debugging info (angles, offsets, state counter)
        cal_pitch, cal_yaw, cal_roll = calibrated_angles
        cv2.putText(frame, f"Angles: P:{pitch:.1f} Y:{yaw:.1f} R:{roll:.1f}",
                    (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, f"Offset: P:{pitch-cal_pitch:.1f} Y:{yaw-cal_yaw:.1f} R:{roll-cal_roll:.1f}",
                    (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, f"State Counter: {head_state_counter}",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Add head direction label on the bounding box
        label_text = f"HEAD: {final_head_direction}"
        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

        # Position label above the bounding box
        label_x = x_min
        label_y = y_min - 10

        # Draw background rectangle for label
        cv2.rectangle(frame, (label_x, label_y - label_size[1] - 5),
                      (label_x + label_size[0] + 10, label_y + 5), (0, 255, 0), -1)
        cv2.putText(frame, label_text, (label_x + 5, label_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        return frame, final_head_direction, angles # Return angles even in monitoring for external display/logging

    # If no face is detected
    return frame, "Face Not Detected", None # Return None for angles when no face

def reset_head_pose_state():
    """Resets the internal state of the head pose module (smoothing history, state counters)."""
    global previous_head_state, head_state_counter, yaw_history, pitch_history, roll_history
    previous_head_state = "Looking at Screen"
    head_state_counter = 0
    yaw_history.clear()
    pitch_history.clear()
    roll_history.clear()
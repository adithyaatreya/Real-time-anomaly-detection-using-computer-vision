import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# MediaPipe Face Mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1, 
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# History for smoothing
gaze_history = deque(maxlen=7)

def get_eye_landmarks(landmarks):
    """Get eye corner and iris landmarks for both eyes"""
    # Left eye corners and iris
    left_eye_left = landmarks[33]   # Left corner of left eye
    left_eye_right = landmarks[133] # Right corner of left eye
    left_iris = landmarks[468]      # Left iris center
    
    # Right eye corners and iris  
    right_eye_left = landmarks[362]  # Left corner of right eye
    right_eye_right = landmarks[263] # Right corner of right eye
    right_iris = landmarks[473]      # Right iris center
    
    return {
        'left_eye_left': left_eye_left,
        'left_eye_right': left_eye_right,
        'left_iris': left_iris,
        'right_eye_left': right_eye_left,
        'right_eye_right': right_eye_right,
        'right_iris': right_iris
    }

def calculate_gaze_ratio(eye_landmarks, image_width, image_height):
    """Calculate gaze ratio for an eye"""
    # Convert normalized coordinates to pixel coordinates
    left_corner_x = eye_landmarks['left_eye_left'].x * image_width
    right_corner_x = eye_landmarks['left_eye_right'].x * image_width
    iris_x = eye_landmarks['left_iris'].x * image_width
    
    # Calculate eye width and iris position relative to eye corners
    eye_width = right_corner_x - left_corner_x
    iris_position = iris_x - left_corner_x
    
    if eye_width == 0:
        return 0.5
    
    # Normalize iris position (0 = far left, 1 = far right, 0.5 = center)
    gaze_ratio = iris_position / eye_width
    return gaze_ratio

def classify_gaze_improved(landmarks, image_width, image_height):
    """Improved gaze classification using eye corner analysis"""
    eye_data = get_eye_landmarks(landmarks)
    
    # Calculate gaze ratios for both eyes
    left_eye_data = {
        'left_eye_left': eye_data['left_eye_left'],
        'left_eye_right': eye_data['left_eye_right'], 
        'left_iris': eye_data['left_iris']
    }
    
    right_eye_data = {
        'left_eye_left': eye_data['right_eye_left'],
        'left_eye_right': eye_data['right_eye_right'],
        'left_iris': eye_data['right_iris']
    }
    
    left_gaze_ratio = calculate_gaze_ratio(left_eye_data, image_width, image_height)
    right_gaze_ratio = calculate_gaze_ratio(right_eye_data, image_width, image_height)
    
    # Average both eyes for more stable detection
    avg_gaze_ratio = (left_gaze_ratio + right_gaze_ratio) / 2
    
    # Adjusted thresholds for better sensitivity
    if avg_gaze_ratio < 0.42:
        return "Looking Left"
    elif avg_gaze_ratio > 0.58:
        return "Looking Right"
    elif avg_gaze_ratio < 0.46:
        return "Looking Slightly Left"
    elif avg_gaze_ratio > 0.54:
        return "Looking Slightly Right"
    else:
        return "Looking Center"

def get_vertical_gaze(landmarks, image_height):
    """Detect vertical gaze direction"""
    # Using upper and lower eyelid landmarks for vertical gaze
    left_upper = landmarks[159]  # Left eye upper lid
    left_lower = landmarks[145]  # Left eye lower lid
    left_iris = landmarks[468]   # Left iris
    
    right_upper = landmarks[386] # Right eye upper lid  
    right_lower = landmarks[374] # Right eye lower lid
    right_iris = landmarks[473]  # Right iris
    
    # Calculate vertical position ratios
    left_eye_height = abs(left_upper.y - left_lower.y) * image_height
    left_iris_pos = (left_iris.y - left_upper.y) * image_height
    
    right_eye_height = abs(right_upper.y - right_lower.y) * image_height
    right_iris_pos = (right_iris.y - right_upper.y) * image_height
    
    if left_eye_height == 0 or right_eye_height == 0:
        return "Center"
    
    left_vertical_ratio = left_iris_pos / left_eye_height
    right_vertical_ratio = right_iris_pos / right_eye_height
    avg_vertical_ratio = (left_vertical_ratio + right_vertical_ratio) / 2
    
    if avg_vertical_ratio < 0.35:
        return "Looking Up"
    elif avg_vertical_ratio > 0.65:
        return "Looking Down"
    else:
        return "Looking Center"

def process_eye_movement(frame):
    global gaze_history
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    height, width = frame.shape[:2]
    gaze_direction = "Unknown"
    vertical_gaze = "Center"

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        try:
            # Get horizontal gaze direction
            gaze_direction = classify_gaze_improved(landmarks, width, height)
            
            # Get vertical gaze direction
            vertical_gaze = get_vertical_gaze(landmarks, height)
            
            # Combine horizontal and vertical if needed
            if vertical_gaze != "Looking Center" and gaze_direction == "Looking Center":
                gaze_direction = vertical_gaze
            elif vertical_gaze != "Looking Center" and gaze_direction != "Looking Center":
                gaze_direction = f"{vertical_gaze.replace('Looking ', '')} {gaze_direction.replace('Looking ', '')}"
                
        except (IndexError, AttributeError) as e:
            gaze_direction = "Unknown"
            print(f"Error in gaze detection: {e}")
    else:
        gaze_direction = "Face Not Detected"

    # Improved smoothing with weighted voting
    gaze_history.append(gaze_direction)
    
    if len(gaze_history) >= 3:
        # Count occurrences of each direction
        direction_counts = {}
        for direction in gaze_history:
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
        
        # Find most common direction
        most_common = max(direction_counts.items(), key=lambda x: x[1])
        
        # Use most common if it appears at least 2 times in recent history
        if most_common[1] >= 2:
            confident_gaze = most_common[0]
        else:
            confident_gaze = gaze_direction
    else:
        confident_gaze = gaze_direction

    # Visual feedback - FIXED: Moved gaze text down to avoid overlap
    cv2.putText(frame, f"Gaze (Raw): {gaze_direction}", (20, 160), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"Gaze (Smooth): {confident_gaze}", (20, 185), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Draw eye landmarks and iris tracking
    if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
        landmarks = results.multi_face_landmarks[0].landmark
        eye_data = get_eye_landmarks(landmarks)
        
        # Draw LARGE RED dots on iris for clear visibility
        for iris_name in ['left_iris', 'right_iris']:
            x = int(eye_data[iris_name].x * width)
            y = int(eye_data[iris_name].y * height)
            # Large red circle for iris
            cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)  # Red filled circle
            cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)  # White outline
        
        # Draw eye corners as smaller blue dots
        for corner_name in ['left_eye_left', 'left_eye_right', 'right_eye_left', 'right_eye_right']:
            x = int(eye_data[corner_name].x * width)
            y = int(eye_data[corner_name].y * height)
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)  # Blue dots for corners
            
        # Draw eye outline for better visualization
        # Left eye outline
        left_eye_points = []
        for idx in [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]:
            x = int(landmarks[idx].x * width)
            y = int(landmarks[idx].y * height)
            left_eye_points.append([x, y])
        cv2.polylines(frame, [np.array(left_eye_points)], True, (0, 255, 255), 1)
        
        # Right eye outline
        right_eye_points = []
        for idx in [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]:
            x = int(landmarks[idx].x * width)
            y = int(landmarks[idx].y * height)
            right_eye_points.append([x, y])
        cv2.polylines(frame, [np.array(right_eye_points)], True, (0, 255, 255), 1)

    return frame, confident_gaze
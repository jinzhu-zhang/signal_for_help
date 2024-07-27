import cv2
import mediapipe as mp
import time

# initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# initialize OpenCV
cap = cv2.VideoCapture(1)

gesture_detected = False
gesture_start_time = None
gesture_timeout = 2

notification_active = False


def detect_gesture(hand_landmarks):
    global gesture_detected, gesture_start_time
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    thumb_tucked = thumb_tip.y > index_tip.y
    fingers_folded = all(tip.y > thumb_tip.y for tip in [index_tip, middle_tip, ring_tip, pinky_tip])

    if not gesture_detected:
        if thumb_tucked:
            gesture_start_time = time.time()
            gesture_detected = True
    else:
        if fingers_folded and (time.time() - gesture_start_time) < gesture_timeout:
            return True
        elif not fingers_folded:
            gesture_detected = False

    return False

# mouse callback function
def mouse_callback(event, x, y, flags, param):
    global notification_active
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Mouse clicked at: ({x}, {y})")
        if 270 <= x <= 300 and 30 <= y <= 60:  # check if click is within the "X" button area
            print("Close button clicked!")
            notification_active = False


cv2.namedWindow('Hand Tracking')
cv2.setMouseCallback('Hand Tracking', mouse_callback)

# main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # convert the BGR image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    # process the image and detect hands
    results = hands.process(image)

    # convert the image color back so it can be displayed
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # draw the hand annotations on the image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if detect_gesture(hand_landmarks):
                notification_active = True

    if notification_active:
        # draw the notification box
        cv2.rectangle(image, (30, 30), (300, 100), (0, 0, 255), -1)
        cv2.putText(image, "Help is needed!", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # draw the "X" button
        cv2.rectangle(image, (270, 30), (300, 60), (255, 255, 255), -1)
        cv2.putText(image, "X", (275, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # display the result
    cv2.imshow('Hand Tracking', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

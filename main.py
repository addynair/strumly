import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import time
from collections import deque, Counter
import pygame

# ------------------------------
# Load trained RandomForest model
# ------------------------------
MODEL_PATH = "hand_rf_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(" Model not found! Run train_rf.py first.")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ------------------------------
# MediaPipe setup
# ------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# ------------------------------
# Constants
# ------------------------------
STRUM_X_START = 500
STRUM_X_END = 640
STRUM_Y_START = 200
STRUM_Y_END = 380

MODE_MENU = "menu"
MODE_PRACTICE = "practice"
MODE_NORMAL = "normal"
MODE_SONG = "song"

current_mode = MODE_MENU
selected_chord = None
last_strum = 0
strum_cooldown = 0.5
last_preds = deque(maxlen=5)

CHORDS = ["A_major", "D_major", "E_major", "G_major"]


# ------------------------------
# Helper functions
# ------------------------------
def play_chord_audio(chord_name):
    sound_path = f"/home/sneha/please/assets/sounds/{chord_name}.mp3"
    if os.path.exists(sound_path):
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(sound_path)
            pygame.mixer.music.play()
            print(f"🎵 Playing: {chord_name}")
        except Exception as e:
            print(" Audio error:", e)
    else:
        print(f" Audio not found for {chord_name}")

def normalize_landmarks(lm):
    arr = np.array([[p.x, p.y, p.z] for p in lm.landmark])
    center = arr[0]
    arr -= center
    hand_size = np.linalg.norm(arr[0] - arr[9])
    if hand_size > 0:
        arr /= hand_size
    return arr.flatten().reshape(1, -1)

def draw_button(frame, text, x, y, w, h, color, hover_color, mouse_pos, clicked):
    hover = x < mouse_pos[0] < x + w and y < mouse_pos[1] < y + h
    rect_color = hover_color if hover else color
    cv2.rectangle(frame, (x, y), (x + w, y + h), rect_color, -1)
    cv2.putText(frame, text, (x + 20, y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return hover and clicked

# ------------------------------
# Mouse handling
# ------------------------------
mouse_pos = (0, 0)
mouse_click = False
def mouse_event(event, x, y, flags, param):
    global mouse_pos, mouse_click
    mouse_pos = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_click = True
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_click = False

# ------------------------------
# Main Loop
# ------------------------------
cap = cv2.VideoCapture(0)
cv2.namedWindow("AI Air Guitar")
cv2.setMouseCallback("AI Air Guitar", mouse_event)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    click = mouse_click
    mouse_click = False

    # ==============================
    # MENU
    # ==============================
    if current_mode == MODE_MENU:
        cv2.putText(frame, "AI AIR GUITAR", (150, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)

        # Smaller buttons
        btn_w = 250
        btn_h = 55

        practice_clicked = draw_button(frame, "Practice Mode", 200, 200, btn_w, btn_h,
                                    (100, 100, 255), (140, 140, 255), mouse_pos, click)

        normal_clicked = draw_button(frame, "Normal Mode", 200, 280, btn_w, btn_h,
                                    (0, 180, 100), (0, 220, 150), mouse_pos, click)

        song_clicked = draw_button(frame, "Song Mode", 200, 360, btn_w, btn_h,
                                (255, 140, 0), (255, 165, 50), mouse_pos, click)


        if practice_clicked:
            current_mode = MODE_PRACTICE
        elif normal_clicked:
            current_mode = MODE_NORMAL
        elif song_clicked:
            current_mode = MODE_SONG
            song_index = 0

    # ==============================
    # PRACTICE MODE
    # ==============================
    elif current_mode == MODE_PRACTICE:

        if selected_chord is None:
            x, y = 100, 50 
            for i, chord in enumerate(CHORDS): 
                if draw_button(frame, chord, x, y + i * 70, 250, 60,
                                (100, 100, 255), (140, 140, 255), mouse_pos, click): 
                                selected_chord = chord 
                                time.sleep(0.3)

            if draw_button(frame, "Back", 500, 420, 120, 50,
                           (100, 100, 100), (150, 150, 150), mouse_pos, click):
                current_mode = MODE_MENU

        else:
            #Heading
            cv2.putText(frame, f"Practice: {selected_chord}", (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            # Draw chord image
            chord_path = os.path.join("assets/chords", f"{selected_chord}.png")
            img = cv2.imread(chord_path)
            if img is not None:
                h_img, w_img, _ = img.shape
                scale = 150 / max(h_img, w_img)
                img = cv2.resize(img, (int(w_img * scale), int(h_img * scale)))
                y_offset, x_offset = 80, frame.shape[1] - img.shape[1] - 30
                frame[y_offset:y_offset + img.shape[0],
                      x_offset:x_offset + img.shape[1]] = img

            # Strum area
            cv2.rectangle(frame, (STRUM_X_START, STRUM_Y_START),
                          (STRUM_X_END, STRUM_Y_END), (150, 150, 150), 2)
            cv2.putText(frame, "STRUM", (STRUM_X_START + 20, STRUM_Y_END - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            predicted_chord = "None"
            chord_hand, strum_hand = None, None

            # Detect both hands
            if results.multi_hand_landmarks:
                hands_list = []
                for lm in results.multi_hand_landmarks:
                    x_coords = [p.x for p in lm.landmark]
                    avg_x = np.mean(x_coords)
                    hands_list.append((avg_x, lm))
                    mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                hands_list.sort(key=lambda x: x[0])
                if len(hands_list) == 1:
                    chord_hand = hands_list[0][1]
                elif len(hands_list) >= 2:
                    chord_hand = hands_list[0][1]
                    strum_hand = hands_list[1][1]

            # Predict chord
            if chord_hand:
                landmarks = normalize_landmarks(chord_hand)
                pred = model.predict(landmarks)[0]
                last_preds.append(pred)
                predicted_chord = Counter(last_preds).most_common(1)[0][0]

            # Strum detect
            if strum_hand:
                hand_x = int(strum_hand.landmark[8].x * w)
                hand_y = int(strum_hand.landmark[8].y * h)
                in_strum = STRUM_X_START <= hand_x <= STRUM_X_END and STRUM_Y_START <= hand_y <= STRUM_Y_END
                cur_time = time.time()
                if in_strum and (cur_time - last_strum) > strum_cooldown:
                    print("Strum detected!")
                    play_chord_audio(predicted_chord)
                    last_strum = cur_time

            # Feedback
            cv2.putText(frame, f"Predicted: {predicted_chord}", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if predicted_chord == selected_chord:
                cv2.putText(frame, "Correct!", (20, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
            else:
                cv2.putText(frame, "Try again", (20, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

            # Navigation buttons
            if draw_button(frame, "Back", 20, 420, 120, 50,
                           (100, 100, 100), (150, 150, 150), mouse_pos, click):
                selected_chord = None
            if draw_button(frame, "Menu", 500, 420, 120, 50,
                           (100, 100, 100), (150, 150, 150), mouse_pos, click):
                selected_chord = None
                current_mode = MODE_MENU

    # ==============================
    # NORMAL MODE (free play)
    # ==============================
    elif current_mode == MODE_NORMAL:
        cv2.rectangle(frame, (STRUM_X_START, STRUM_Y_START),
                      (STRUM_X_END, STRUM_Y_END), (150, 150, 150), 2)
        cv2.putText(frame, "STRUM (Free Play)", (STRUM_X_START + 20, STRUM_Y_END - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        predicted_chord = "None"
        chord_hand, strum_hand = None, None

        if results.multi_hand_landmarks:
            hands_list = []
            for lm in results.multi_hand_landmarks:
                x_coords = [p.x for p in lm.landmark]
                avg_x = np.mean(x_coords)
                hands_list.append((avg_x, lm))
                mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            hands_list.sort(key=lambda x: x[0])
            if len(hands_list) == 1:
                chord_hand = hands_list[0][1]
            elif len(hands_list) >= 2:
                chord_hand = hands_list[0][1]
                strum_hand = hands_list[1][1]

        if chord_hand:
            landmarks = normalize_landmarks(chord_hand)
            pred = model.predict(landmarks)[0]
            last_preds.append(pred)
            predicted_chord = Counter(last_preds).most_common(1)[0][0]

        if strum_hand:
            hand_x = int(strum_hand.landmark[8].x * w)
            hand_y = int(strum_hand.landmark[8].y * h)
            in_strum = STRUM_X_START <= hand_x <= STRUM_X_END and STRUM_Y_START <= hand_y <= STRUM_Y_END
            cur_time = time.time()
            if in_strum and (cur_time - last_strum) > strum_cooldown:
                print("Strum detected!")
                play_chord_audio(predicted_chord)
                last_strum = cur_time

        cv2.putText(frame, f"Predicted: {predicted_chord}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        if draw_button(frame, "Back", 20, 420, 120, 50,
                       (100, 100, 100), (150, 150, 150), mouse_pos, click):
            current_mode = MODE_MENU

    
   
# ==============================
# SONG MODE
# ==============================
    elif current_mode == MODE_SONG:
        cv2.putText(frame, "Song Mode: Bad Moon Rising", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

       # Strum area (same as normal mode) 
        cv2.rectangle(frame, (STRUM_X_START, STRUM_Y_START),
                      (STRUM_X_END, STRUM_Y_END), (150, 150, 150), 2) 
        cv2.putText(frame, "STRUM", (STRUM_X_START + 20, STRUM_Y_END - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)


        # Wonderwall chord sequence
        SONG_CHORDS = ["D_major", "A_major", "G_major", "D_major"]
        
        # Display all chords horizontally
        x_start = 50
        y_start = 100
        spacing = 150
        for i, chord in enumerate(SONG_CHORDS):
            color = (0, 255, 255) if i == song_index else (200, 200, 200)
            cv2.putText(frame, chord, (x_start + i * spacing, y_start),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # Current chord to play
        expected_chord = SONG_CHORDS[song_index]

        # Detect hands
        predicted_chord = "None"
        chord_hand, strum_hand = None, None

        if results.multi_hand_landmarks:
            hands_list = []
            for lm in results.multi_hand_landmarks:
                x_coords = [p.x for p in lm.landmark]
                avg_x = np.mean(x_coords)
                hands_list.append((avg_x, lm))
                mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            hands_list.sort(key=lambda x: x[0])
            if len(hands_list) == 1:
                chord_hand = hands_list[0][1]
            elif len(hands_list) >= 2:
                chord_hand = hands_list[0][1]
                strum_hand = hands_list[1][1]

        # Predict chord
        if chord_hand:
            landmarks = normalize_landmarks(chord_hand)
            pred = model.predict(landmarks)[0]
            last_preds.append(pred)
            predicted_chord = Counter(last_preds).most_common(1)[0][0]

        # Strum detection
        if strum_hand:
            hand_x = int(strum_hand.landmark[8].x * w)
            hand_y = int(strum_hand.landmark[8].y * h)
            in_strum = STRUM_X_START <= hand_x <= STRUM_X_END and STRUM_Y_START <= hand_y <= STRUM_Y_END
            cur_time = time.time()
            if in_strum and (cur_time - last_strum) > strum_cooldown:
                print("🎸 Strum detected!")
                play_chord_audio(predicted_chord)
                last_strum = cur_time

                # Move to next chord if correct
                if predicted_chord == expected_chord:
                    song_index = (song_index + 1) % len(SONG_CHORDS)

        # Feedback
        cv2.putText(frame, f"Predicted: {predicted_chord}", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Navigation
        if draw_button(frame, "Back", 20, 420, 100, 50,
                    (100, 100, 100), (150, 150, 150), mouse_pos, click):
            current_mode = MODE_MENU
            song_index = 0


    # ==============================
    # Display
    # ==============================
    cv2.imshow("AI Air Guitar", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

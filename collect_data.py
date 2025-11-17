# collect_data.py
import cv2
import mediapipe as mp
import pandas as pd
import os
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

OUTPUT_CSV = "hand_chords_dataset.csv"
# Only major chords (you said you want only majors)
CHORDS = ["Not Any chord","A_major","B major","C major", "D_major", "G_major", "E_major","F major","A minor","E minor","D minor"]  # add more major chords here

def landmarks_to_row(landmarks):
    # flatten x,y,z for 21 points => 63 values
    row = []
    for p in landmarks.landmark:
        row.extend([p.x, p.y, p.z])
    return row

def main():
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    data = []
    cv2.namedWindow("Record Chords", cv2.WINDOW_NORMAL)
    for chord in CHORDS:
        print(f"Get ready to record '{chord}'. Press 's' to start (space to snapshot frames too).")
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"Show {chord} and press 's' to start", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0),2)
            cv2.imshow("Record Chords", frame)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break

        print("Recording... press 'q' to stop this chord, 'p' to pause/resume.")
        paused = False
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks and not paused:
                lm = results.multi_hand_landmarks[0]
                row = landmarks_to_row(lm)
                row.append(chord)
                row.append(time.time())  # timestamp (optional)
                data.append(row)
                mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

                cv2.putText(frame, f"{chord} captured: {len([r for r in data if r[-2]==chord])}",
                            (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),2)

            cv2.putText(frame, f"Recording {chord} - q to stop", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),1)

            cv2.imshow("Record Chords", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            if k == ord('p'):
                paused = not paused

    cap.release()
    cv2.destroyAllWindows()

    # build DataFrame with headers
    cols = []
    for i in range(21):
        cols += [f"x{i}", f"y{i}", f"z{i}"]
    cols += ["label", "timestamp"]
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Dataset saved to {OUTPUT_CSV} with {len(df)} rows.")

if __name__ == "__main__":
    main()
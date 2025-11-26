#!/usr/bin/env python3
"""
MediaPipe Holistic visual demo + UDP sender for head (nose) position.

- Visual styling follows the original notebook (face/hands/pose DrawingSpec).
- Sends messages "nx,ny,t" (normalized nose x,y and timestamp) via UDP to (host,port).
- Default: host=127.0.0.1, port=5005, show window (so appearance doesn't change).

Usage:
  pip install mediapipe opencv-python
  python mediapipe_holistic_udp.py --host 127.0.0.1 --port 5005 --fps 60

Adapted from:
https://github.com/nicknochnack/Full-Body-Estimation-using-Media-Pipe-Holistic/blob/ff19bb1a3cd610ff515f1e7bcb9548f86cb7b76c/Media%20Pipe%20Holistic%20Tutorial.ipynb
"""
import argparse
import socket
import time
import sys
import os

# reduce TF/TFLite verbose logs (optional)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

try:
    import cv2
    import mediapipe as mp
except Exception as e:
    print("Missing modules:", e)
    print("Install with: pip install mediapipe opencv-python")
    sys.exit(1)


def get_face_connections(mp_holistic):
    """Return face connections compatible with draw_landmarks if available."""
    try:
        return mp_holistic.FACE_CONNECTIONS
    except Exception:
        pass
    try:
        return mp.solutions.face_mesh.FACEMESH_TESSELATION
    except Exception:
        pass
    try:
        return mp.solutions.face_mesh.FACEMESH_CONTOURS
    except Exception:
        pass
    return None


def main(host, port, camera, mirror, target_fps, show_window):
    # UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (host, port)

    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    FACE_CONNECTIONS = get_face_connections(mp_holistic)

    # Drawing specs taken to match the notebook's styling
    face_conn_spec = mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1)
    face_land_spec = mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)

    right_conn_spec = mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4)
    right_land_spec = mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)

    left_conn_spec = mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4)
    left_land_spec = mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)

    pose_conn_spec = mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4)
    pose_land_spec = mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f"Cannot open camera {camera}")
        return

    frame_interval = 1.0 / max(1, target_fps)
    last_send = 0.0

    # Use the holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        try:
            while True:
                t0 = time.time()
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame")
                    break

                if mirror:
                    frame = cv2.flip(frame, 1)

                # Optional resize for performance:
                # frame = cv2.resize(frame, (640, 480))

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = holistic.process(image_rgb)
                image_rgb.flags.writeable = True

                # Prepare image for display (BGR)
                image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

                # Draw face landmarks (with fallback if necessary)
                if results.face_landmarks:
                    try:
                        if FACE_CONNECTIONS is not None:
                            mp_drawing.draw_landmarks(
                                image,
                                results.face_landmarks,
                                FACE_CONNECTIONS,
                                face_conn_spec,
                                face_land_spec,
                            )
                        else:
                            mp_drawing.draw_landmarks(image, results.face_landmarks)
                    except Exception:
                        mp_drawing.draw_landmarks(image, results.face_landmarks)

                # Right hand
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        results.right_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        right_conn_spec,
                        right_land_spec,
                    )

                # Left hand
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        results.left_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        left_conn_spec,
                        left_land_spec,
                    )

                # Pose (body)
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                        pose_conn_spec,
                        pose_land_spec,
                    )

                # Send nose position via UDP when available (rate-limited)
                now = time.time()
                if results.pose_landmarks and (now - last_send) >= frame_interval:
                    try:
                        # Some Mediapipe versions use mp_holistic.PoseLandmark.NOSE
                        nose_landmark = mp_holistic.PoseLandmark.NOSE
                        nose = results.pose_landmarks.landmark[nose_landmark]
                        nx = float(nose.x)  # normalized 0..1
                        ny = float(nose.y)
                        msg = f"{nx:.6f},{ny:.6f},{now:.6f}"
                        try:
                            sock.sendto(msg.encode('utf-8'), addr)
                        except Exception as e:
                            # non-fatal: print occasional errors
                            if int(now) % 5 == 0:
                                print("Send error:", e)
                        last_send = now
                    except Exception as e:
                        # occasional indexing/compatibility issue: ignore, print occasionally
                        if int(now) % 5 == 0:
                            print("Nose extraction error:", e)

                # Show the window (keep naming like the original notebook)
                if show_window:
                    cv2.imshow("Raw Webcam Feed", image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # maintain approximate target fps / rate
                elapsed = time.time() - t0
                to_sleep = frame_interval - elapsed
                if to_sleep > 0:
                    # small cap to avoid long sleeps
                    time.sleep(min(to_sleep, 0.01))
        finally:
            cap.release()
            if show_window:
                cv2.destroyAllWindows()
            sock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediaPipe Holistic + UDP head sender (visual)")
    parser.add_argument("--host", default="127.0.0.1", help="UDP destination host (Unity)")
    parser.add_argument("--port", type=int, default=5005, help="UDP destination port")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--fps", type=int, default=60, help="Target send rate (Hz)")
    parser.add_argument("--no-mirror", dest="mirror", action="store_false", help="Disable horizontal flip")
    parser.add_argument("--no-window", dest="show_window", action="store_false", help="Disable cv2 window (if you don't want the visual)")
    args = parser.parse_args()
    try:
        main(host=args.host, port=args.port, camera=args.camera, mirror=args.mirror, target_fps=args.fps, show_window=args.show_window)
    except KeyboardInterrupt:
        pass
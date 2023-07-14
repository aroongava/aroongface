import cv2
import mediapipe as mp
import streamlit as st

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def find_available_camera():
    num_cameras = 10  # 시스템에서 확인할 카메라 수 (증가시켜보면서 확인해보세요)
    for i in range(num_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Found available camera at index: {i}")
            cap.release()
            return i
    
    raise ValueError("No available cameras found")

def overlay(image, x, y, w, h, overlay_image):
    alpha = overlay_image[:, :, 3]
    mask_image = alpha / 255
    for c in range(0, 3):
        y_min, y_max = max(y - h, 0), min(y + h, image.shape[0])
        x_min, x_max = max(x - w, 0), min(x + w, image.shape[1])
        overlay_area = image[y_min:y_max, x_min:x_max, c]
        overlay_image_area = overlay_image[max(0, h - y):h + min(h, image.shape[0] - y),
                                           max(0, w - x):w + min(w, image.shape[1] - x), c]
        image[y_min:y_max, x_min:x_max, c] = (overlay_image_area * mask_image[:overlay_image_area.shape[0],
                                                                             :overlay_image_area.shape[1]]) + (
                                                    overlay_area * (1 - mask_image[:overlay_image_area.shape[0],
                                                                                    :overlay_image_area.shape[1]]))

def main():
    st.title("귀여운 아룽이로 변신하기")
    st.header("Webcam을 이용하여 얼굴 인식 후 아룽이로 변신하는 애플리케이션")

    # Load and display the 'aroong.png' image
    aroong_image = cv2.imread('aroong.png')
    aroong_resized = cv2.resize(aroong_image, (0, 0), fx=0.2, fy=0.2)
    aroong_resized = cv2.cvtColor(aroong_resized, cv2.COLOR_BGR2RGB)
    st.image(aroong_resized, channels="RGB", caption="Aroong Image")

    image_right_eye = cv2.imread('aroong_right_eye.png', cv2.IMREAD_UNCHANGED)
    image_left_eye = cv2.imread('aroong_left_eye.png', cv2.IMREAD_UNCHANGED)
    image_nose = cv2.imread('aroong_nose.png', cv2.IMREAD_UNCHANGED)
    image_mouth = cv2.imread('aroong_mouth.png', cv2.IMREAD_UNCHANGED)
    image_right_ear = cv2.imread('aroong_right_ear.png', cv2.IMREAD_UNCHANGED)
    image_left_ear = cv2.imread('aroong_left_ear.png', cv2.IMREAD_UNCHANGED)

    # For webcam input:
    camera_index = find_available_camera()
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        st.error("Failed to open the camera.")
        return

    stframe = st.empty()

    with mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5) as face_detection:
        while True:
            success, image = cap.read()
            if not success:
                st.error("Failed to read a frame from the camera.")
                break

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image, detection)
                    keypoints = detection.location_data.relative_keypoints

                    right_eye, left_eye, nose_tip, mouth_tip, right_ear, left_ear = keypoints[:6]

                    h, w, _ = image.shape  # 가로, 세로 크기
                    right_eye = (int(right_eye.x * w), int(right_eye.y * h))
                    left_eye = (int(left_eye.x * w), int(left_eye.y * h))
                    nose_tip = (int(nose_tip.x * w), int(nose_tip.y * h - 30))
                    right_ear = (int(right_ear.x * w), int(right_ear.y * h))
                    left_ear = (int(left_ear.x * w), int(left_ear.y * h))
                    mouth_tip = (int(mouth_tip.x * w), int(mouth_tip.y * h + 20))

                    overlays = [(right_eye, image_right_eye), (left_eye, image_left_eye), (nose_tip, image_nose),
                                (right_ear, image_right_ear), (left_ear, image_left_ear), (mouth_tip, image_mouth)]

                    for keypoint, keypoint_image in overlays:
                        overlay(image, *keypoint, int(keypoint_image.shape[1] / 2), int(keypoint_image.shape[0] / 2),
                                keypoint_image)

            # Flip the image horizontally for a selfie-view display.
            stframe.image(cv2.flip(image, 1), channels="BGR", caption='MediaPipe Face Detection', use_column_width=True)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

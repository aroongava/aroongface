import cv2
import mediapipe as mp
import streamlit as st
import streamlit_webrtc as webrtc

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

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

    webrtc_ctx = webrtc.StreamerRTC()
    video_transformer = webrtc_ctx.video_transformer()

    with mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5) as face_detection:
        while True:
            if not webrtc_ctx.video_receiver():
                st.warning("Waiting for video...")
                continue

            image = webrtc_ctx.video_receiver().data
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            results = face_detection.process(image)

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
                # Your face detection and overlay code here

            # Flip the image horizontally for a selfie-view display.
            image = cv2.flip(image, 1)
            webrtc_ctx.video_sender().send(image)

    webrtc_ctx.video_receiver().stop()
    webrtc_ctx.video_sender().stop()

if __name__ == '__main__':
    main()

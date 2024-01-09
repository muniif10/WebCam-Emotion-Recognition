import numpy as np
import cv2
import streamlit as st
from tensorflow import keras
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import (
    webrtc_streamer,
    VideoTransformerBase,
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
)
import tensorflow as tf


# Emotions dictionary used by the model.
emotion_dict = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happiness",
    4: "sadness",
    5: "surprise",
    6: "neutral",
}


classifier = tf.keras.models.load_model(
    "/home/wackster/MachineLearning/Working_Directory/model.keras"
)


# load face
try:
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
except Exception:
    st.write("Error loading cascade classifiers")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


class Faceemotion(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5
        )
        for x, y, w, h in faces:
            cv2.rectangle(
                img=img, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2
            )
            roi_gray = img_gray[y : y + h, x : x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                print(prediction)
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout) + " " + "{:,.0%}".format(prediction[maxindex])
            label_position = (x, y)
            cv2.putText(
                img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

        return img


def main():
    # Face Analysis Application #
    st.title("Real Time Face Recognition for Food Industry")

    st.header("Webcam Feed with CV2 haarcascade and CNN model.")
    st.write("Click on start to use webcam and detect your face emotion")
    webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=Faceemotion,
    )


if __name__ == "__main__":
    main()

import cv2
import streamlit as st
import numpy as np

# Importing HARR CASCADE XML file
face_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface_extended.xml')
human_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# CSS styling
st.write("""
    <style>
    .text-justify {
        text-align: justify;
        text-justify: inter-word;
    }
    </style>
""", unsafe_allow_html=True)

# Tulisan yang akan dijustify
text1 = "Aplikasi ini bertujuan untuk mendeteksi wajah menggunakan algoritma pengolahan citra. Dalam aplikasi ini, terdapat dua jenis deteksi yaitu deteksi wajah pada manusia dan wajah kucing."
text2 = "Anda dapat menggunakan aplikasi ini untuk mengamati keberadaan wajah manusia dan kucing dalam sebuah kamera. Selamat mencoba!"
# Menampilkan tulisan yang sudah di justify


def main():

    st.title("Real-Time Human and Cat Face Detector")
    st.write(f"<p class='text-justify'>{text1}</p>", unsafe_allow_html=True)
    st.write(f"<p class='text-justify'>{text2}</p>", unsafe_allow_html=True)
    st.write("Tekan tombol Mulai untuk memulai :) ")
    run_detection = st.button("Mulai")

    # Creating a loop to capture each frame of the video in the name of Img
    if run_detection:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.write("Unable to access camera")
            st.stop()

        while True:
            _, img = cap.read()

            # Converting to grey scale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Allowing multiple face detection
            faces = face_cascade.detectMultiScale(gray, 1.1, 6)
            human = human_cascade.detectMultiScale(gray, 1.1, 6)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 250), 2)
                cv2.putText(img, "Cat", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

            for (x, y, w, h) in human:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 250), 2)
                cv2.putText(img, "Human", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

            # Displaying the image
            cv2.imshow('Detected Face Image',  img)

            # Checking if the window is closed
            if cv2.getWindowProperty('Detected Face Image', cv2.WND_PROP_VISIBLE) < 1:
                break

            # Waiting for escape key for image to close adding the break statement to end the face detection screen
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        # Real-time releasing the captured frames
        cap.release()

        # Closing all windows
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

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
text1 = "Pada bagian ini sama dengan sebelumnya, namun media yang digunakan disini merupakan upload sebuah gambar untuk mendeteksi wajah manusia dan kucing."
text2 = "Anda dapat menggunakan aplikasi ini untuk mengamati keberadaan wajah manusia dan kucing dalam sebuah foto/gambar. Selamat mencoba!"


def main():
    st.title("Human and Cat Face Image Detector")
    st.write(f"<p class='text-justify'>{text1}</p>", unsafe_allow_html=True)
    st.write(f"<p class='text-justify'>{text2}</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Pilih gambar yang akan dideteksi")
    if uploaded_file is not None:
        st.write("You have uploaded a file.")
        file_bytes = np.asarray(
            bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

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

        # Displaying the uploaded image
        st.image(img, channels="BGR", caption="Uploaded Image")


if __name__ == "__main__":
    main()

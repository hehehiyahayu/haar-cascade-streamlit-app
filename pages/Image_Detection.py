import cv2
import streamlit as st
import numpy as np

# Digunakan untuk melakukan impor data haarcascade dari file xml untuk data muka kucing.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface_extended.xml')
# Digunakan untuk melakukan impor data haarcascade dari file xml untuk data muka manusia.
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
    # Digunakan untuk memberikan title pada halaman.
    st.title("Human and Cat Face Image Detector")
    # Digunakan untuk menampilkan data penjelesan dari variable text1 diatas.
    st.write(f"<p class='text-justify'>{text1}</p>", unsafe_allow_html=True)
    # Digunakan untuk menampilkan data penjelesan dari variable text2 diatas.
    st.write(f"<p class='text-justify'>{text2}</p>", unsafe_allow_html=True)
    # Digunakan untuk melakukan proses upload citra yang ingin dideteksi oleh program.
    uploaded_file = st.file_uploader("Pilih gambar yang akan dideteksi")
    # Memeriksa keberadaan citra (apakah ada atau tidak) dari hasil proses upload sebelumnya.
    if uploaded_file is not None:
        st.write("You have uploaded a file.")
        file_bytes = np.asarray(
            bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        # Mengubah citra menjadi grayscale.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Digunakan untuk mendeteksi muka kucing.
        faces = face_cascade.detectMultiScale(gray, 1.1, 6)
        # Digunakan untuk mendeteksi muka manusia.
        human = human_cascade.detectMultiScale(gray, 1.1, 6)

        # Digunakan untuk mendeteksi setiap bagian citra dan melakukan perulangan untuk memberikan tanda persegi dan label "Cat" pada daerah disekita muka kucing.
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 250), 2)
            cv2.putText(img, "Cat", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # Digunakan untuk mendeteksi setiap bagian citra dan melakukan perulangan untuk memberikan tanda persegi dan label "Human" pada daerah disekita muka manusia.
        for (x, y, w, h) in human:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 250), 2)
            cv2.putText(img, "Human", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # Menampilkan hasil deteksi dari proses sebelumnya lalu ditampilkan secara format BGR (Berwarna).
        st.image(img, channels="BGR", caption="Uploaded Image")


if __name__ == "__main__":
    main()

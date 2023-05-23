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
text1 = "Aplikasi ini bertujuan untuk mendeteksi wajah menggunakan algoritma pengolahan citra. Dalam aplikasi ini, terdapat dua jenis deteksi yaitu deteksi wajah pada manusia dan wajah kucing."
text2 = "Anda dapat menggunakan aplikasi ini untuk mengamati keberadaan wajah manusia dan kucing dalam sebuah kamera. Selamat mencoba!"
# Menampilkan tulisan yang sudah di justify


def main():
    # Digunakan untuk memberikan title pada halaman.
    st.title("Real-Time Human and Cat Face Detector")
    # Digunakan untuk menampilkan data penjelesan dari variable text1 diatas.
    st.write(f"<p class='text-justify'>{text1}</p>", unsafe_allow_html=True)
    # Digunakan untuk menampilkan data penjelesan dari variable text2 diatas.
    st.write(f"<p class='text-justify'>{text2}</p>", unsafe_allow_html=True)
    st.write("Tekan tombol Mulai untuk memulai :) ")

    # Digunakan untuk mendeteksi event berupa penekanan tombol mulai.
    run_detection = st.button("Mulai")

    # Digunakan untuk menjalankan pendeteksian melalui kamera device jika tombol mulai ditekan.
    if run_detection:
        # Digunakan untuk memulai pendeteksian melalui kamera.
        cap = cv2.VideoCapture(0)
        # Jika kamera tidak bisa dibuka maka akan memberikan peringatan berupa text .
        if not cap.isOpened():
            st.write("Unable to access camera")
            st.stop()
        
        # Menjalankan proses pendeteksian kamera serta melakukan proses pendeteksian objek / benda.
        while True:
            _, img = cap.read()

            # Digunakan untuk mengubah hasil deteksi menjadi grayscale.
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
            cv2.imshow('Detected Face Image',  img)

            # Digunakan untuk memeriksa apakah window yang menampilkan hasil kamera / memeriksa apakah kamera dijalankan atau tidak.
            if cv2.getWindowProperty('Detected Face Image', cv2.WND_PROP_VISIBLE) < 1:
                break

            # Digunakan untuk memberhentikan kamera dan window yang menampilkan hasil kamera.
            # Tombole yang digunakan yaitu ESC
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        # Menghapus semua hasil frame yang dihasilkan dari kamera dari device.
        cap.release()

        # Digunakan untuk menutup semua windows / halaman yang berkaitan dengan aplikasi.
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

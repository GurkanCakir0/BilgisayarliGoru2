import cv2
import numpy as np
import mediapipe as mp

# Mediapipe yüz algılama için ayar
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Webcam'i başlat
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam açılamadı!")
    exit()

# Hangi filtrenin aktif olduğunu belirlemek için bir değişken
filter_mode = None  # None: filtre yok, "gaussian": Gauss filtresi, "median": Medyan filtresi, "average": Ortalama filtresi, "sobel": Sobel filtresi, "prewitt": Prewitt filtresi, "laplacian": Laplacian filtresi

while True:
    # Kameradan kare oku
    ret, frame = cap.read()
    if not ret:
        print("Kare alınamadı!")
        break

    # RGB renk alanına dönüştür (Mediapipe RGB'yi kullanıyor)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Yüz algılama
    results = face_mesh.process(rgb_frame)

    # Yüz bulunduysa işlemleri yap
    if results.multi_face_landmarks:
        # İlk yüzü alıyoruz
        face_landmarks = results.multi_face_landmarks[0]

        # Yüz bölgesi için bir maske oluştur
        h, w, _ = frame.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        # Yüzün çevresindeki noktaları kullanarak yüz maskesini doldur
        face_points = []
        for lm in face_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            face_points.append((x, y))

        # Yüz konturunu çiz ve doldur
        face_points = np.array([face_points], dtype=np.int32)
        cv2.fillPoly(mask, face_points, 255)

        # Seçili filtreyi yüz bölgesine uygula
        if filter_mode == "gaussian":
            filtered_frame = cv2.GaussianBlur(frame, (15, 15), 0)
        elif filter_mode == "median":
            filtered_frame = cv2.medianBlur(frame, 15)
        elif filter_mode == "average":
            filtered_frame = cv2.blur(frame, (15, 15))
        elif filter_mode == "sobel":
            sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)
            filtered_frame = cv2.convertScaleAbs(sobelx + sobely)
        elif filter_mode == "prewitt":
            prewittx = cv2.filter2D(frame, -1, np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]))
            prewitty = cv2.filter2D(frame, -1, np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]))
            filtered_frame = cv2.convertScaleAbs(prewittx + prewitty)
        elif filter_mode == "laplacian":
            laplacian = cv2.Laplacian(frame, cv2.CV_64F)
            filtered_frame = cv2.convertScaleAbs(laplacian)
        else:
            filtered_frame = frame

        # Yalnızca yüz bölgesine filtre uygulayalım
        face_region = cv2.bitwise_and(filtered_frame, filtered_frame, mask=mask)
        inverse_mask = cv2.bitwise_not(mask)
        background_region = cv2.bitwise_and(frame, frame, mask=inverse_mask)
        combined_frame = cv2.add(background_region, face_region)
    else:
        # Yüz bulunamazsa orijinal görüntüyü göster
        combined_frame = frame

    # Görüntüyü göster
    cv2.imshow("Webcam", combined_frame)

    # Tuş girişini kontrol et
    key = cv2.waitKey(1) & 0xFF

    # Filtreleri tuşlara atayalım
    if key == ord("1"):
        filter_mode = "gaussian" if filter_mode != "gaussian" else None
    elif key == ord("2"):
        filter_mode = "median" if filter_mode != "median" else None
    elif key == ord("3"):
        filter_mode = "average" if filter_mode != "average" else None
    elif key == ord("4"):
        filter_mode = "sobel" if filter_mode != "sobel" else None
    elif key == ord("5"):
        filter_mode = "prewitt" if filter_mode != "prewitt" else None
    elif key == ord("6"):
        filter_mode = "laplacian" if filter_mode != "laplacian" else None
    elif key == ord("q"):
        break

# Kamerayı ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()
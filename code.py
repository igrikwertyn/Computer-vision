import cv2
import mediapipe as mp
import time

closed_time = None
blinks = 0
start_time = time.time()
yawned = 0
closed_mouth = None
closed_eyes = 0
prev_eyes = False
prev_yawn = False
sleeping_start_time = None

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Определяем индексы точек вокруг левого и правого глаза
left_eye_idxs = [362, 385, 387, 263, 373, 380]
right_eye_idxs = [33, 160, 158, 133, 153, 144]

# Индексы точек на губах
mouth_idxs = [13, 14, 312, 317, 38, 87]

# Индексы точек на подбородке и лбу
chin_idx = 152
n_idx = 168

def calculate_ear(eye_landmarks):
    """
    Вычисляет Eye Aspect Ratio (EAR) на основе координат точек глаза.

    Args:
        eye_landmarks: Список координат точек глаза.

    Returns:
        ear: Значение Eye Aspect Ratio.
    """
    vertical_dist_1 = abs(eye_landmarks[1].y - eye_landmarks[5].y)
    vertical_dist_2 = abs(eye_landmarks[2].y - eye_landmarks[4].y)
    horizontal_dist = abs(eye_landmarks[0].x - eye_landmarks[3].x)
    ear = (vertical_dist_1 + vertical_dist_2) / (2 * horizontal_dist)
    return ear

def calculate_yawn(mouth_landmarks):
    """
    Вычисляет коэффициент открытости рта (EAR) на основе координат точек рта.

    Args:
        mouth_landmarks: Список координат точек рта.

    Returns:
        ear: Значение коэффициента открытости рта (EAR).
    """
    vertical_dist_1 = abs(mouth_landmarks[1].y - mouth_landmarks[0].y)
    vertical_dist_2 = abs(mouth_landmarks[3].y - mouth_landmarks[2].y)
    vertical_dist_3 = abs(mouth_landmarks[5].y - mouth_landmarks[4].y)
    average = (vertical_dist_1 + vertical_dist_2 + vertical_dist_3) / 3
    return average

# Запуск видеопотока с камеры по умолчанию
cap = cv2.VideoCapture(0)

# Запуск модели для поиска точек лица
with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Не удалось получить кадр.")
            continue

        # Преобразование изображения из BGR в RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Обработка изображения для поиска точек лица
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Отрисовка точек вокруг левого глаза
                for idx in left_eye_idxs:
                    lm = face_landmarks.landmark[idx]
                    height, width, _ = image.shape
                    x, y = int(lm.x * width), int(lm.y * height)
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

                # Отрисовка точек вокруг правого глаза
                for idx in right_eye_idxs:
                    lm = face_landmarks.landmark[idx]
                    height, width, _ = image.shape
                    x, y = int(lm.x * width), int(lm.y * height)
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

                # Отрисовка точек рта
                for idx in mouth_idxs:
                    lm = face_landmarks.landmark[idx]
                    height, width, _ = image.shape
                    x, y = int(lm.x * width), int(lm.y * height)
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

                # Отрисовка точек на подбородке и лбу
                chin_lm = face_landmarks.landmark[chin_idx]
                chin_x, chin_y = int(chin_lm.x * width), int(chin_lm.y * height)
                cv2.circle(image, (chin_x, chin_y), 2, (0, 255, 0), -1)

                n_lm = face_landmarks.landmark[n_idx]
                n_x, n_y = int(n_lm.x * width), int(n_lm.y * height)
                cv2.circle(image, (n_x, n_y), 2, (0, 255, 0), -1)

                # Рассчитываем Eye Aspect Ratio (EAR)
                left_eye = [face_landmarks.landmark[i] for i in left_eye_idxs]
                right_eye = [face_landmarks.landmark[i] for i in right_eye_idxs]
                ear_left = calculate_ear(left_eye)
                ear_right = calculate_ear(right_eye)
                ear = (ear_left + ear_right) / 2
                cv2.putText(image, f'EAR: {ear:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            cv2.LINE_AA)

                #Расчитываем Mouth Aspect Ratio (MER)
                average_yawn = calculate_yawn([face_landmarks.landmark[i] for i in mouth_idxs])
                cv2.putText(image, f'MER: {average_yawn:.2f}', (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            cv2.LINE_AA)

                # Обновляем переменную closed_time в случае, если глаза открыты
                if ear >= 0.31:
                    if closed_time is not None:
                        # Подсчет морганий при переходе от закрытых глаз к открытым
                        blinks += 1
                        closed_time = None
                else:
                    if closed_time is None:
                        closed_time = time.time()
                    else:
                        elapsed_time = time.time() - closed_time
                        cv2.putText(image, f'{round(elapsed_time, 2)}c.', (475, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2,
                                    cv2.LINE_AA)
                        if elapsed_time >= 3:
                            if not prev_eyes:
                                closed_eyes += 1  # Увеличиваем счетчик closed_eyes на 1
                            prev_eyes = True
                            print(closed_eyes)
                            cv2.putText(image, "YOU'RE SLEEPING!", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                        cv2.LINE_AA)
                        else:
                            prev_eyes = False


                if average_yawn >= 0.05:
                    if not prev_yawn:
                        yawned += 1
                    prev_yawn = True
                    cv2.putText(image, "YOU YAWNED!", (150, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                cv2.LINE_AA)
                else:
                    prev_yawn = False

                # Проверяем положение лица
                dist_1 = abs(chin_y - n_y)
                if dist_1 <= 120:
                    cv2.putText(image, "KEEP YOUR HEAD STRAIGHT!", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Проверяем, прошла ли минута
        elapsed_time = time.time() - start_time
        if elapsed_time >= 60:
            # Рассчитываем частоту морганий в минуту (BPM)
            bpm = blinks
            print("Частота морганий в минуту:", bpm)
            start_time = time.time()
            blinks = 0

        # Отображаем количество зеваний на экране
        cv2.putText(image, f'Yawned: {yawned}', (10, 480), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    cv2.LINE_AA)

        if closed_eyes >= 3 and yawned >= 2:
            if sleeping_start_time is None:
                sleeping_start_time = time.time()
            else:
                elapsed_sleeping_time = time.time() - sleeping_start_time
                if elapsed_sleeping_time <= 5:
                    cv2.putText(image, "YOU'RE SLEEPING!", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                cv2.LINE_AA)
                else:
                    yawned = 0
                    closed_eyes = 0
                    sleeping_start_time = None

        # Отображение изображения с отмеченными точками
        cv2.imshow('Face Mesh', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

# Освобождение ресурсов и закрытие окон OpenCV
cap.release()
cv2.destroyAllWindows()

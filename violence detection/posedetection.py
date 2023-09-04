import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt

# MediaPipe 포즈 클래스 초기화
mp_pose = mp.solutions.pose

# 포즈_비디오 함수 설정
pose_video = mp_pose.Pose(static_image_mode = False, min_detection_confidence = 0.5, model_complexity = 1)

# MediaPipe 드로잉 클래스 초기화
mp_drawing = mp.solutions.drawing_utils

# 비디오캡쳐 오브젝트 읽기 초기화
video = cv2.VideoCapture(0)

# 창 설정
cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)

# 비디오 카메라 크기 설정
video.set(3, 1280)
video.set(4, 1280)

# 시간 변수 설정
time1 = 0


def detectPose(image, pose, display = True):

    output_image = image.copy()

    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    height, width, _ = image.shape
    landmarks = []

    if results.pose_landmarks :
        mp_drawing.draw_landmarks(image = output_image, landmark_list = results.pose_landmarks, connections = mp_pose.POSE_CONNECTIONS)

        for landmark in results.pose_landmarks.landmark:

            landmarks.append((int(landmark.x * width), int(landmark.y * height), landmark.z * width))
        
    if display:

        plt.figure(figsize = [22, 22])

    return output_image
        
                            

# 비디오에 성공적으로 액세스할 때까지 반복
while video.isOpened():

    # 프레임 읽기
    ok, frame = video.read()

    # 프레임이 제대로 읽히지 않는지 확인
    if not ok:

        break
    
    # 셀피 카메라 전환
    frame = cv2.flip(frame, 1)
    # 프레임 정보 받아들이기
    frame_height, frame_width, _ = frame.shape
    # 가로 세로 비율을 유지하면서 프레임 크기 조정
    frame = cv2.resize(frame, (int(frame_width * (1280 / frame_height)), 1280))
    # 포즈 랜드마크 검출을 수행
    frame = detectPose(frame, pose_video, display = False)

    time2 = time()

    if (time2 - time1 > 0):
        frames_per_second = 10.0 / (time2 - time1)

        cv2.putText(frame, 'FPS: {}'.format(int(frames_per_second)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (55, 255, 55), 3)

    time1 = time2

    cv2.imshow('Pose Detection', frame)

    k = cv2.waitKey(1) & 0xFF
    if(k == 27):
        break


video.release()

cv2.destroyAllWindows()
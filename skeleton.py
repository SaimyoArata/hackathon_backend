import os
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from pose_analysis import landmark2np, manual_cos

os.environ["CUDA_VISIBLE_DEVICES"] = ""

def func(target_image, player_image):
    # PIL ImageオブジェクトをNumPy配列へ変換
    target_image_np = np.array(target_image)
    player_image_np = np.array(player_image)
    # 必要に応じてRGBからBGRに変換（OpenCVはBGRフォーマットを使用）
    target_image_bgr = cv2.cvtColor(target_image_np, cv2.COLOR_RGB2BGR)
    player_image_bgr = cv2.cvtColor(player_image_np, cv2.COLOR_RGB2BGR)

    # Mediapipeの初期化
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
        #use_gpu=False  # GPUを使用しない設定
    )
    # Mediapipeの姿勢推定を実行
    target_results = pose.process(target_image_bgr)
    player_results = pose.process(player_image_bgr)

    # 姿勢推定の結果があれば、ランドマークを描画
    if player_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            player_image_bgr,
            player_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )
    
    if player_results.pose_landmarks:
        if target_results.pose_landmarks:
            target_mortion = landmark2np(target_results.pose_landmarks)
            player_mortion = landmark2np(player_results.pose_landmarks)
      
        score = manual_cos(target_mortion, player_mortion)

        # 右手の座標を取得してプリント
        # right_hand_index = 16  # 右手のキーポイントのインデックス
        # right_hand_landmark = results.pose_landmarks.landmark[right_hand_index]
        # right_hand_coordinates = (right_hand_landmark.x, right_hand_landmark.y)
        # print(f"右手の座標: {right_hand_coordinates}")

    pose.close()
    return Image.fromarray(cv2.cvtColor(player_image_bgr, cv2.COLOR_BGR2RGB)), score

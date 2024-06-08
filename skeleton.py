import os
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = ""

def func(image):
    # PIL ImageオブジェクトをNumPy配列へ変換
    image_np = np.array(image)

    # 必要に応じてRGBからBGRに変換（OpenCVはBGRフォーマットを使用）
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Mediapipeの初期化
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
        #use_gpu=False  # GPUを使用しない設定
    )

    # # 画像ファイルの読み込み
    # image_rgb = cv2.imread(image)
    # # Mediapipeの姿勢推定を実行
    # results = pose.process(image_rgb)

    # Mediapipeの姿勢推定を実行
    results = pose.process(image_bgr)

    # 姿勢推定の結果があれば、ランドマークを描画
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image_bgr,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        # 右手の座標を取得してプリント
        # right_hand_index = 16  # 右手のキーポイントのインデックス
        # right_hand_landmark = results.pose_landmarks.landmark[right_hand_index]
        # right_hand_coordinates = (right_hand_landmark.x, right_hand_landmark.y)
        # print(f"右手の座標: {right_hand_coordinates}")

    pose.close()
    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
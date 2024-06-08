import cv2
import mediapipe as mp

def func(image):
    # Mediapipeの初期化
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
        use_gpu=False  # GPUを使用しない設定
    )

    # 画像ファイルの読み込み
    image = cv2.imread(image)
    # Mediapipeの姿勢推定を実行
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    # 姿勢推定の結果があれば、ランドマークを描画
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        # 右手の座標を取得してプリント
        # right_hand_index = 16  # 右手のキーポイントのインデックス
        # right_hand_landmark = results.pose_landmarks.landmark[right_hand_index]
        # right_hand_coordinates = (right_hand_landmark.x, right_hand_landmark.y)
        # print(f"右手の座標: {right_hand_coordinates}")

    pose.close()
    return image
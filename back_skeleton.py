# 骨格検出する関数
# 参考サイト：https://qiita.com/michelle0915/items/f69e6255595fe82799b8
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import keypointrcnn_resnet50_fpn
import os

KEYPOINT_THRESHOLD = 0.2


def skeleton_detection(image_path, output_path):
    # モデルの読み込み
    model = keypointrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # 画像を読み込む
    image = cv2.imread(image_path)
    if image is None:
        print("Error: image not found")
        return

    # 推論実行
    keypoints, scores, bboxes = run_inference(model, image)

    # レンダリング画像
    result_image = render_skeleton(image, keypoints, scores, bboxes)

    # 結果を保存
    cv2.imwrite(output_path, result_image)
    # cv2.namedWindow("result", cv2.WINDOW_FULLSCREEN)
    # cv2.imshow("result", result_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# 推論実行
def run_inference(model, image):
    # 画像の前処理
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    input_image = transform(image)
    input_image = input_image.unsqueeze(0)

    # 推論実行
    with torch.no_grad():
        prediction = model(input_image)

    # 結果の取得
    prediction = prediction[0]
    keypoints = prediction[0]["keypoints"][0].cpu().numpy()
    scores = prediction[0]["keypoints_scores"][0].cpu().numpy()
    bboxes = prediction[0]["boxes"][0].cpu().numpy()

    keypoints = keypoints.astype(int)

    return keypoints, scores, bboxes

# 骨格を描画
def render_skeleton(image, keypoints, scores, bboxes):
    render = image.copy()
    for i, (keypoints, scores, bboxes) in enumerate(zip(keypoints, scores, bboxes)):
        if scores[i] < KEYPOINT_THRESHOLD:
            continue

        cv2.rectangle(render, (int(bboxes[0]), int(bboxes[1])), (int(bboxes[2]), int(bboxes[3]), (0, 255, 0), 2))

        # 骨格の組み合わせ（0: 鼻, 1: 左目, 2: 右目, 3: 左耳, 4: 右耳, 5: 左肩, 6: 右肩, 7: 左肘, 8: 右肘, 9: 左手首, 10: 右手首, 11: 左腰, 12: 右腰, 13: 左膝, 14: 右膝, 15: 左足首, 16: 右足首）
        kp_links = [
            (0, 1), (0, 2), (1, 3), (2, 4), # 顔
            (0, 5), (0, 6), (5, 6), # 鼻から肩
            (5, 7), (7, 9), # 左腕
            (6, 8), (8, 10),# 右腕
            (11, 12),# 腰
            (5, 11), (11, 13), (13, 15), # 左脚
            (6, 12), (12, 14), (14, 16)  # 右脚
        ]

        for kp_idx_1, kp_idx_2 in kp_links:
            kp_1 = keypoints[kp_idx_1]
            kp_2 = keypoints[kp_idx_2]
            if scores[i][kp_idx_1] > KEYPOINT_THRESHOLD and scores[i][kp_idx_2] > KEYPOINT_THRESHOLD:
                cv2.line(render, tuple(kp_1[:2]), tuple(kp_2[:2]), (0, 0, 255), 2)

        for idx, keypoint in enumerate(keypoints):
            if scores[i][idx] > KEYPOINT_THRESHOLD:
                cv2.circle(render, tuple(keypoint[:2]), 4, (0, 0, 255), -1)

    return render


if __name__ == '__main__':
    # image_path = "./"
    # skeleton_detection(image_path)
    input_folder = './example_posture'
    output_folder = './example_posture_skeleton'
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            image_path = os.path.join(input_folder, filename)
            filename = filename.replace('.jpg', '_skeleton.jpg').replace('.jpeg', '_skeleton.jpeg').replace('.png', '_skeleton.png')
            output_path = os.path.join(output_folder, filename)
            skeleton_detection(image_path, output_path)
            print(f"Processed and saved: {output_path}")
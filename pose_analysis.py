import numpy as np

#入力(pose_landmarks)：姿勢推定から得られたランドマークのデータ
#出力：12の点(右肩)からの相対座標(numpy配列)
def landmark2np(pose_landmarks):
    detected_point = 12
    li = []
    for j in pose_landmarks.landmark:
        li.append([j.x, j.y, j.z])
    for i, k in enumerate(li):
        if k[0] == 0 and k[1] == 0 and k[2] == 0:
            print("No detected")
            li[i] = li[detected_point]

    return np.array(li) - np.array(li[detected_point])


#入力：比較する2つの座標ランドマーク群
#出力：各ベクトルごとに比較したコサイン平均類似度
def manual_cos(A, B):
    dot = np.sum(A * B, axis=-1)
    A_norm = np.linalg.norm(A, axis=-1)
    B_norm = np.linalg.norm(B, axis=-1)
    cos = dot / (A_norm * B_norm + 1e-10)

    # 検出できない場合の処理
    cos = np.array([i for i in cos if i != 0])

    if len(cos) == 0:
        print("No valid cos values found")
        return 0

    return cos.mean()

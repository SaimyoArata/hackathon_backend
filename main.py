from flask import Flask, request, jsonify
import cv2
import io
import base64
import numpy as np

app = Flask(__name__)

def process_image(image_data):
    # 画像データをバイナリデータにデコード
    image_bytes = base64.b64decode(image_data)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # 画像をグレースケールに変換（ここでは例としてグレースケールに変換しているが、任意の画像処理を行っても良い）
    processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 処理された画像をバイナリデータに変換
    _, buffered = cv2.imencode('.jpg', processed_image)
    encoded_image = base64.b64encode(buffered).decode('utf-8')

    return encoded_image

@app.route('/upload', methods=['POST'])
def upload():
    try:
        data = request.get_json()
        image_data = data['image']

        processed_image = process_image(image_data)

        return jsonify({'processed_image': processed_image})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# if __name__ == '__main__':
#     app.run(debug=True, port=5000)



# from fastapi import FastAPI

# app = FastAPI()


# @app.get("/")
# async def root():
#     def hello_world():
#         return "Hello World"
#     # ここで関数を呼び出す
#     result = hello_world()
#     return {"message": result}

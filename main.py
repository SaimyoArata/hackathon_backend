from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
from fastapi.responses import StreamingResponse
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import io
from fastapi import UploadFile, File 
from skeleton import func
from pose_analysis import landmark2np, manual_cos

app = FastAPI()

if __name__ == '__main__':
    app.run(debug=True, port=8000)

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["score"],
)


class ImageData(BaseModel):
    image: str


@app.post("/")
async def upload_image(
    data1: UploadFile = File(...), # target_image
    data2: UploadFile = File(...)  # player_image
):
    try:
        # 0.5秒たいき
        # time.sleep(0.5)
        # 画像のバイナリデータを読み込み
        player_image = Image.open(io.BytesIO(await data1.read()))
        target_image = Image.open(io.BytesIO(await data2.read()))
        # 画像を表示
        # target_image.show()
        # player_image.show()

    
        # 骨格検出後の画像とランドマーク(list(tuple))を取得
        processed_image, score = func(target_image, player_image)

        # 処理された画像をバイナリデータに変換
        print("Encoding processed image")
        buffered = BytesIO()
        processed_image.save(buffered, format="JPEG")
        buffered.seek(0)

        # 画像とスコアをレスポンスする
        response = StreamingResponse(buffered, media_type="image/jpeg")
        response.headers["score"] = str(score)

        return response


        # return {"message": "Image processed successfully", "processed_image": encoded_image}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

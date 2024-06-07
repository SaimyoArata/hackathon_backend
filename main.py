from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
from fastapi.responses import StreamingResponse
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import io
from fastapi import UploadFile, File

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
)


class ImageData(BaseModel):
    image: str


@app.post("/")
async def upload_image(data: UploadFile = File(...)):
    # 画像のバイナリデータを読み込み
    image = Image.open(io.BytesIO(await data.read()))
    # 画像を表示
    # image.show()

    try:
        # 画像をグレースケールに変換
        print("Processing image")
        processed_image = image.convert("L")

        # 処理された画像をバイナリデータに変換
        print("Encoding processed image")
        buffered = BytesIO()
        processed_image.save(buffered, format="JPEG")
        buffered.seek(0)

        # return {"message": "Image processed successfully", "processed_image": encoded_image}
        return StreamingResponse(buffered, media_type="image/jpeg")
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image, UnidentifiedImageError

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
async def upload_image(data: ImageData):
    try:
        # Base64エンコードされた画像データをデコード
        try:
            print("Decoding image")
            image_data = base64.b64decode(data.image.split(",")[1])  # 'data:image/jpeg;base64,' を除去
        except base64.binascii.Error as decode_error:
            raise HTTPException(status_code=400, detail="Invalid base64 string")

        try:
            print("Opening image")
            image = Image.open(BytesIO(image_data))
        except UnidentifiedImageError as img_error:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # 画像をグレースケールに変換
        print("Processing image")
        processed_image = image.convert("L")

        # 処理された画像をバイナリデータに変換
        print("Encoding processed image")
        buffered = BytesIO()
        processed_image.save(buffered, format="JPEG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return {"message": "Image processed successfully", "processed_image": encoded_image}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

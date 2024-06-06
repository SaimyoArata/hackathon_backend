from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np

app = FastAPI()

# 静的ファイルを提供する設定
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 画像を表示（サーバー側）
    cv2.imshow('Uploaded Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return {"filename": file.filename}

@app.get("/")
async def main():
    content = """
    <html>
        <body>
            <h1>Upload an image</h1>
            <form action="/uploadfile/" enctype="multipart/form-data" method="post">
                <input name="file" type="file">
                <input type="submit">
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=content)




# from fastapi import FastAPI

# app = FastAPI()


# @app.get("/")
# async def root():
#     def hello_world():
#         return "Hello World"
#     # ここで関数を呼び出す
#     result = hello_world()
#     return {"message": result}

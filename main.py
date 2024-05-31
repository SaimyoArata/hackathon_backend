from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    def hello_world():
        return "Hello World"
    # ここで関数を呼び出す
    result = hello_world()
    return {"message": "result"}

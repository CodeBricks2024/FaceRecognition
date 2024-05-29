from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
import aiomysql
import os

app = FastAPI()


# 데이터베이스에 이미지 메타데이터 저장
async def save_image_to_database(filename, filepath):
    try:
        conn = await aiomysql.connect(
            host="127.0.0.1",
            port=3306,
            user="myuser",
            password="mypassword",
            db="mydatabase",
            charset="utf8",
        )
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO images (filename, filepath) VALUES (%s, %s)",
                (filename, filepath),
            )
            await conn.commit()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 파일 업로드 엔드포인트
@app.post("/upload/")
async def upload_image(
    file: UploadFile = File(...), filename: str = Form(...), filepath: str = Form(...)
):
    try:
        # 파일 저장 폴더가 없으면 생성
        upload_folder = "uploads"
        os.makedirs(upload_folder, exist_ok=True)

        # 파일 전체 경로 설정
        file_location = os.path.join(upload_folder, filename)

        # 파일 시스템에 파일 쓰기
        with open(file_location, "wb+") as file_object:
            file_object.write(await file.read())

        # 데이터베이스에 파일 메타데이터 저장
        await save_image_to_database(filename, filepath)
        return {"info": "File saved", "filename": filename, "filepath": filepath}
    except Exception as e:
        return {"error": str(e)}


# 파비콘 제공 엔드포인트
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    file_path = os.path.join(os.path.dirname(__file__), "static", "favicon.ico")
    return FileResponse(file_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

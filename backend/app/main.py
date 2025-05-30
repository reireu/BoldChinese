from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.routers.analyze import router as analyze_router
import logging

# ロギングの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BoldChinese API",
    description="中国語発音分析APIサービス",
    version="1.0.0"
)

# CORSミドルウェアの設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 必要に応じて制限
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
# ルーターの登録
app.include_router(
    analyze_router,
    prefix="/api/v1",
    tags=["pronunciation"]
)

@app.get("/")
async def root():
    try:
        return {"message": "Welcome to BoldChinese API", "status": "healthy"}
    except Exception as e:
        logger.error(f"Error in root endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
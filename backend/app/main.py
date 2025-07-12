from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.routers.analyze import router as analyze_router
import logging
import time

# FastAPIアプリケーションの初期化
app = FastAPI(
    title="BoldChinese API",
    description="中国語発音分析APIサービス",
    version="1.0.0"
)

# CORSミドルウェア設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time"]
)

# ルートエンドポイント定義
@app.get("/")
async def root():
    return {
        "status": "online",
        "version": "1.0.0",
        "docs_url": "/docs",
    }

# analyzeルーター登録
app.include_router(
    analyze_router,
    prefix="/api/v1/analyze",
    tags=["pronunciation"]
)
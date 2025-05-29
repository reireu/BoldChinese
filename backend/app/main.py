from fastapi import FastAPI
from app.routers.analyze import router as analyze_router

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to BoldChinese API"}

app.include_router(analyze_router, prefix="/analyze")
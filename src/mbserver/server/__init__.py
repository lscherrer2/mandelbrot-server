from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mbserver.server.routes import router


app = FastAPI(debug=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Connected Successfully to API"}


app.include_router(router)
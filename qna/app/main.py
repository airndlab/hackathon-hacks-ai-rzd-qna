#  Copyright 2024 AI RnD Lab
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import logging
import sys
import uuid

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from app.chats import init_db, save_answer, set_feedback
from app.pipeline import get_answer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

app = FastAPI()


@app.get("/")
async def root():
    return {"status": "UP"}


# Модель вопроса
class Question(BaseModel):
    question: str
    chat_id: str


# Модель ответа
class Answer(BaseModel):
    id: str
    answer: str


@app.post("/api/answers", response_model=Answer)
async def ask(request: Question) -> Answer:
    answer_response = await get_answer(request.question)
    answer = answer_response.answer
    answer_id = str(uuid.uuid4())
    await save_answer(
        answer_id=answer_id,
        chat_id=request.chat_id,
        question=request.question,
        answer=answer,
    )
    return Answer(id=answer_id, answer=answer)


@app.post("/api/answers/{answer_id}/liking")
async def like(answer_id: str) -> None:
    await set_feedback(answer_id, 1)


@app.post("/api/answers/{answer_id}/disliking")
async def dislike(answer_id: str) -> None:
    await set_feedback(answer_id, -1)


@app.on_event("startup")
async def startup_event():
    await init_db()


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8080)

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
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.db import init_db, save_answer, set_feedback, save_chat, get_chat, set_profile, Chat
from app.pipeline import get_answer
from app.profiles import get_profiles, Profile

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
    chatId: str


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
        chat_id=request.chatId,
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


@app.get("/api/profiles", response_model=List[Profile])
async def profiles() -> List[Profile]:
    return await get_profiles()


class NewChat(BaseModel):
    id: str
    username: Optional[str]
    type: Optional[str]


@app.post("/api/chats", response_model=Chat)
async def new_chat(request: NewChat) -> Chat:
    await save_chat(request.id, request.username, request.type)
    return await get_chat(request.id)


@app.get("/api/chats/{chat_id}", response_model=Chat)
async def chat(chat_id: str) -> Chat:
    result = await get_chat(chat_id)
    if result:
        return result
    raise HTTPException(status_code=404, detail="Item not found")


@app.patch("/api/chats/{chat_id}/profiles/{profile}")
async def chat(chat_id: str, profile: str) -> None:
    await set_profile(chat_id, profile)


@app.on_event("startup")
async def startup_event():
    await init_db()


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8080)

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
import threading
import uuid
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.db import init_db, save_answer, set_feedback, save_chat, get_chat, set_profile, Chat, Profile
from app.indexing import run_indexing_manually, start_observer
from app.pipeline import get_answer
from app.profiles import get_profiles, get_profile, get_default_profile

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
    chat_id = request.chatId
    question = request.question
    curr_chat = await get_chat(chat_id)
    prof = curr_chat.as_profile()
    answer_response = await get_answer(question)
    answer = answer_response.answer
    answer_id = str(uuid.uuid4())
    await save_answer(
        answer_id=answer_id,
        chat_id=chat_id,
        question=question,
        answer=answer,
        profile=prof
    )
    return Answer(id=answer_id, answer=answer)


@app.post("/api/answers/{answer_id}/liking")
async def like(answer_id: str) -> None:
    await set_feedback(answer_id, 1)


@app.post("/api/answers/{answer_id}/disliking")
async def dislike(answer_id: str) -> None:
    await set_feedback(answer_id, -1)


# Вспомогательный функционал для демонстрации

@app.get("/api/profiles", response_model=List[Profile])
async def profiles() -> List[Profile]:
    return await get_profiles()


class NewChat(BaseModel):
    id: str
    username: Optional[str]
    type: Optional[str]


@app.post("/api/chats", response_model=Chat)
async def new_chat(request: NewChat) -> Chat:
    chat_id = request.id
    prof = await get_default_profile()
    await save_chat(
        chat_id=chat_id,
        username=request.username,
        chat_type=request.type,
        profile=prof
    )
    return await get_chat(chat_id)


@app.get("/api/chats/{chat_id}", response_model=Chat)
async def chat(chat_id: str) -> Chat:
    result = await get_chat(chat_id)
    if result:
        return result
    raise HTTPException(status_code=404, detail="Chat not found")


@app.patch("/api/chats/{chat_id}/profiles/{profile}")
async def patch_chat(chat_id: str, profile: str) -> None:
    prof = await get_profile(profile)
    await set_profile(chat_id, prof)


class PutChatProfile(BaseModel):
    profile: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    person_name: Optional[str] = None
    position: Optional[str] = None
    organization: Optional[str] = None
    region: Optional[str] = None
    sex: Optional[str] = None
    age: Optional[int] = None
    child_count: Optional[int] = None
    work_years: Optional[int] = None
    veteran_of_labor: Optional[bool] = None
    llm_request: Optional[str] = None
    details_md: Optional[str] = None


# noinspection DuplicatedCode
@app.put("/api/chats/{chat_id}/profiles")
async def put_chat(chat_id: str, request: PutChatProfile) -> None:
    curr_chat = await get_chat(chat_id)
    prof = curr_chat.as_profile()
    if request.profile is not None:
        prof.id = request.profile
    if request.title is not None:
        prof.title = request.title
    if request.description is not None:
        prof.description = request.description
    if request.person_name is not None:
        prof.person_name = request.person_name
    if request.position is not None:
        prof.position = request.position
    if request.organization is not None:
        prof.organization = request.organization
    if request.region is not None:
        prof.region = request.region
    if request.sex is not None:
        prof.sex = request.sex
    if request.age is not None:
        prof.age = request.age
    if request.child_count is not None:
        prof.child_count = request.child_count
    if request.work_years is not None:
        prof.work_years = request.work_years
    if request.veteran_of_labor is not None:
        prof.veteran_of_labor = request.veteran_of_labor
    if request.llm_request is not None:
        prof.llm_request = request.llm_request
    if request.details_md is not None:
        prof.details_md = request.details_md
    await set_profile(chat_id, prof)


@app.post("/api/indexing")
async def indexing():
    await run_indexing_manually()
    return {'status': 'started'}


@app.on_event("startup")
async def startup_event():
    await init_db()
    thread = threading.Thread(target=start_observer, daemon=True)
    thread.start()


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8080)

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
import os
from typing import Optional

import aiosqlite
from pydantic import BaseModel

logger = logging.getLogger(__name__)

QNA_DB_PATH = os.getenv('QNA_DB_PATH', 'qna.db')


# Инициализация базы данных
async def init_db() -> None:
    async with aiosqlite.connect(QNA_DB_PATH) as db:
        await db.execute('''
            CREATE TABLE IF NOT EXISTS answers (
                answer_id TEXT PRIMARY KEY,
                chat_id TEXT,
                question TEXT,
                answer TEXT,
                answered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                feedback INTEGER DEFAULT 0,
                profile TEXT,
                person_name TEXT,
                organization TEXT,
                region TEXT,
                sex TEXT,
                age INTEGER,
                child_count INTEGER,
                work_years INTEGER,
                veteran_of_labor BOOLEAN
            )
        ''')
        await db.execute('''
            CREATE TABLE IF NOT EXISTS chats (
                chat_id TEXT PRIMARY KEY,
                username TEXT,
                type TEXT,
                profile TEXT,
                person_name TEXT,
                organization TEXT,
                region TEXT,
                sex TEXT,
                age INTEGER,
                child_count INTEGER,
                work_years INTEGER,
                veteran_of_labor BOOLEAN
            )
        ''')
        await db.commit()


class Profile(BaseModel):
    id: str
    title: str
    description: str
    person_name: str
    organization: str
    region: str
    sex: str
    age: int
    child_count: int
    work_years: int
    veteran_of_labor: bool
    llm_request: str
    details_md: str


# Сохранить ответ
async def save_answer(answer_id: str, chat_id: str, question: str, answer: str, profile: Profile) -> None:
    async with aiosqlite.connect(QNA_DB_PATH) as db:
        await db.execute('''
            INSERT INTO answers (
                answer_id, chat_id, question, answer,
                profile, person_name, organization, region, sex, age, child_count, work_years, veteran_of_labor
            ) 
            VALUES (
                ?, ?, ?, ?, 
                ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        ''', (
            answer_id, chat_id, question, answer,
            profile.id, profile.person_name, profile.organization, profile.region, profile.sex,
            profile.age, profile.child_count, profile.work_years, profile.veteran_of_labor
        ))
        await db.commit()
        logger.info(
            f'saved answer: answer_id="{answer_id}" chat_id="{chat_id}" '
            f'question="{question}" answer="{answer}" profile={profile}'
        )


# Проставить фидбек
async def set_feedback(answer_id: str, feedback: int) -> None:
    async with aiosqlite.connect(QNA_DB_PATH) as db:
        await db.execute('''
            UPDATE answers 
            SET feedback = ? 
            WHERE answer_id = ?
        ''', (feedback, answer_id))
        await db.commit()
        logger.info(f'set feedback: answer_id="{answer_id}" feedback="{feedback}"')


class Chat(BaseModel):
    id: str
    username: Optional[str]
    type: Optional[str]
    profile: str
    title: str
    description: str
    person_name: str
    organization: str
    region: str
    sex: str
    age: int
    child_count: int
    work_years: int
    veteran_of_labor: bool
    llm_request: str
    details_md: str


async def save_chat(chat_id: str, username, chat_type, profile: Profile) -> None:
    async with aiosqlite.connect(QNA_DB_PATH) as db:
        await db.execute('''
            INSERT INTO chats (
                chat_id, username, type,
                profile, person_name, organization, region, sex, age, child_count, work_years, veteran_of_labor,
                llm_request, details_md
            ) 
            VALUES (
                ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?
            )
        ''', (
            chat_id, username, chat_type,
            profile.id, profile.person_name, profile.organization, profile.region, profile.sex,
            profile.age, profile.child_count, profile.work_years, profile.veteran_of_labor,
            profile.llm_request, profile.details_md
        ))
        await db.commit()
        logger.info(
            f'saved chat: chat_id="{chat_id}" username="{username} type="{chat_type}" profile="{profile}"'
        )


async def get_chat(chat_id: str) -> Optional[Chat]:
    async with aiosqlite.connect(QNA_DB_PATH) as db:
        async with db.execute("SELECT * FROM chats WHERE chat_id = ?", (chat_id,)) as cursor:
            chat = await cursor.fetchone()
            if chat:
                return Chat(
                    id=chat[0],
                    username=chat[1],
                    type=chat[2],
                    profile=chat[3],
                    title=chat[4],
                    description=chat[5],
                    person_name=chat[6],
                    organization=chat[7],
                    region=chat[8],
                    sex=chat[9],
                    age=chat[10],
                    child_count=chat[11],
                    work_years=chat[12],
                    veteran_of_labor=chat[13],
                    llm_request=chat[14],
                    details_md=chat[15],
                )
            else:
                return None


async def set_profile(chat_id: str, profile: Profile) -> None:
    async with aiosqlite.connect(QNA_DB_PATH) as db:
        await db.execute('''
            UPDATE chats 
            SET profile = ?, person_name = ?, organization = ?, region = ?, sex = ?,
                age = ?, child_count = ?, work_years = ?, veteran_of_labor = ?,
                llm_request = ?, details_md = ?
            WHERE chat_id = ?
        ''', (
            profile.id, profile.person_name, profile.organization, profile.region, profile.sex,
            profile.age, profile.child_count, profile.work_years, profile.veteran_of_labor,
            profile.llm_request, profile.details_md,
            chat_id
        ))
        await db.commit()
        logger.info(f'set profile: chat_id="{chat_id}" profile="{profile}"')

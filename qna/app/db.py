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
                answer_id TEXT,
                chat_id TEXT,
                question TEXT,
                answer TEXT,
                answered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                feedback INTEGER DEFAULT 0,
                profile TEXT,
                person_name TEXT,
                position TEXT,
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
                chat_id TEXT,
                username TEXT,
                type TEXT,
                profile TEXT,
                title TEXT,
                description TEXT,
                person_name TEXT,
                position TEXT,
                organization TEXT,
                region TEXT,
                sex TEXT,
                age INTEGER,
                child_count INTEGER,
                work_years INTEGER,
                veteran_of_labor BOOLEAN,
                llm_request TEXT,
                details_md TEXT
            )
        ''')
        await db.execute('''
            CREATE TABLE IF NOT EXISTS indexing (
                id TEXT,
                status TEXT,
                way TEXT,
                started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                finished_at DATETIME DEFAULT NULL
            )
        ''')
        await db.commit()


class Profile(BaseModel):
    id: str
    title: Optional[str]
    description: Optional[str]
    person_name: Optional[str]
    position: Optional[str]
    organization: Optional[str]
    region: Optional[str]
    sex: Optional[str]
    age: Optional[int]
    child_count: Optional[int]
    work_years: Optional[int]
    veteran_of_labor: Optional[bool]
    llm_request: Optional[str]
    details_md: Optional[str]


# Сохранить ответ
async def save_answer(answer_id: str, chat_id: str, question: str, answer: str, profile: Profile) -> None:
    async with aiosqlite.connect(QNA_DB_PATH) as db:
        await db.execute('''
            INSERT INTO answers (
                answer_id, chat_id, question, answer,
                profile, person_name, position, organization, region, sex, age, child_count, work_years, veteran_of_labor
            ) 
            VALUES (
                ?, ?, ?, ?, 
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        ''', (
            answer_id, chat_id, question, answer,
            profile.id, profile.person_name, profile.position, profile.organization, profile.region, profile.sex,
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
    profile: Optional[str]
    title: Optional[str]
    description: Optional[str]
    person_name: Optional[str]
    position: Optional[str]
    organization: Optional[str]
    region: Optional[str]
    sex: Optional[str]
    age: Optional[int]
    child_count: Optional[int]
    work_years: Optional[int]
    veteran_of_labor: Optional[bool]
    llm_request: Optional[str]
    details_md: Optional[str]

    def as_profile(self) -> Profile:
        return Profile(
            id=self.profile,
            title=self.title,
            description=self.description,
            person_name=self.person_name,
            position=self.position,
            organization=self.organization,
            region=self.region,
            sex=self.sex,
            age=self.age,
            child_count=self.child_count,
            work_years=self.work_years,
            veteran_of_labor=self.veteran_of_labor,
            llm_request=self.llm_request,
            details_md=self.details_md
        )


async def save_chat(chat_id: str, username, chat_type, profile: Profile) -> None:
    async with aiosqlite.connect(QNA_DB_PATH) as db:
        await db.execute('''
            INSERT INTO chats (
                chat_id, username, type,
                profile, title, description, person_name, position, organization, region, sex,
                age, child_count, work_years, veteran_of_labor,
                llm_request, details_md
            ) 
            VALUES (
                ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?
            )
        ''', (
            chat_id, username, chat_type,
            profile.id, profile.title, profile.description, profile.person_name,
            profile.position, profile.organization, profile.region, profile.sex,
            profile.age, profile.child_count, profile.work_years, profile.veteran_of_labor,
            profile.llm_request, profile.details_md
        ))
        await db.commit()
        logger.info(
            f'saved chat: chat_id="{chat_id}" username="{username} type="{chat_type}" profile="{profile}"'
        )


async def get_chat(chat_id: str) -> Optional[Chat]:
    async with aiosqlite.connect(QNA_DB_PATH) as db:
        async with db.execute('''
                SELECT chat_id, username, type,
                    profile, title, description, person_name, position, organization, region, sex,
                    age, child_count, work_years, veteran_of_labor,
                    llm_request, details_md
                FROM chats WHERE chat_id = ?
            ''', (chat_id,)) as cursor:
            chat = await cursor.fetchone()
            print(chat)
            if chat:
                return Chat(
                    id=chat[0],
                    username=chat[1],
                    type=chat[2],
                    profile=chat[3],
                    title=chat[4],
                    description=chat[5],
                    person_name=chat[6],
                    position=chat[7],
                    organization=chat[8],
                    region=chat[9],
                    sex=chat[10],
                    age=chat[11],
                    child_count=chat[12],
                    work_years=chat[13],
                    veteran_of_labor=chat[14],
                    llm_request=chat[15],
                    details_md=chat[16],
                )
            else:
                return None


async def set_profile(chat_id: str, profile: Profile) -> None:
    async with aiosqlite.connect(QNA_DB_PATH) as db:
        await db.execute('''
            UPDATE chats 
            SET profile = ?, title = ?, description = ?, person_name = ?, position = ?, organization = ?, region = ?, sex = ?,
                age = ?, child_count = ?, work_years = ?, veteran_of_labor = ?,
                llm_request = ?, details_md = ?
            WHERE chat_id = ?
        ''', (
            profile.id, profile.title, profile.description, profile.person_name,
            profile.position, profile.organization, profile.region, profile.sex,
            profile.age, profile.child_count, profile.work_years, profile.veteran_of_labor,
            profile.llm_request, profile.details_md,
            chat_id
        ))
        await db.commit()
        logger.info(f'set profile: chat_id="{chat_id}" profile="{profile}"')


async def save_indexing(indexing_id: str, status: str, way: str) -> None:
    async with aiosqlite.connect(QNA_DB_PATH) as db:
        await db.execute('''
            INSERT INTO indexing (id, status, way) 
            VALUES (?, ?, ?)
        ''', (indexing_id, status, way))
        await db.commit()
        logger.info(
            f'saved indexing: indexing_id="{indexing_id}" status="{status} way="{way}"'
        )


async def set_indexing(indexing_id: str, status: str) -> None:
    async with aiosqlite.connect(QNA_DB_PATH) as db:
        await db.execute('''
            UPDATE indexing 
            SET status = ?, finished_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (indexing_id, status))
        await db.commit()
        logger.info(f'set indexing: indexing_id="{indexing_id}" status="{status}"')

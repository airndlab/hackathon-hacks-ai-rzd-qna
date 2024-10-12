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
                feedback INTEGER DEFAULT 0
            )
        ''')
        await db.execute('''
            CREATE TABLE IF NOT EXISTS chats (
                chat_id TEXT PRIMARY KEY,
                username TEXT,
                type TEXT,
                profile TEXT DEFAULT 'default'
            )
        ''')
        await db.commit()


# Сохранить ответ
async def save_answer(answer_id: str, chat_id: str, question: str, answer: str) -> None:
    async with aiosqlite.connect(QNA_DB_PATH) as db:
        await db.execute('''
            INSERT INTO answers (answer_id, chat_id, question, answer) 
            VALUES (?, ?, ?, ?)
        ''', (answer_id, chat_id, question, answer))
        await db.commit()
        logger.info(
            f'saved answer: answer_id="{answer_id}" chat_id="{chat_id}"'
            f'question="{question}" answer="{answer}"'
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


async def save_chat(chat_id: str, username, chat_type) -> None:
    async with aiosqlite.connect(QNA_DB_PATH) as db:
        await db.execute('''
            INSERT INTO chats (chat_id, username, type) 
            VALUES (?, ?, ?)
        ''', (chat_id, username, chat_type))
        await db.commit()
        logger.info(
            f'saved chat: chat_id="{chat_id}" username="{username} type="{chat_type}"'
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
                )
            else:
                return None


async def set_profile(chat_id: str, profile: str) -> None:
    async with aiosqlite.connect(QNA_DB_PATH) as db:
        await db.execute('''
            UPDATE chats 
            SET profile = ? 
            WHERE chat_id = ?
        ''', (profile, chat_id))
        await db.commit()
        logger.info(f'set profile: chat_id="{chat_id}" profile="{profile}"')

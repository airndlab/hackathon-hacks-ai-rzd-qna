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
import aiosqlite
import logging
import os

logger = logging.getLogger(__name__)

QNA_CHATS_DB_PATH = os.getenv('QNA_CHATS_DB_PATH', 'chats.db')


# Инициализация базы данных
async def init_db() -> None:
    async with aiosqlite.connect(QNA_CHATS_DB_PATH) as db:
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
        await db.commit()


# Сохранить ответ
async def save_answer(answer_id: str, chat_id: str, question: str, answer: str) -> None:
    async with aiosqlite.connect(QNA_CHATS_DB_PATH) as db:
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
    async with aiosqlite.connect(QNA_CHATS_DB_PATH) as db:
        await db.execute('''
            UPDATE answers 
            SET feedback = ? 
            WHERE answer_id = ?
        ''', (feedback, answer_id))
        await db.commit()
        logger.info(f'set feedback: answer_id="{answer_id}" feedback="{feedback}"')

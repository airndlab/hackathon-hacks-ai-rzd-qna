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

import os

import aiohttp
from pydantic import BaseModel

# URL сервиса вопросов и ответов
QNA_URL = os.getenv('QNA_URL', 'http://qna:8080')


# Модель ответа
class Answer(BaseModel):
    id: str
    answer: str


# Получение ответа от пайплайна
async def get_answer(question: str, chat_id: str) -> Answer:
    async with aiohttp.ClientSession() as session:
        request = {'question': question, 'chat_id': chat_id}
        async with session.post(f'{QNA_URL}/api/answers', json=request) as response:
            if response.status == 200:
                json = await response.json()
                return Answer(**json)  # Возвращаем объект Answer
            else:
                raise Exception(
                    f"Ошибка получения ответа: {response.status} {await response.text()}")


# Лайкнуть ответ
async def like_answer(answer_id: str) -> None:
    async with aiohttp.ClientSession() as session:
        async with session.post(f'{QNA_URL}/api/answers/{answer_id}/liking') as response:
            if response.status != 200:
                raise Exception(
                    f"Ошибка положительной оценки ответа '{answer_id}': {response.status} {await response.text()}")


# Дизлайкнуть ответ
async def dislike_answer(answer_id: str) -> None:
    async with aiohttp.ClientSession() as session:
        async with session.post(f'{QNA_URL}/api/answers/{answer_id}/disliking') as response:
            if response.status != 200:
                raise Exception(
                    f"Ошибка отрицательной оценки ответа '{answer_id}': {response.status} {await response.text()}")

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

import asyncio
import logging
import os
import sys
import yaml
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery

from qna import get_answer, Answer, like_answer, dislike_answer

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Загрузка сообщений бота из файла
BOT_MESSAGES_FILE_PATH = os.getenv('BOT_MESSAGES_FILE_PATH')
with open(BOT_MESSAGES_FILE_PATH, 'r', encoding='utf-8') as file:
    bot_messages = yaml.safe_load(file)

# Инициализация диспетчера
dp = Dispatcher()


# Команда /start для приветствия
@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    await message.reply(bot_messages['start'])


# Обработчик лайка
@dp.callback_query(lambda query: query.data.startswith('like:'))
async def like_handler(query: CallbackQuery):
    await like_answer(query.data.split(':')[1])
    await query.answer(bot_messages['like'])


# Обработчик дизлайка
@dp.callback_query(lambda query: query.data.startswith('dislike:'))
async def dislike_handler(query: CallbackQuery):
    await dislike_answer(query.data.split(':')[1])
    await query.answer(bot_messages['dislike'])


# Обработчик вопросов
@dp.message()
async def question_handler(message: Message) -> None:
    try:
        question = message.text
        answer_data = await get_answer(question, f'{message.chat.id}')
        text = create_answer_text(answer_data, verbose)
        markup = create_answer_markup(answer_data.id)
        await message.reply(text, reply_markup=markup)
    except Exception as exception:
        error_text = bot_messages['error']
        await message.reply(error_text)


# Создание кнопок для сообщения

def create_answer_markup(answer_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="👍", callback_data=f'like:{answer_id}'),
        InlineKeyboardButton(text="👎", callback_data=f'dislike:{answer_id}')
    ]])


# Вспомогательные методы

def create_answer_text(response: Answer, verbose: bool) -> str:
    docs_text = get_docs_text(response.source)
    other_text = response.get_other_inline() if verbose else ''
    return bot_messages['answer'].format(
        answer=response.answer,
        class_1=response.class_1,
        class_2=response.class_2,
        assurance=assurance_text,
        docs=docs_text,
        other=other_text
    )


async def main() -> None:
    bot_token = os.getenv('BOT_TOKEN')
    bot = Bot(token=bot_token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

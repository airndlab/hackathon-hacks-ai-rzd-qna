import logging
import os

import aiohttp

logger = logging.getLogger(__name__)

BOT_API_URL = os.getenv('BOT_API_URL', 'http://bot:8088')


async def send_message(chat_id, message):
    payload = {"chatId": chat_id, "message": message}
    async with aiohttp.ClientSession() as session:
        async with session.post(f'{BOT_API_URL}/send-message', json=payload) as response:
            if response.status == 200:
                logger.info(f'Sent bot message: chat_id="{chat_id}" message="{message}"')
            else:
                logger.info(f'Error bot message: chat_id="{chat_id}" message="{message}"')

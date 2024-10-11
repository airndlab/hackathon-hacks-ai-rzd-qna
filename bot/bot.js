// Импортируем необходимые модули
const { Telegraf } = require('telegraf');
const axios = require('axios');
const config = require('./config');

// Вставьте сюда ваш токен
const bot = new Telegraf(config.telegramToken);

// Обработчик для получения и отправки вопроса
bot.on('text', async (ctx) => {
  const question = ctx.message.text; // Текст сообщения пользователя
  const chat_id = ctx.chat.id.toString(); // ID чата, преобразуем в строку для API

  try {
    // Отправляем POST-запрос с вопросом и chat_id
    const { data } = await axios.post(`${config.apiUrl}/api/answers`, {
      question,
      chat_id,
    });

    // Получаем ответ от API
    const answer = data.answer;

    // Отправляем ответ в чат
    ctx.reply(answer);

  } catch (error) {
    console.error('Ошибка при отправке запроса:', error);
    ctx.reply('Произошла ошибка при получении ответа.');
  }
});

// Запуск бота
bot.launch();

// Обработка завершения работы
process.once('SIGINT', () => bot.stop('SIGINT'));
process.once('SIGTERM', () => bot.stop('SIGTERM'));

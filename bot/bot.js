// Импортируем необходимые модули
const { Telegraf, Markup } = require('telegraf');
const fs = require('fs'); // Модуль для работы с файловой системой
const yaml = require('js-yaml'); // Библиотека для работы с YAML
const { createLogger, format, transports } = require('winston');
const { botToken, messagesPath } = require('./config');
const { getAnswer, likeAnswer, dislikeAnswer } = require('./qna');

// Вставьте сюда ваш токен
const bot = new Telegraf(botToken);

// Настройка логирования
const logger = createLogger({
  level: 'info',
  format: format.combine(
      format.timestamp(),
      format.printf(({ timestamp, level, message }) => `${timestamp} - ${level}: ${message}`)
  ),
  transports: [new transports.Console()]
});

// Загружаем сообщения
const botMessages = loadMessages();

// Обработчик команды /start
bot.start((ctx) => ctx.reply(botMessages.start));

// Обработчик для лайка
bot.action(/^like:(.+)$/, async (ctx) => {
  const answerId = ctx.match[1];
  await likeAnswer(answerId);
  await ctx.answerCbQuery(botMessages.like);
});

// Обработчик для дизлайка
bot.action(/^dislike:(.+)$/, async (ctx) => {
  const answerId = ctx.match[1];
  await dislikeAnswer(answerId);
  await ctx.answerCbQuery(botMessages.dislike);
});

// Обработчик текстовых сообщений (вопросов)
bot.on('text', async (ctx) => {
  const question = ctx.message.text;
  try {
    const answerData = await getAnswer(question, ctx.chat.id.toString());
    const text = createAnswerText(answerData);
    const markup = createAnswerMarkup(answerData.id);
    await ctx.reply(text, markup);
  } catch (error) {
    ctx.reply(botMessages.error);
    logger.error('Ошибка при обработке вопроса:', error);
  }
});

// Вспомогательные функции
function createAnswerMarkup(answerId) {
  return Markup.inlineKeyboard([
    Markup.button.callback('👍', `like:${answerId}`),
    Markup.button.callback('👎', `dislike:${answerId}`)
  ]);
}

function createAnswerText(response) {
  return botMessages.answer.replace('{answer}', response.answer);
}


// Функция для загрузки сообщений из YAML файла
function loadMessages() {
  try {
    const fileContents = fs.readFileSync(messagesPath, 'utf8'); // Читаем содержимое файла
    const messages = yaml.load(fileContents); // Загружаем YAML в объект
    return messages; // Возвращаем объект с сообщениями
  } catch (e) {
    console.error('Ошибка при загрузке сообщений:', e);
    return {};
  }
}

// Запуск бота
bot.launch()
   .then(() => logger.info('Бот запущен'))
   .catch((error) => logger.error('Ошибка при запуске бота:', error));

// Обработка завершения работы
process.once('SIGINT', () => bot.stop('SIGINT'));
process.once('SIGTERM', () => bot.stop('SIGTERM'));

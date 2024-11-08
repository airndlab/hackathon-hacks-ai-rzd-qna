// Импортируем необходимые модули
const { Telegraf, Markup } = require('telegraf');
const { message } = require('telegraf/filters');
const fs = require('fs'); // Модуль для работы с файловой системой
const yaml = require('js-yaml'); // Библиотека для работы с YAML
const { createLogger, format, transports } = require('winston');
const { botToken, messagesPath, faqPath, quideUrl } = require('./config');
const { getAnswer, likeAnswer, dislikeAnswer, getProfiles, postChats, getInfo, getFaq, updateProfile } = require('./qna');

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
const botMessages = loadMessages(messagesPath);
const faqMessages = loadMessages(faqPath);

// Обработчик команды /start
bot.start(async (ctx) => {
  try {
    const profiles = await getProfiles();
    await postChats(ctx.chat.id.toString(), ctx.from.username);
    const markup = createProfilesMarkup(profiles);
    const { details_md } = await getInfo(ctx.chat.id.toString());
    ctx.reply(`${botMessages.start}\n\n*Сейчас выбран:\n\n*${details_md}`, { ...markup, parse_mode: 'Markdown' });
  } catch (error) {
    ctx.reply(botMessages.error);
    logger.error('Ошибка при старте бота:', error);
  }
});

bot.command('info', async (ctx) => {
  try {
    const { details_md } = await getInfo(ctx.chat.id.toString());

    const markup = createInfoMarkup();
    ctx.reply(details_md, { ...markup, parse_mode: 'Markdown' });
  } catch (error) {
    ctx.reply(botMessages.error);
    logger.error('Ошибка при получении информации:', error);
  }
});

bot.command('guide', (ctx) => {
  ctx.reply(quideUrl);
});

bot.command('help', (ctx) => {
  ctx.reply('Я постоянно учусь и могу не знать всю требуемую Вам информацию.\n' +
      'Но Вы всегда можете обратиться за помощью к сотрудникам отдела кадров Вашей организации.');
});

bot.command('faq', async (ctx) => {
  try {
    const faq = await getFaq();
    const markup = createFaqMarkup(faq);
    ctx.reply('Выберите секцию', { ...markup, parse_mode: 'Markdown' });
  } catch (error) {
    if (faqMessages.faq) {
      const markup = createFaqMarkup(faqMessages.faq);
      ctx.reply('Выберите секцию', { ...markup, parse_mode: 'Markdown' });
    } else {
      ctx.reply(botMessages.error);
    }
    logger.error('Ошибка при получении часто задаваемых вопросов:', error);
  }
});

// Обработчик текстовых сообщений (вопросов)
bot.on(message('text'), async (ctx) => {
  const question = ctx.message.text;
  try {
    const answerData = await getAnswer(question, ctx.chat.id.toString());
    const text = createAnswerText(answerData);
    const markup = createAnswerMarkup(answerData.id);
    await ctx.reply(text, { ...markup, parse_mode: 'Markdown' });
  } catch (error) {
    ctx.reply(botMessages.error);
    logger.error('Ошибка при обработке вопроса:', error);
  }
});

//Обработчики колбэков
bot.action(/^profile_btn:(.+)$/, async (ctx) => {
  const callbackData = ctx.callbackQuery.data;
  const [, profileId] = callbackData.split(':'); // Разделяем данные

  try {
    await updateProfile(ctx.chat.id.toString(), profileId);
    const { details_md } = await getInfo(ctx.chat.id.toString());

    await ctx.answerCbQuery();
    ctx.reply(
        `*Отлично, вы выбрали следующую персону:*\n\n${details_md}`,
        { parse_mode: 'Markdown' },
    );
  } catch (error) {
    ctx.reply(botMessages.error);
    logger.error('Ошибка при получении информации:', error);
  }
});

bot.action('change_profile', async (ctx) => {
  try {
    const profiles = await getProfiles();
    const markup = createProfilesMarkup(profiles);
    await ctx.answerCbQuery();
    ctx.reply('Выберите профиль', { ...markup, parse_mode: 'Markdown' });
  } catch (error) {
    ctx.reply(botMessages.error);
    logger.error('Ошибка при обработке событий:', error);
  }
});

bot.action(/^section:(.+)$/, async (ctx) => {
  const callbackData = ctx.callbackQuery.data;
  const [, sectionIdx] = callbackData.split(':'); // Разделяем данные

  try {
    const faq = await getFaq();
    const { section, questions } = faq[sectionIdx];
    await ctx.answerCbQuery();
    ctx.reply(
        parseToMarkdownQuestions(section, questions),
        { parse_mode: 'Markdown' },
    );
  } catch (error) {
    const { section, questions } = faqMessages.faq[sectionIdx];
    await ctx.answerCbQuery();
    ctx.reply(
        parseToMarkdownQuestions(section, questions),
        { parse_mode: 'Markdown' },
    );
  }
});

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

function createProfilesMarkup(profiles) {
  return Markup.inlineKeyboard(profiles.map(({ id, title }) => [
    Markup.button.callback(title, `profile_btn:${id}`)
  ]));
}

function createFaqMarkup(sections) {
  return Markup.inlineKeyboard(sections.map(({ section }, idx) => [
    Markup.button.callback(section, `section:${idx}`)
  ]));
}

function createInfoMarkup() {
  return Markup.inlineKeyboard([
    Markup.button.callback('Сменить профиль', 'change_profile'),
  ]);
}

// Функция для преобразования в Markdown
function parseToMarkdownQuestions(section, questions) {
  let markdownText = '';
  markdownText += `*${section}*\n\n`;

  questions.forEach(q => {
      markdownText += `${q.question}\n`;
      markdownText += `${q.answer}\n\n`;
  });

  return markdownText;
}

// Функция для загрузки сообщений из YAML файла
function loadMessages(path) {
  try {
    const fileContents = fs.readFileSync(path, 'utf8'); // Читаем содержимое файла
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

// Включение API
require('./api')(bot, logger);

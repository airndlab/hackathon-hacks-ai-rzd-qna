require('dotenv').config();

const config = {
  botToken: process.env.BOT_TOKEN,
  qnaUrl: process.env.QNA_URL,
  messagesPath: process.env.BOT_MESSAGES_FILE_PATH,
  faqPath: process.env.FAQ_FILE_PATH,
  quideUrl: process.env.QUIDE_URL,
};

module.exports = config;

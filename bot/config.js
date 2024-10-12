require('dotenv').config();

const config = {
  botToken: process.env.BOT_TOKEN,
  qnaUrl: process.env.QNA_URL,
  messagesPath: process.env.BOT_MESSAGES_FILE_PATH,
};

module.exports = config;

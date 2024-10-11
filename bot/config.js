require('dotenv').config();

const config = {
  telegramToken: process.env.BOT_TOKEN,
  apiUrl: process.env.QNA_URL
};

module.exports = config;

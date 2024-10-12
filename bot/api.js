const express = require('express');
const bodyParser = require('body-parser');

// Функция для настройки API
module.exports = (bot, logger) => {
  const app = express();
  app.use(bodyParser.json());

  // Новый endpoint для отправки сообщений
  app.post('/send-message', async (req, res) => {
    const { chatId, message } = req.body;
    if (!chatId || !message) {
      return res.status(400).json({ error: 'chatId и message являются обязательными полями.' });
    }

    try {
      await bot.telegram.sendMessage(chatId, message);
      res.status(200).json({ success: true, message: 'Сообщение успешно отправлено.' });
    } catch (error) {
      logger.error('Ошибка при отправке сообщения:', error);
      res.status(500).json({ error: 'Ошибка при отправке сообщения.' });
    }
  });

  // Запуск express-сервера
  const PORT = process.env.BOT_API_PORT || 8088;
  app.listen(PORT, () => {
    logger.info(`Сервер API запущен на порту ${PORT}`);
  });
};

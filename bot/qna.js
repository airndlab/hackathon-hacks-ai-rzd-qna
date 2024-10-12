const axios = require('axios');
const { qnaUrl } = require('./config');

// URL сервиса вопросов и ответов
const QNA_URL = qnaUrl || 'http://qna:8080';

// Модель ответа
class Answer {
  constructor(id, answer) {
    this.id = id;
    this.answer = answer;
  }
}

// Получение ответа от пайплайна
async function getAnswer(question, chatId) {
  try {
    const request = { question, chatId: chatId };
    const response = await axios.post(`${QNA_URL}/api/answers`, request);

    if (response.status === 200) {
      const { id, answer } = response.data;
      return new Answer(id, answer);
    } else {
      throw new Error(`Ошибка получения ответа: ${response.status} ${response.statusText}`);
    }
  } catch (error) {
    throw new Error(`Ошибка получения ответа: ${error.message}`);
  }
}

// Лайкнуть ответ
async function likeAnswer(answerId) {
  try {
    const response = await axios.post(`${QNA_URL}/api/answers/${answerId}/liking`);

    if (response.status !== 200) {
      throw new Error(`Ошибка положительной оценки ответа '${answerId}': ${response.status} ${response.statusText}`);
    }
  } catch (error) {
    throw new Error(`Ошибка положительной оценки ответа '${answerId}': ${error.message}`);
  }
}

// Дизлайкнуть ответ
async function dislikeAnswer(answerId) {
  try {
    const response = await axios.post(`${QNA_URL}/api/answers/${answerId}/disliking`);

    if (response.status !== 200) {
      throw new Error(`Ошибка отрицательной оценки ответа '${answerId}': ${response.status} ${response.statusText}`);
    }
  } catch (error) {
    throw new Error(`Ошибка отрицательной оценки ответа '${answerId}': ${error.message}`);
  }
}

// Экспортируем функции
module.exports = {
  getAnswer,
  likeAnswer,
  dislikeAnswer
};

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
    const request = { question, chatId };
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

async function getProfiles() {
  try {
    const response = await axios.get(`${QNA_URL}/api/profiles`);

    if (response.status === 200) {
      return response.data;
    } else {
      throw new Error(`Ошибка получения профилей: ${response.status} ${response.statusText}`);
    }
  } catch (error) {
    throw new Error(`Ошибка получения профилей: ${error.message}`);
  }
}

async function postChats(id, username) {
  try {
    const request = { id, username, type: 'telegram' };
    const response = await axios.post(`${QNA_URL}/api/chats`, request);

    if (response.status !== 200) {
      throw new Error(`Ошибка отправки чата: ${response.status} ${response.statusText}`);
    }
  } catch (error) {
    throw new Error(`Ошибка отправки чата: ${error.message}`);
  }
}

async function getInfo(chatId) {
  try {
    const response = await axios.get(`${QNA_URL}/api/chats/${chatId}`);

    if (response.status === 200) {
      return response.data;
    } else {
      throw new Error(`Ошибка получения информации: ${response.status} ${response.statusText}`);
    }
  } catch (error) {
    throw new Error(`Ошибка получения информации: ${error.message}`);
  }
}

async function getFaq() {
  try {
    const response = await axios.get(`${QNA_URL}/api/faq`);

    if (response.status === 200) {
      return response.data;
    } else {
      throw new Error(`Ошибка получения часто задаваемых вопросов: ${response.status} ${response.statusText}`);
    }
  } catch (error) {
    throw new Error(`Ошибка получения часто задаваемых вопросов: ${error.message}`);
  }
}

async function updateProfile(chatId, profileId) {
  try {
    const response = await axios.patch(`${QNA_URL}/api/chats/${chatId}/profiles/${profileId}`);

    if (response.status === 200) {
      return response.data;
    } else {
      throw new Error(`Ошибка обновления профиля: ${response.status} ${response.statusText}`);
    }
  } catch (error) {
    throw new Error(`Ошибка обновления профиля: ${error.message}`);
  }
}

// Экспортируем функции
module.exports = {
  getAnswer,
  likeAnswer,
  dislikeAnswer,
  getProfiles,
  postChats,
  getInfo,
  getFaq,
  updateProfile,
};

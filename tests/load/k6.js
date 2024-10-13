import http from 'k6/http';
import { check, sleep } from 'k6';
import { SharedArray } from 'k6/data';
import { htmlReport } from "https://raw.githubusercontent.com/benc-uk/k6-reporter/main/dist/bundle.js";

const folderPath = './faq_questions/'; // Папка с файлами вопросов

// Массив имен файлов в папке
const fileNames = new SharedArray('file_names', function () {
  return [
    '1_Бесплатный_проезд.txt',
    '2_Отпуск.txt',
    '3_Здоровье.txt'
  ];
});

// Массив вопросов, загружаемый из случайного файла
const questions = new SharedArray('questions', function () {
  // Выбираем случайное имя файла из массива
  const randomFile = fileNames[Math.floor(Math.random() * fileNames.length)];

  // Используем open для чтения содержимого файла
  return open(`${folderPath}${randomFile}`).split('\n').filter(q => q.trim() !== '');
});

const server_ip = '176.123.163.193'

export function handleSummary(data) {
  return {
    "summary.html": htmlReport(data),
  };
}

export let options = {
  stages: [
    { duration: '1m', target: 3 }, // Ramp-up to 10 RPS over 1 minute
    { duration: '1m', target: 10 }, // Stay at 10 RPS for 5 minutes
    { duration: '1m', target: 0 },  // Ramp-down to 0 RPS
  ],
};

export default function () {
  const url = 'http://' + server_ip + ':8080/api/answers'; // Replace with your actual endpoint
  // Pick a random question from the list
  const question = questions[Math.floor(Math.random() * questions.length)];

  let chatId='1';
  const payload = JSON.stringify({
    question: question,
    chatId: chatId,
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
    },
  };

  const res = http.post(url, payload, params);

  check(res, {
    'status was 200': (r) => r.status === 200,
    'response contains answer': (r) => JSON.parse(r.body).answer !== ''
  });

  sleep(1); // Sleep for 1 second to simulate pacing for 1 RPS per virtual user
}

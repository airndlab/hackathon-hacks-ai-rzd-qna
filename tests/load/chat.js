import http from 'k6/http';
import { check, sleep } from 'k6';
import { randomString } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

let currentId = 5015; // Начальное значение ID
export let options = {
    stages: [
        { duration: '40s', target: 2 }, // Ramp-up to 10 RPS over 1 minute
        // { duration: '1m', target: 10 }, // Stay at 10 RPS for 5 minutes
        // { duration: '1m', target: 0 },  // Ramp-down to 0 RPS
    ],
};

export default function () {
    const url = 'http://176.123.163.193:8080/api/chats';

    // Генерация последовательного ID
    const id = getNextId(); // Получаем следующий ID
    const username = randomString(8); // Генерация случайного username

    const payload = JSON.stringify({
        id: String(id), // Преобразуем ID в строку
        username: username,
        type: 'telegram'
    });

    const params = {
        headers: {
            'Content-Type': 'application/json',
        },
    };

    // Отправка POST-запроса
    const res = http.post(url, payload, params);

    // Проверка успешного выполнения запроса (например, статус 200 или 201)
    check(res, {
        'status is 200 or 201': (r) => r.status === 200 || r.status === 201,
    });

    // Логируем ответ в консоль
    console.log(`ID: ${id}, Username: ${username}, Response: ${res.body}`);

    // Подготавливаем данные для записи в JSON
    const resultData = {
        id: id,
        username: username,
        response: res.body
    };

    sleep(1); // Задержка для имитации времени отклика
}

function getNextId() {
    if (currentId > 5100) {
        currentId = 5000; // Если достигли 5100, возвращаемся к 5000
    }
    return currentId++;
}
import http from 'k6/http';
import { check, sleep } from 'k6';
import { SharedArray } from 'k6/data';
import { randomIntBetween, randomItem } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

// Чтение JSON-файла с массивом профилей
const profiles = new SharedArray('profiles', function () {
    // Открываем файл и парсим его как JSON
    return JSON.parse(open('profiles/profiles.json'));
});

export let options = {
    stages: [
        { duration: '2m', target: 50 }, // Ramp-up to 10 RPS over 1 minute
        // { duration: '1m', target: 10 }, // Stay at 10 RPS for 5 minutes
        // { duration: '1m', target: 0 },  // Ramp-down to 0 RPS
    ],
};


export default function () {
    // Случайный chat_id от 5000 до 5100
    const chatId = randomIntBetween(5000, 5048);

    // Выбираем случайный профиль из массива
    const profile = randomItem(profiles);

    const url = `http://176.123.163.193:8080/api/chats/${chatId}/profiles`;

    const payload = JSON.stringify(profile);

    const params = {
        headers: {
            'Content-Type': 'application/json',
        },
    };

    // Отправка PUT-запроса
    const res = http.put(url, payload, params);

    // Проверка успешного выполнения запроса (например, статус 200)
    check(res, {
        'status is 200': (r) => r.status === 200,
    });

    // Логируем результат запроса
    console.log(`Sent profile to chat ID: ${chatId}, Response: ${res.status}`);

    // Задержка для имитации времени отклика
    sleep(1);
}
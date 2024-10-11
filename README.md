# РЖД: QnA чат-бот на основе базы знаний

[![Docker Image Version (latest semver)](https://img.shields.io/docker/v/airndlab/rzd-bot?label=bot)](https://hub.docker.com/r/airndlab/rzd-bot)

Состав проекта:

- [Telegram бот](bot)
- [Конфигурация](config/README.md)

## Разработка

Установить:

- python
- poetry

## Сборка

> Настроена сборка через
> [GitHub Actions](https://github.com/airndlab/hackathon-hacks-ai-rzd-qna/actions/workflows/docker.yml).

Установить:

- docker

Перейти в нужный подпроект и запустить:

```
docker build -t <название вашего образа>:<тег вашего образа> .
```

FROM python:3.10-alpine

RUN apt-get update && apt-get install -y gcc libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

LABEL org.opencontainers.image.vendor="AI RnD Lab"

CMD ["uvicorn", "app.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8080"]

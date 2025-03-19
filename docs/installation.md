# Установка и быстрый старт R2R Scrapy

## Системные требования

- Python 3.9 или выше
- pip (менеджер пакетов Python)
- Git (для установки из исходного кода)
- Docker (опционально, для установки через Docker)

## Установка из PyPI

```bash
pip install r2r-scrapy
```

## Установка из исходного кода

```bash
git clone https://github.com/eagurin/r2r-scrapy.git
cd r2r-scrapy
pip install -e .
```

## Установка с помощью Docker

```bash
docker pull eagurin/r2r-scrapy
```

или сборка из исходного кода:

```bash
git clone https://github.com/eagurin/r2r-scrapy.git
cd r2r-scrapy
docker build -t r2r-scrapy .
```

## Настройка R2R API

1. Получите API ключ от R2R сервиса
2. Создайте файл конфигурации `config.yaml`:

```yaml
r2r:
  api_url: "https://your-r2r-api.com"
  api_key: "your-api-key"
  batch_size: 10
  max_concurrency: 5

scrapy:
  concurrent_requests: 16
  concurrent_requests_per_domain: 8
  download_delay: 0.5
  user_agent: "R2R Scrapy/1.0"

processing:
  default_chunking_strategy: "semantic"
  chunk_size: 800
  chunk_overlap: 150
  preserve_code_blocks: true
  extract_metadata: true

monitoring:
  enabled: true
  prometheus_port: 9090
  quality_threshold: 0.8
  alert_on_error: true
```

Или установите переменные окружения:

```bash
export R2R_API_KEY="your-api-key"
export R2R_API_URL="https://your-r2r-api.com"
```

## Первый запуск

1. Создайте коллекцию:

```bash
r2r-scrapy create-collection --name "python-docs" --description "Python Documentation"
```

2. Запустите сбор документации:

```bash
r2r-scrapy scrape \
  --library "python" \
  --url "https://docs.python.org/3/" \
  --type api \
  --chunking semantic \
  --chunk-size 800 \
  --monitor
```

3. Проверьте результаты:

```bash
r2r-scrapy list-documents --collection "python-docs"
```

4. Сгенерируйте отчет:

```bash
r2r-scrapy generate-report --format html --output report.html
```

## Примеры использования

### Сбор API документации

```bash
r2r-scrapy scrape \
  --library "fastapi" \
  --url "https://fastapi.tiangolo.com/" \
  --type api \
  --chunking code_aware
```

### Сбор туториалов

```bash
r2r-scrapy scrape \
  --library "react" \
  --url "https://react.dev/learn" \
  --type tutorial \
  --chunking markdown_header
```

### Сбор GitHub репозитория

```bash
r2r-scrapy scrape \
  --library "langchain" \
  --url "https://github.com/langchain-ai/langchain" \
  --type github \
  --branch main \
  --include-readme \
  --include-docs
```

### Сбор технического блога

```bash
r2r-scrapy scrape \
  --library "pytorch-blog" \
  --url "https://pytorch.org/blog/" \
  --type blog \
  --chunking semantic
```

## Управление коллекциями

Список коллекций:
```bash
r2r-scrapy list-collections
```

Создание коллекции:
```bash
r2r-scrapy create-collection \
  --name "my-collection" \
  --description "My Documentation Collection" \
  --metadata '{"version": "1.0.0"}'
```

Удаление коллекции:
```bash
r2r-scrapy delete-collection --id "my-collection" --force
```

## Управление документами

Список документов:
```bash
r2r-scrapy list-documents --collection "my-collection"
```

Получение документа:
```bash
r2r-scrapy get-document --id "doc-id" --format json
```

Удаление документа:
```bash
r2r-scrapy delete-document --id "doc-id" --force
```

## Мониторинг

Генерация отчета:
```bash
r2r-scrapy generate-report \
  --format html \
  --output report.html \
  --collection "my-collection"
```

## Дополнительная информация

- [Полная документация](https://r2r-scrapy.readthedocs.io/)
- [GitHub репозиторий](https://github.com/eagurin/r2r-scrapy)
- [Примеры конфигурации](https://github.com/eagurin/r2r-scrapy/tree/main/examples)
- [Список изменений](https://github.com/eagurin/r2r-scrapy/blob/main/CHANGELOG.md) 
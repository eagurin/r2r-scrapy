# Конфигурация R2R Scrapy

## Конфигурационный файл YAML

R2R Scrapy использует YAML файл для конфигурации. По умолчанию ищет файл `config.yaml` в текущей директории.

### Базовая структура

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
  javascript_rendering: false
  splash_url: "http://localhost:8050"

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

### Секции конфигурации

#### R2R API (r2r)

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|-----------|
| api_url | string | - | URL R2R API |
| api_key | string | - | API ключ для доступа к R2R |
| batch_size | integer | 10 | Размер пакета для отправки документов |
| max_concurrency | integer | 5 | Максимальное количество одновременных запросов |

#### Scrapy (scrapy)

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|-----------|
| concurrent_requests | integer | 16 | Максимальное количество одновременных запросов |
| concurrent_requests_per_domain | integer | 8 | Максимальное количество запросов на домен |
| download_delay | float | 0.5 | Задержка между запросами (в секундах) |
| user_agent | string | "R2R Scrapy/1.0" | User-Agent для запросов |
| javascript_rendering | boolean | false | Включить рендеринг JavaScript |
| splash_url | string | "http://localhost:8050" | URL Splash сервера |

#### Обработка (processing)

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|-----------|
| default_chunking_strategy | string | "semantic" | Стратегия разбиения на чанки |
| chunk_size | integer | 800 | Размер чанка в символах |
| chunk_overlap | integer | 150 | Перекрытие между чанками |
| preserve_code_blocks | boolean | true | Сохранять блоки кода целиком |
| extract_metadata | boolean | true | Извлекать метаданные |

#### Мониторинг (monitoring)

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|-----------|
| enabled | boolean | true | Включить мониторинг |
| prometheus_port | integer | 9090 | Порт Prometheus |
| quality_threshold | float | 0.8 | Порог качества документов |
| alert_on_error | boolean | true | Оповещать об ошибках |

## Переменные окружения

Все настройки можно переопределить через переменные окружения. Используйте префикс и подчеркивания:

```bash
# R2R API
export R2R_API_URL="https://your-r2r-api.com"
export R2R_API_KEY="your-api-key"
export R2R_BATCH_SIZE="20"
export R2R_MAX_CONCURRENCY="10"

# Scrapy
export SCRAPY_CONCURRENT_REQUESTS="32"
export SCRAPY_CONCURRENT_REQUESTS_PER_DOMAIN="16"
export SCRAPY_DOWNLOAD_DELAY="1.0"
export SCRAPY_USER_AGENT="Custom User Agent"
export SCRAPY_JAVASCRIPT_RENDERING="true"
export SCRAPY_SPLASH_URL="http://splash:8050"

# Processing
export PROCESSING_DEFAULT_CHUNKING_STRATEGY="code_aware"
export PROCESSING_CHUNK_SIZE="1000"
export PROCESSING_CHUNK_OVERLAP="200"
export PROCESSING_PRESERVE_CODE_BLOCKS="true"
export PROCESSING_EXTRACT_METADATA="true"

# Monitoring
export MONITORING_ENABLED="true"
export MONITORING_PROMETHEUS_PORT="9091"
export MONITORING_QUALITY_THRESHOLD="0.9"
export MONITORING_ALERT_ON_ERROR="true"
```

## Аргументы командной строки

Многие настройки можно указать через аргументы командной строки. Они имеют наивысший приоритет:

```bash
r2r-scrapy scrape \
  --library "python" \
  --url "https://docs.python.org/3/" \
  --type api \
  --chunking semantic \
  --chunk-size 800 \
  --chunk-overlap 150 \
  --monitor \
  --config custom_config.yaml
```

## Приоритет настроек

Настройки применяются в следующем порядке (от низшего к высшему приоритету):

1. Значения по умолчанию
2. Конфигурационный файл YAML
3. Переменные окружения
4. Аргументы командной строки

## Примеры конфигурации

### Базовая конфигурация

```yaml
r2r:
  api_url: "https://your-r2r-api.com"
  api_key: "your-api-key"

scrapy:
  concurrent_requests: 16
  download_delay: 0.5

processing:
  default_chunking_strategy: "semantic"
  chunk_size: 800
```

### Конфигурация для JavaScript-сайтов

```yaml
r2r:
  api_url: "https://your-r2r-api.com"
  api_key: "your-api-key"

scrapy:
  javascript_rendering: true
  splash_url: "http://splash:8050"
  download_delay: 1.0

processing:
  default_chunking_strategy: "code_aware"
  preserve_code_blocks: true
```

### Конфигурация для высокой производительности

```yaml
r2r:
  batch_size: 50
  max_concurrency: 20

scrapy:
  concurrent_requests: 32
  concurrent_requests_per_domain: 16
  download_delay: 0.1

processing:
  chunk_size: 1000
  chunk_overlap: 100
```

### Конфигурация для качественного сбора

```yaml
r2r:
  batch_size: 5
  max_concurrency: 2

scrapy:
  concurrent_requests: 8
  download_delay: 2.0

processing:
  chunk_size: 500
  chunk_overlap: 200
  preserve_code_blocks: true
  extract_metadata: true

monitoring:
  quality_threshold: 0.9
  alert_on_error: true
``` 
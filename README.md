# R2R Scrapy

![R2R Scrapy Logo]()

> Асинхронный сборщик документации для RAG-систем

[![Stars]()]() [![Forks]()]() [![Issues]()]() [![License]()]() [![PyPI]()]() [![Python Versions]()]() [![CI]()]()

R2R Scrapy - это высокопроизводительный инструмент для асинхронного сбора, обработки и индексации технической документации в R2R. Проект оптимизирован для работы с документацией библиотек, фреймворков и API, обеспечивая интеллектуальную обработку и структурирование данных для максимальной эффективности RAG-систем.

## 📋 Содержание

- [R2R Scrapy](#r2r-scrapy)
  - [📋 Содержание](#-содержание)
  - [🚀 Установка](#-установка)
    - [Предварительные требования](#предварительные-требования)
    - [Установка из PyPI](#установка-из-pypi)
    - [Установка с дополнительными зависимостями](#установка-с-дополнительными-зависимостями)
    - [Установка из исходного кода](#установка-из-исходного-кода)
    - [Docker установка](#docker-установка)
  - [🚀 Быстрый старт](#-быстрый-старт)
  - [✨ Возможности](#-возможности)
  - [🏗️ Архитектура](#️-архитектура)
  - [⚙️ Конфигурация](#️-конфигурация)
    - [Конфигурационный файл](#конфигурационный-файл)
  - [📖 Использование](#-использование)
    - [Базовое использование](#базовое-использование)
      - [Скрапинг документации](#скрапинг-документации)
      - [Управление коллекциями](#управление-коллекциями)
    - [Продвинутое использование](#продвинутое-использование)
      - [Инкрементальные обновления](#инкрементальные-обновления)
      - [Распределенный скрапинг](#распределенный-скрапинг)
      - [Пользовательский пайплайн обработки](#пользовательский-пайплайн-обработки)
    - [Docker использование](#docker-использование)
  - [📚 API Reference](#-api-reference)
    - [Интерфейс командной строки](#интерфейс-командной-строки)
    - [Python API](#python-api)
  - [🤝 Участие в разработке](#-участие-в-разработке)
    - [Настройка окружения разработки](#настройка-окружения-разработки)
    - [Рекомендации по участию](#рекомендации-по-участию)
  - [📞 Контакты](#-контакты)

## 🚀 Установка

### Предварительные требования

- Python 3.9+
- R2R API access
- (Optional) Redis for distributed crawling
- (Optional) Docker for containerized deployment

### Установка из PyPI

```bash
pip install r2r-scrapy
```

### Установка с дополнительными зависимостями

```bash
# Установка с поддержкой JavaScript рендеринга
pip install "r2r-scrapy[js]"
```

```bash
# Установка с поддержкой распределенного скрапинга
pip install "r2r-scrapy[distributed]"
```

```bash
# Установка со всеми дополнительными зависимостями
pip install "r2r-scrapy[all]"
```

### Установка из исходного кода

```bash
# Клонировать репозиторий
git clone https://github.com/eagurin/r2r-scrapy.git
cd r2r-scrapy
```

```bash
# Создать виртуальное окружение
python -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate
```

```bash
# Установить зависимости
pip install -e .
```

### Docker установка

```bash
# Загрузить Docker образ
docker pull eagurin/r2r-scrapy:latest
```

```bash
# Или собрать из исходного кода
docker build -t r2r-scrapy .
```

## 🚀 Быстрый старт

1. Настройте учетные данные R2R API:

```bash
# Установите переменные окружения
export R2R_API_KEY=your_api_key
export R2R_API_URL=https://api.r2r.example.com
```

Или создайте конфигурационный файл:

```yaml
# config.yaml
r2r:
  api_key: your_api_key
  api_url: https://api.r2r.example.com
```

2. Запустите простую задачу скрапинга:

```bash
# Собрать документацию библиотеки
r2r-scrapy scrape --library fastapi --url https://fastapi.tiangolo.com/
```

3. Проверьте результаты:

```bash
# Список собранных документов
r2r-scrapy list-documents
```

```bash
# Сгенерировать отчет о качестве
r2r-scrapy generate-report
```

## ✨ Возможности

- **Асинхронный сбор данных**: Использует Scrapy для эффективного неблокирующего веб-скрапинга
- **Параллельная обработка документов**: Динамически управляет ресурсами для оптимальной производительности
- **Интеллектуальные стратегии разбиения**: Адаптирует чанкинг на основе типа контента
- **Специализированные пауки**: Настроены для различных источников документации
- **Мониторинг качества в реальном времени**: Обеспечивает высокое качество собираемых данных
- **Интеграция с R2R**: Оптимизировано для индексации в R2R
- **Инкрементальные обновления**: Поддерживает актуальность документации
- **Поддержка множества форматов**: Обрабатывает Markdown, HTML, Jupyter Notebooks, OpenAPI спецификации и другие
- **Обработка кода**: Специальная обработка блоков кода с определением языка
- **Обогащение метаданными**: Улучшает поиск с помощью расширенных метаданных
- **Контроль версий**: Отслеживает изменения в документации
- **Безопасность**: Безопасная обработка API ключей и конфиденциальных данных
- **Расширяемость**: Модульная архитектура для простой кастомизации

## 🏗️ Архитектура

R2R Scrapy построен на модульной архитектуре, обеспечивающей гибкость и расширяемость:

```yml
r2r_scrapy/
├── spiders/                  # Специализированные пауки для разных типов документации
│   ├── api_spider.py         # Паук для API документации
│   ├── tutorial_spider.py    # Паук для туториалов и руководств
│   ├── github_spider.py      # Паук для GitHub репозиториев
│   └── blog_spider.py        # Паук для технических блогов
├── processors/               # Процессоры контента
│   ├── code_processor.py     # Обработка блоков кода
│   ├── markdown_processor.py # Обработка Markdown
│   ├── api_processor.py      # Обработка API документации
│   └── html_processor.py     # Общая обработка HTML
├── chunkers/                 # Стратегии разбиения
│   ├── semantic_chunker.py   # Семантическое разбиение
│   ├── code_chunker.py       # Разбиение с учетом кода
│   ├── markdown_chunker.py   # Разбиение на основе Markdown
│   └── recursive_chunker.py  # Рекурсивное разбиение
├── exporters/                # Data exporters
│   ├── r2r_exporter.py       # R2R API exporter
│   └── file_exporter.py      # Local file exporter
├── middleware/               # Scrapy middleware
│   ├── javascript_middleware.py  # JavaScript rendering
│   └── rate_limiter.py       # Intelligent rate limiting
├── utils/                    # Utility functions
│   ├── url_prioritizer.py    # URL prioritization
│   ├── resource_manager.py   # Resource management
│   ├── quality_monitor.py    # Quality monitoring
│   └── version_control.py    # Version control
├── pipelines/                # Processing pipelines
│   ├── preprocessing_pipeline.py  # Preprocessing
│   ├── content_pipeline.py   # Content processing
│   ├── chunking_pipeline.py  # Chunking
│   └── r2r_pipeline.py       # R2R integration
├── integrations/             # External integrations
│   ├── github_integration.py # GitHub API integration
│   ├── stackoverflow_integration.py # Stack Overflow integration
│   └── wikipedia_integration.py # Wikipedia integration
├── security/                 # Security components
│   ├── key_manager.py        # API key management
│   └── secure_logger.py      # Secure logging
├── cli/                      # Command-line interface
│   ├── commands/             # CLI commands
│   └── main.py               # CLI entry point
├── settings.py               # Global settings
├── config.py                 # Configuration management
└── main.py                   # Main entry point
```

## ⚙️ Конфигурация

R2R Scrapy может быть настроен через YAML-файл конфигурации, переменные окружения или аргументы командной строки.

### Конфигурационный файл

```yaml
r2r:
  api_key: your_api_key
  api_url: https://api.r2r.example.com
  batch_size: 10
  max_concurrency: 5

scrapy:
  concurrent_requests: 16
  concurrent_requests_per_domain: 8
  download_delay: 0.5
  user_agent: "R2R Scrapy/1.0 (+https://github.com/eagurin/r2r-scrapy)"
  javascript_rendering: false
  splash_url: http://localhost:8050

processing:
  default_chunking_strategy: semantic
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

## 📖 Использование

### Базовое использование

#### Скрапинг документации

```bash
# Базовый скрапинг
r2r-scrapy scrape --library fastapi --url https://fastapi.tiangolo.com/
```

```bash
# Указание типа документации
r2r-scrapy scrape --library react --url https://reactjs.org/docs/ --type framework
```

```bash
# Настройка чанкинга
r2r-scrapy scrape --library pandas --url https://pandas.pydata.org/docs/ --chunking semantic
```

#### Управление коллекциями

```bash
# Список коллекций
r2r-scrapy list-collections
```

```bash
# Создание новой коллекции
r2r-scrapy create-collection --name fastapi-docs --description "Документация FastAPI"
```

```bash
# Удаление коллекции
r2r-scrapy delete-collection --id fastapi-docs
```

### Продвинутое использование

#### Инкрементальные обновления

```bash
# Инкрементальное обновление существующей коллекции
r2r-scrapy scrape --library tensorflow --url https://www.tensorflow.org/api_docs/ --incremental
```

#### Распределенный скрапинг

```bash
# Распределенный скрапинг с использованием Redis
r2r-scrapy scrape --library pytorch --url https://pytorch.org/docs/ --distributed --redis-url redis://localhost:6379
```

#### Пользовательский пайплайн обработки

```bash
# Использование пользовательского пайплайна обработки
r2r-scrapy scrape --library scikit-learn --url https://scikit-learn.org/stable/ --pipeline custom_pipeline.py
```

### Docker использование

```bash
# Запуск через Docker
docker run -v $(pwd)/config.yaml:/app/config.yaml \
  -v $(pwd)/data:/app/data \
  r2r-scrapy scrape --library numpy --url https://numpy.org/doc/
```

## 📚 API Reference

### Интерфейс командной строки

```bash
r2r-scrapy [OPTIONS] COMMAND [ARGS]...

Опции:
  --config FILE  Путь к конфигурационному файлу
  --verbose      Включить подробный вывод
  --help         Показать справку

Команды:
  scrape              Скрапинг документации
  list-collections    Список коллекций R2R
  create-collection   Создать новую коллекцию R2R
  delete-collection   Удалить коллекцию R2R
  list-documents      Список документов в коллекции
  get-document        Получить детали документа
  delete-document     Удалить документ
  generate-report     Сгенерировать отчет о качестве
  test-search         Тестировать качество поиска
  monitor             Запустить сервер мониторинга
  version             Показать информацию о версии
```

### Python API

```python
from r2r_scrapy import R2RScraper

# Инициализация скрапера
scraper = R2RScraper(
    r2r_api_key="your_api_key",
    r2r_api_url="https://api.r2r.example.com"
)

# Скрапинг документации библиотеки
result = scraper.scrape(
    library="fastapi",
    url="https://fastapi.tiangolo.com/",
    chunking_strategy="semantic",
    chunk_size=800,
    chunk_overlap=150
)

# Получение результатов
print(f"Собрано {result['documents_count']} документов")
print(f"Создано {result['chunks_count']} чанков")
print(f"Оценка качества: {result['quality_score']}")

# Получение деталей документа
document = scraper.get_document(document_id="document_id")
print(document)
```

## 🤝 Участие в разработке

Мы приветствуем ваше участие! Пожалуйста, не стесняйтесь создавать Pull Request.

### Настройка окружения разработки

```bash
# Клонировать репозиторий
git clone https://github.com/eagurin/r2r-scrapy.git
cd r2r-scrapy
```

```bash
# Создать виртуальное окружение
python -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate
```

```bash
# Установить зависимости для разработки
pip install -e ".[dev]"
```

```bash
# Запустить тесты
pytest
```

```bash
# Запустить линтеры
flake8
black .
```

### Рекомендации по участию

1. Форкните репозиторий
2. Создайте ветку для вашей функциональности (`git checkout -b feature/amazing-feature`)
3. Зафиксируйте изменения (`git commit -m 'Добавлена потрясающая функциональность'`)
4. Отправьте изменения в ваш форк (`git push origin feature/amazing-feature`)
5. Откройте Pull Request

Пожалуйста, убедитесь, что ваш код соответствует стилю проекта и проходит все тесты.


## 📞 Контакты

- Автор: Евгений Гурин
- Email: e.a.gurin@gmail.com
- GitHub: https://github.com/eagurin

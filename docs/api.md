# API Reference R2R Scrapy

## CLI API

### scrape

Сбор документации с веб-сайта.

```bash
r2r-scrapy scrape [OPTIONS]

Опции:
  --library TEXT          Название библиотеки [обязательно]
  --url TEXT             URL для сбора [обязательно]
  --type [api|tutorial|github|blog]  Тип документации [по умолчанию: api]
  --chunking [semantic|code_aware|markdown_header|recursive]
                        Стратегия разбиения [по умолчанию: semantic]
  --chunk-size INTEGER   Размер чанка [по умолчанию: 800]
  --chunk-overlap INTEGER  Перекрытие чанков [по умолчанию: 150]
  --incremental         Инкрементальное обновление
  --monitor             Включить мониторинг
  --config TEXT         Путь к файлу конфигурации
```

### list-collections

Список коллекций.

```bash
r2r-scrapy list-collections [OPTIONS]

Опции:
  --format [table|json]  Формат вывода [по умолчанию: table]
  --config TEXT         Путь к файлу конфигурации
```

### create-collection

Создание новой коллекции.

```bash
r2r-scrapy create-collection [OPTIONS]

Опции:
  --name TEXT           Название коллекции [обязательно]
  --description TEXT    Описание коллекции
  --metadata TEXT       Метаданные в формате JSON
  --config TEXT         Путь к файлу конфигурации
```

### delete-collection

Удаление коллекции.

```bash
r2r-scrapy delete-collection [OPTIONS]

Опции:
  --id TEXT            ID коллекции [обязательно]
  --force             Удалить без подтверждения
  --config TEXT       Путь к файлу конфигурации
```

### list-documents

Список документов.

```bash
r2r-scrapy list-documents [OPTIONS]

Опции:
  --collection TEXT    ID коллекции
  --limit INTEGER     Максимальное количество документов [по умолчанию: 100]
  --offset INTEGER    Смещение для пагинации [по умолчанию: 0]
  --format [table|json]  Формат вывода [по умолчанию: table]
  --config TEXT       Путь к файлу конфигурации
```

### get-document

Получение документа.

```bash
r2r-scrapy get-document [OPTIONS]

Опции:
  --id TEXT           ID документа [обязательно]
  --format [json|text|summary]  Формат вывода [по умолчанию: json]
  --output TEXT       Путь для сохранения
  --config TEXT       Путь к файлу конфигурации
```

### delete-document

Удаление документа.

```bash
r2r-scrapy delete-document [OPTIONS]

Опции:
  --id TEXT          ID документа [обязательно]
  --force           Удалить без подтверждения
  --config TEXT     Путь к файлу конфигурации
```

### generate-report

Генерация отчета.

```bash
r2r-scrapy generate-report [OPTIONS]

Опции:
  --format [html|json]  Формат отчета [по умолчанию: html]
  --output TEXT       Путь для сохранения
  --collection TEXT   ID коллекции для фильтрации
  --config TEXT       Путь к файлу конфигурации
```

## Python API

### Конфигурация

```python
from r2r_scrapy.config import Config

# Загрузка конфигурации
config = Config('config.yaml')

# Получение настроек
api_url = config.get('r2r.api_url')
batch_size = config.get('r2r.batch_size', 10)  # С значением по умолчанию

# Получение всех настроек
all_settings = config.get_all()
```

### Процессоры

```python
from r2r_scrapy.processors.code_processor import CodeProcessor
from r2r_scrapy.processors.markdown_processor import MarkdownProcessor
from r2r_scrapy.processors.html_processor import HTMLProcessor
from r2r_scrapy.processors.api_processor import APIDocProcessor

# Обработка кода
code_processor = CodeProcessor()
result = code_processor.process_code(code, language='python')

# Обработка Markdown
markdown_processor = MarkdownProcessor()
content, metadata = markdown_processor.process_markdown(markdown)

# Обработка HTML
html_processor = HTMLProcessor()
content, metadata = html_processor.process(response)

# Обработка API документации
api_processor = APIDocProcessor()
elements = api_processor.extract_api_elements(content)
```

### Чанкеры

```python
from r2r_scrapy.chunkers.semantic_chunker import SemanticChunker
from r2r_scrapy.chunkers.code_chunker import CodeChunker
from r2r_scrapy.chunkers.markdown_chunker import MarkdownChunker
from r2r_scrapy.chunkers.recursive_chunker import RecursiveChunker

# Семантический чанкинг
chunker = SemanticChunker(chunk_size=800, chunk_overlap=150)
chunks = chunker.chunk_text(text)

# Чанкинг с сохранением кода
code_chunker = CodeChunker(chunk_size=800, chunk_overlap=150)
chunks = code_chunker.chunk_text(text)

# Чанкинг по заголовкам Markdown
md_chunker = MarkdownChunker(chunk_size=800, chunk_overlap=150)
chunks = md_chunker.chunk_text(text)

# Рекурсивный чанкинг
recursive_chunker = RecursiveChunker(chunk_size=800, chunk_overlap=150)
chunks = recursive_chunker.chunk_text(text)
```

### Экспортеры

```python
from r2r_scrapy.exporters.r2r_exporter import R2RExporter
from r2r_scrapy.exporters.file_exporter import FileExporter

# Экспорт в R2R API
exporter = R2RExporter(
    api_url="https://your-r2r-api.com",
    api_key="your-api-key",
    batch_size=10,
    max_concurrency=5
)
await exporter.initialize()
result = await exporter.export_documents(documents, collection_id)
await exporter.close()

# Экспорт в файлы
file_exporter = FileExporter(output_dir='./output', format='json')
results = file_exporter.export_documents(documents, collection_id)
chunk_results = file_exporter.export_chunks(chunks, document_id)
```

### Пауки

```python
from r2r_scrapy.spiders.api_spider import APIDocSpider
from r2r_scrapy.spiders.tutorial_spider import TutorialSpider
from r2r_scrapy.spiders.github_spider import GitHubSpider
from r2r_scrapy.spiders.blog_spider import BlogSpider

# Паук для API документации
class CustomAPISpider(APIDocSpider):
    name = 'custom_api'
    allowed_domains = ['docs.example.com']
    start_urls = ['https://docs.example.com/api']

# Паук для туториалов
class CustomTutorialSpider(TutorialSpider):
    name = 'custom_tutorial'
    allowed_domains = ['tutorials.example.com']
    start_urls = ['https://tutorials.example.com/guide']

# Паук для GitHub
class CustomGitHubSpider(GitHubSpider):
    name = 'custom_github'
    owner = 'username'
    repo = 'repo'
    branch = 'main'

# Паук для блогов
class CustomBlogSpider(BlogSpider):
    name = 'custom_blog'
    allowed_domains = ['blog.example.com']
    start_urls = ['https://blog.example.com/tech']
```

### Интеграции

```python
from r2r_scrapy.integrations.github_integration import GitHubIntegration
from r2r_scrapy.integrations.stackoverflow_integration import StackOverflowIntegration
from r2r_scrapy.integrations.wikipedia_integration import WikipediaIntegration

# GitHub API
github = GitHubIntegration(token="your-github-token")
await github.initialize()
contents = await github.get_repository_contents("owner", "repo")
readme = await github.get_readme("owner", "repo")
await github.close()

# Stack Overflow API
stackoverflow = StackOverflowIntegration(api_key="your-so-key")
await stackoverflow.initialize()
questions = await stackoverflow.search_questions("python scrapy")
answers = await stackoverflow.get_question_answers(question_id)
await stackoverflow.close()

# Wikipedia API
wikipedia = WikipediaIntegration(language="ru")
await wikipedia.initialize()
results = await wikipedia.search("scrapy")
article = await wikipedia.get_article("Web scraping")
await wikipedia.close()
```

### Безопасность

```python
from r2r_scrapy.security.key_manager import KeyManager
from r2r_scrapy.security.secure_logger import SecureLogger

# Управление ключами
key_manager = KeyManager(storage_path="./keys.json")
key_manager.set_key("api_key", "secret-value")
value = key_manager.get_key("api_key")
key_manager.delete_key("api_key")

# Безопасное логирование
logger = SecureLogger(log_dir="./logs", mask_sensitive=True)
logger.info("Using API key: {key}", key="secret-value")  # Будет замаскировано
```

## Интеграция с другими системами

### Prometheus

R2R Scrapy экспортирует метрики в формате Prometheus:

```python
from r2r_scrapy.utils.quality_monitor import QualityMonitor

monitor = QualityMonitor(port=9090)
monitor.record_page_scraped(status="success", doc_type="api")
monitor.record_content_size(size=1024, doc_type="markdown")
monitor.record_chunk_stats(chunk_count=5, strategy="semantic")
```

### Docker

Запуск в Docker:

```bash
docker run -v ./config.yaml:/app/config.yaml \
  -e R2R_API_KEY="your-api-key" \
  eagurin/r2r-scrapy scrape \
  --library "python" \
  --url "https://docs.python.org/3/"
```

### CI/CD

Пример GitHub Actions:

```yaml
name: R2R Scrapy

on:
  schedule:
    - cron: '0 0 * * *'  # Ежедневно

jobs:
  scrape:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install r2r-scrapy
      - run: r2r-scrapy scrape --library "python" --url "https://docs.python.org/3/"
        env:
          R2R_API_KEY: ${{ secrets.R2R_API_KEY }}
``` 
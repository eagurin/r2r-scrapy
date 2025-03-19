# Justfile для проекта R2R Scrapy
# Для использования установите just: https://github.com/casey/just

# Загрузить переменные окружения из .env файла, если он существует
set dotenv-load := true

# Установить оболочку по умолчанию
set shell := ["bash", "-c"]

# Показать доступные команды
default:
    @just --list

# Установить проект в режиме разработки
install-dev:
    poetry install --with dev

# Установить проект с дополнительными зависимостями
install-extras extras="all":
    poetry install -E {{extras}}

# Установить pre-commit хуки
setup-pre-commit:
    poetry run pre-commit install

# Запустить линтеры
lint:
    poetry run ruff r2r_scrapy tests
    poetry run black --check r2r_scrapy tests
    poetry run isort --check r2r_scrapy tests
    poetry run mypy r2r_scrapy

# Исправить ошибки линтеров
lint-fix:
    poetry run ruff r2r_scrapy tests --fix
    poetry run black r2r_scrapy tests
    poetry run isort r2r_scrapy tests

# Запустить тесты
test:
    poetry run pytest

# Запустить тесты с покрытием
test-cov:
    poetry run pytest --cov=r2r_scrapy --cov-report=term --cov-report=html

# Запустить тесты с подробным выводом
test-verbose:
    poetry run pytest -v

# Собрать пакет
build:
    poetry build

# Опубликовать пакет в PyPI
publish:
    poetry publish

# Запустить сбор документации для указанной библиотеки
scrape library url:
    poetry run r2r-scrapy scrape --library {{library}} --url {{url}}

# Запустить сбор документации с дополнительными параметрами
scrape-advanced library url type="api" chunking="semantic" chunk-size="800" chunk-overlap="150":
    poetry run r2r-scrapy scrape --library {{library}} --url {{url}} --type {{type}} --chunking {{chunking}} --chunk-size {{chunk-size}} --chunk-overlap {{chunk-overlap}}

# Запустить сбор документации из GitHub репозитория
scrape-github owner repo branch="main":
    poetry run r2r-scrapy scrape --type github --library {{repo}} --url "https://github.com/{{owner}}/{{repo}}" --branch {{branch}}

# Вывести список коллекций
list-collections format="table":
    poetry run r2r-scrapy list-collections --format {{format}}

# Вывести список документов в коллекции
list-documents collection="":
    poetry run r2r-scrapy list-documents --collection {{collection}}

# Создать новую коллекцию
create-collection name description="":
    poetry run r2r-scrapy create-collection --name {{name}} --description {{description}}

# Удалить коллекцию
delete-collection id:
    poetry run r2r-scrapy delete-collection --id {{id}}

# Получить документ
get-document id format="json":
    poetry run r2r-scrapy get-document --id {{id}} --format {{format}}

# Удалить документ
delete-document id:
    poetry run r2r-scrapy delete-document --id {{id}}

# Сгенерировать отчет
generate-report format="html" output="report.html":
    poetry run r2r-scrapy generate-report --format {{format}} --output {{output}}

# Запустить Docker контейнер
docker-run:
    docker run -v $(pwd)/config.yaml:/app/config.yaml -v $(pwd)/data:/app/data r2r-scrapy:latest

# Собрать Docker образ
docker-build:
    docker build -t r2r-scrapy:latest .

# Запустить Splash для JavaScript рендеринга
run-splash:
    docker run -p 8050:8050 scrapinghub/splash

# Создать новый релиз
release version:
    #!/usr/bin/env bash
    set -euo pipefail
    # Обновить версию в pyproject.toml
    poetry version {{version}}
    # Создать коммит с новой версией
    git add pyproject.toml
    git commit -m "Bump version to {{version}}"
    # Создать тег
    git tag -a v{{version}} -m "Version {{version}}"
    # Отправить изменения и тег
    git push
    git push --tags
    # Собрать и опубликовать пакет
    just build
    just publish

# Создать базовый конфигурационный файл
create-config:
    @echo "Creating config.yaml file with default settings"
    @cat > config.yaml << 'EOF'
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

# Инициализировать проект (установка, настройка pre-commit, создание .env и config.yaml)
init:
    just install-dev
    just setup-pre-commit
    just create-env
    just create-config
    just create-dockerfile

# Очистить временные файлы и артефакты сборки
clean:
    rm -rf dist build .pytest_cache .coverage htmlcov .mypy_cache .ruff_cache
    find . -type d -name __pycache__ -exec rm -rf {} +
    find . -type d -name "*.egg-info" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete

# Запустить проверку безопасности
security-check:
    poetry run pip-audit
    poetry run bandit -r r2r_scrapy

# Создать виртуальное окружение и активировать его
venv:
    python -m venv .venv
    @echo "Virtual environment created. Activate it with:"
    @echo "source .venv/bin/activate  # Linux/macOS"
    @echo ".venv\\Scripts\\activate  # Windows"

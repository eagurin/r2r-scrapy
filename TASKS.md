Assistant
Полный список запросов для сборки проекта R2R Scrapy с Claude 3.7 Sonnet


___


Этап 1: Создание структуры проекта и базовых файлов


___

Запрос 1: Создание структуры проекта

Создай структуру каталогов для проекта R2R Scrapy. Проект должен иметь следующую структуру:
r2r_scrapy/
├── r2r_scrapy/
│   ├── __init__.py
│   ├── spiders/
│   │   └── __init__.py
│   ├── processors/
│   │   └── __init__.py
│   ├── chunkers/
│   │   └── __init__.py
│   ├── exporters/
│   │   └── __init__.py
│   ├── middleware/
│   │   └── __init__.py
│   ├── utils/
│   │   └── __init__.py
│   ├── pipelines/
│   │   └── __init__.py
│   ├── integrations/
│   │   └── __init__.py
│   ├── security/
│   │   └── __init__.py
│   ├── cli/
│   │   ├── __init__.py
│   │   └── commands/
│   │       └── __init__.py
│   ├── config.py
│   ├── settings.py
│   └── main.py
├── tests/
│   └── __init__.py
└── pyproject.toml

Для каждого __init__.py файла добавь базовый код с указанием версии и импортами основных классов модуля.

___

Запрос 2: Создание pyproject.toml

Создай полный pyproject.toml для проекта R2R Scrapy с использованием Poetry. Проект должен включать следующие зависимости:

- scrapy
- aiohttp
- asyncio
- pyyaml
- click
- tqdm
- langchain
- langchain-text-splitters
- beautifulsoup4
- markdown
- html2text
- nbformat
- openai
- tiktoken
- cryptography
- prometheus-client
- psutil
- gitpython
- scikit-learn
- numpy
- pygments
- playwright

Также добавь опциональные зависимости для:

- JavaScript рендеринга (scrapy-splash, playwright)
- Распределенного скрапинга (redis, scrapy-redis)
- Машинного обучения (spacy, nltk)
- OpenAPI (jsonschema)
- Перевода (deepl)

Добавь инструменты разработки:

- pytest, pytest-cov
- black, isort
- flake8, ruff, mypy
- sphinx, sphinx-rtd-theme
- pre-commit

Настрой точку входа CLI: r2r-scrapy = "r2r_scrapy.main:main"

___

Запрос 3: Создание README.md

Создай README.md для проекта R2R Scrapy. Документ должен включать:

1. Заголовок и краткое описание проекта
2. Бейджи (stars, forks, issues, license, PyPI, Python versions, CI)
3. Таблицу содержания
4. Разделы: Installation, Quick Start, Features, Architecture, Configuration, Usage, API Reference, Examples, Contributing, License, Dependencies, Changelog, Roadmap, FAQ, Contact

В разделе Architecture опиши структуру проекта, как она организована в каталоги и модули.

___


Этап 2: Базовые компоненты


___

Запрос 4: Создание config.py

Реализуй модуль r2r_scrapy/config.py для проекта R2R Scrapy.

Этот модуль должен:

1. Загружать конфигурацию из YAML файла
2. Поддерживать переопределение настроек через переменные окружения
3. Предоставлять удобный доступ к настройкам через метод get()

Используй следующие импорты:
import os
import yaml
from typing import Dict, Any, Optional

Основной класс должен называться Config с методами load_from_file(), load_from_env(), get() и get_all().

___

Запрос 5: Создание settings.py

Реализуй модуль r2r_scrapy/settings.py для проекта R2R Scrapy.

Этот модуль должен:

1. Загружать конфигурацию из Config
2. Настраивать параметры Scrapy
3. Настраивать параметры R2R API
4. Настраивать параметры обработки и мониторинга

Используй следующие импорты:
import os
from r2r_scrapy.config import Config

Модуль должен создавать экземпляр Config и использовать его для настройки различных параметров.

___

Запрос 6: Создание main.py

Реализуй модуль r2r_scrapy/main.py для проекта R2R Scrapy.

Этот модуль должен:

1. Служить точкой входа для CLI
2. Импортировать и запускать CLI из r2r_scrapy.cli.main
3. Добавлять путь проекта в sys.path

Используй следующие импорты:
import sys
import os
from r2r_scrapy.cli.main import main
import asyncio

Функция main() должна запускаться через asyncio.run().

___


Этап 3: Процессоры


___

Запрос 7: Создание code_processor.py

Реализуй модуль r2r_scrapy/processors/code_processor.py для проекта R2R Scrapy.

Этот модуль должен:

1. Обрабатывать блоки кода, извлеченные из документации
2. Определять язык программирования, если он не указан
3. Форматировать код для лучшей читаемости
4. Очищать код от лишних пробелов и отступов

Используй следующие импорты:
import re
from pygments import lexers, highlight
from pygments.formatters import HtmlFormatter

Основной класс должен называться CodeProcessor с методами process_code(), clean_code() и detect_language().

___

Запрос 8: Создание markdown_processor.py

Реализуй модуль r2r_scrapy/processors/markdown_processor.py для проекта R2R Scrapy.

Этот модуль должен:

1. Обрабатывать Markdown контент
2. Извлекать метаданные из Markdown (заголовки, ссылки)
3. Очищать Markdown от лишних элементов
4. Преобразовывать Markdown в чистый текст при необходимости

Используй следующие импорты:
import re
import markdown
from bs4 import BeautifulSoup
import html2text

Основной класс должен называться MarkdownProcessor с методами process_markdown(), extract_metadata() и clean_markdown().

___

Запрос 9: Создание html_processor.py

Реализуй модуль r2r_scrapy/processors/html_processor.py для проекта R2R Scrapy.

Этот модуль должен:

1. Обрабатывать HTML контент
2. Извлекать основное содержимое, игнорируя навигацию, сайдбары и т.д.
3. Извлекать метаданные из HTML (заголовок, описание, ключевые слова)
4. Преобразовывать HTML в Markdown или чистый текст

Используй следующие импорты:
import re
from bs4 import BeautifulSoup
import html2text
from r2r_scrapy.processors.code_processor import CodeProcessor

Основной класс должен называться HTMLProcessor с методами process(), extract_metadata() и clean_markdown().

___

Запрос 10: Создание api_processor.py

Реализуй модуль r2r_scrapy/processors/api_processor.py для проекта R2R Scrapy.

Этот модуль должен:

1. Обрабатывать API документацию
2. Определять структуру API документации
3. Извлекать API элементы (функции, методы, классы)
4. Обрабатывать примеры кода в API документации

Используй следующие импорты:
import re
from bs4 import BeautifulSoup
from r2r_scrapy.processors.code_processor import CodeProcessor

Основной класс должен называться APIDocProcessor с методами detect_structure(), extract_api_elements() и process().

___


Этап 4: Чанкеры


___

Запрос 11: Создание semantic_chunker.py

Реализуй модуль r2r_scrapy/chunkers/semantic_chunker.py для проекта R2R Scrapy.

Этот модуль должен:

1. Разбивать текст на семантически связные чанки
2. Использовать TF-IDF для определения семантической близости
3. Обеспечивать перекрытие между чанками
4. Находить оптимальные точки разделения текста

Используй следующие импорты:
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

Основной класс должен называться SemanticChunker с методами chunk_text(),_split_into_paragraphs(), _create_initial_chunks(),_refine_chunks() и_find_break_point().

___

Запрос 12: Создание code_chunker.py

Реализуй модуль r2r_scrapy/chunkers/code_chunker.py для проекта R2R Scrapy.

Этот модуль должен:

1. Разбивать текст на чанки, сохраняя целостность блоков кода
2. Извлекать и сохранять блоки кода при разбиении
3. Обеспечивать перекрытие между чанками
4. Находить оптимальные точки разделения текста

Используй следующие импорты:
import re

Основной класс должен называться CodeChunker с методами chunk_text(),_extract_code_blocks(), _simple_chunk() и _find_break_point().

___

Запрос 13: Создание markdown_chunker.py

Реализуй модуль r2r_scrapy/chunkers/markdown_chunker.py для проекта R2R Scrapy.

Этот модуль должен:

1. Разбивать Markdown на чанки по заголовкам и размеру
2. Сохранять структуру Markdown при разбиении
3. Обеспечивать перекрытие между чанками
4. Находить оптимальные точки разделения текста

Используй следующие импорты:
import re

Основной класс должен называться MarkdownChunker с методами chunk_text(),_split_by_headings(), _split_by_size() и_find_break_point().

___

Запрос 14: Создание recursive_chunker.py

Реализуй модуль r2r_scrapy/chunkers/recursive_chunker.py для проекта R2R Scrapy.

Этот модуль должен:

1. Рекурсивно разбивать текст на чанки
2. Использовать разные разделители на разных уровнях рекурсии
3. Обеспечивать перекрытие между чанками
4. Находить оптимальные точки разделения текста

Используй следующие импорты:
import re

Основной класс должен называться RecursiveChunker с методами chunk_text(),_recursive_chunk(),_simple_chunk() и_find_break_point().

___


Этап 5: Пауки (Spiders)


___

Запрос 15: Создание api_spider.py

Реализуй модуль r2r_scrapy/spiders/api_spider.py для проекта R2R Scrapy.

Этот модуль должен:

1. Реализовывать паук для сбора API документации
2. Следовать по ссылкам в API документации
3. Извлекать API элементы и их описания
4. Определять язык программирования

Используй следующие импорты:
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
import re
from r2r_scrapy.processors.api_processor import APIDocProcessor

Основной класс должен называться APIDocSpider, наследоваться от CrawlSpider и иметь методы parse_api_doc(), extract_version() и detect_programming_language().

___

Запрос 16: Создание tutorial_spider.py

Реализуй модуль r2r_scrapy/spiders/tutorial_spider.py для проекта R2R Scrapy.

Этот модуль должен:

1. Реализовывать паук для сбора туториалов и руководств
2. Следовать по ссылкам в документации
3. Определять тип контента (Markdown или HTML)
4. Извлекать структуру туториала и оценивать его сложность

Используй следующие импорты:
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
import re
from r2r_scrapy.processors.markdown_processor import MarkdownProcessor
from r2r_scrapy.processors.html_processor import HTMLProcessor

Основной класс должен называться TutorialSpider, наследоваться от CrawlSpider и иметь методы parse_tutorial(), detect_content_type(), extract_structure(), detect_tutorial_level() и calculate_reading_time().

___

Запрос 17: Создание github_spider.py

Реализуй модуль r2r_scrapy/spiders/github_spider.py для проекта R2R Scrapy.

Этот модуль должен:

1. Реализовывать паук для сбора документации из GitHub репозиториев
2. Использовать GitHub API для получения содержимого репозитория
3. Обрабатывать различные типы файлов (Markdown, код, конфигурации)
4. Извлекать метаданные из репозитория

Используй следующие импорты:
import scrapy
import json
import base64
import re
from urllib.parse import urljoin
from r2r_scrapy.processors.markdown_processor import MarkdownProcessor
from r2r_scrapy.processors.code_processor import CodeProcessor

Основной класс должен называться GitHubSpider, наследоваться от scrapy.Spider и иметь методы start_requests(), parse_repo_contents(), parse_readme(), parse_file(),_is_test_path(), _is_doc_file(),_get_file_type(), _detect_language_from_path() и _get_title_from_path().

___

Запрос 18: Создание blog_spider.py

Реализуй модуль r2r_scrapy/spiders/blog_spider.py для проекта R2R Scrapy.

Этот модуль должен:

1. Реализовывать паук для сбора технических блогов
2. Извлекать содержимое статей, игнорируя навигацию и сайдбары
3. Извлекать метаданные (автор, дата публикации, теги)
4. Оценивать время чтения

Используй следующие импорты:
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from datetime import datetime
import re
from r2r_scrapy.processors.html_processor import HTMLProcessor

Основной класс должен называться BlogSpider, наследоваться от CrawlSpider и иметь методы parse_blog_post(), extract_publication_date(), extract_author(), extract_tags() и calculate_reading_time().

___


Этап 6: Экспортеры


___

Запрос 19: Создание r2r_exporter.py

Реализуй модуль r2r_scrapy/exporters/r2r_exporter.py для проекта R2R Scrapy.

Этот модуль должен:

1. Экспортировать обработанные документы в R2R API
2. Поддерживать пакетную обработку документов
3. Управлять коллекциями в R2R
4. Генерировать уникальные ID для документов

Используй следующие импорты:
import asyncio
import aiohttp
import json
import hashlib
import uuid
from datetime import datetime

Основной класс должен называться R2RExporter с методами initialize(), close(), export_documents(), process_batch(), prepare_r2r_data(), generate_document_id(), get_collection_id(), determine_chunk_strategy() и create_collection().

___

Запрос 20: Создание file_exporter.py

Реализуй модуль r2r_scrapy/exporters/file_exporter.py для проекта R2R Scrapy.

Этот модуль должен:

1. Экспортировать обработанные документы в локальные файлы
2. Поддерживать различные форматы (JSON, YAML, текст)
3. Организовывать файлы по коллекциям
4. Экспортировать метаданные отдельно

Используй следующие импорты:
import os
import json
import yaml
import hashlib
from datetime import datetime

Основной класс должен называться FileExporter с методами export_documents(),_export_document(),_export_metadata(),_export_collection_metadata(), _generate_document_id() и export_chunks().

___


Этап 7: Пайплайны


___

Запрос 21: Создание preprocessing_pipeline.py

Реализуй модуль r2r_scrapy/pipelines/preprocessing_pipeline.py для проекта R2R Scrapy.

Этот модуль должен:

1. Предварительно обрабатывать собранный контент
2. Определять тип контента (HTML, Markdown, текст)
3. Очищать контент от ненужных элементов
4. Извлекать базовые метаданные

Используй следующие импорты:
import logging
from bs4 import BeautifulSoup
import re
import html2text

Основной класс должен называться PreprocessingPipeline с методами from_crawler(), process_item(), _detect_content_type(),_process_html(),_process_markdown(),_process_text(),_clean_markdown() и_extract_metadata().

___

Запрос 22: Создание content_pipeline.py

Реализуй модуль r2r_scrapy/pipelines/content_pipeline.py для проекта R2R Scrapy.

Этот модуль должен:

1. Обрабатывать контент с помощью специализированных процессоров
2. Извлекать блоки кода и API элементы
3. Обогащать метаданные
4. Подготавливать контент для чанкинга

Используй следующие импорты:
import logging
import re
from r2r_scrapy.processors.code_processor import CodeProcessor
from r2r_scrapy.processors.markdown_processor import MarkdownProcessor
from r2r_scrapy.processors.html_processor import HTMLProcessor
from r2r_scrapy.processors.api_processor import APIDocProcessor

Основной класс должен называться ContentPipeline с методами from_crawler(), process_item() и _extract_code_blocks().

___

Запрос 23: Создание chunking_pipeline.py

Реализуй модуль r2r_scrapy/pipelines/chunking_pipeline.py для проекта R2R Scrapy.

Этот модуль должен:

1. Разбивать обработанный контент на чанки
2. Выбирать оптимальную стратегию чанкинга
3. Добавлять информацию о чанкинге в метаданные
4. Подготавливать чанки для экспорта

Используй следующие импорты:
import logging
from r2r_scrapy.chunkers.semantic_chunker import SemanticChunker
from r2r_scrapy.chunkers.code_chunker import CodeChunker
from r2r_scrapy.chunkers.markdown_chunker import MarkdownChunker
from r2r_scrapy.chunkers.recursive_chunker import RecursiveChunker

Основной класс должен называться ChunkingPipeline с методами from_crawler(), process_item() и _determine_strategy().

___

Запрос 24: Создание r2r_pipeline.py

Реализуй модуль r2r_scrapy/pipelines/r2r_pipeline.py для проекта R2R Scrapy.

Этот модуль должен:

1. Экспортировать обработанные чанки в R2R
2. Обрабатывать документы пакетами
3. Подключаться к событиям открытия и закрытия паука
4. Обрабатывать ошибки экспорта

Используй следующие импорты:
import logging
import asyncio
from r2r_scrapy.exporters.r2r_exporter import R2RExporter

Основной класс должен называться R2RPipeline с методами from_crawler(), spider_opened(), spider_closed(), process_item() и _process_batch().

___


Этап 8: Middleware


___

Запрос 25: Создание javascript_middleware.py

Реализуй модуль r2r_scrapy/middleware/javascript_middleware.py для проекта R2R Scrapy.

Этот модуль должен:

1. Рендерить JavaScript на страницах с помощью Playwright
2. Обрабатывать запросы, требующие JavaScript
3. Управлять браузером и контекстом
4. Обрабатывать ошибки рендеринга

Используй следующие импорты:
from scrapy.http import HtmlResponse
from scrapy.downloadermiddlewares.retry import RetryMiddleware
import logging
import asyncio
from playwright.async_api import async_playwright

Основной класс должен называться JavaScriptMiddleware с методами from_crawler(),_start_browser(),_close_browser(),_render_page(), process_request(), spider_opened() и spider_closed().

___

Запрос 26: Создание rate_limiter.py

Реализуй модуль r2r_scrapy/middleware/rate_limiter.py для проекта R2R Scrapy.

Этот модуль должен:

1. Ограничивать скорость запросов к доменам
2. Адаптивно регулировать задержки на основе ответов сервера
3. Отслеживать успешные и неудачные запросы
4. Предотвращать блокировку со стороны серверов

Используй следующие импорты:
import time
import logging
import random
from scrapy.exceptions import IgnoreRequest
from collections import defaultdict

Основной класс должен называться RateLimiter с методами from_crawler(), process_request(), process_response(), process_exception(), _get_domain(), _get_delay(), _get_concurrent_limit() и get_stats().

___


Этап 9: Утилиты


___

Запрос 27: Создание url_prioritizer.py

Реализуй модуль r2r_scrapy/utils/url_prioritizer.py для проекта R2R Scrapy.

Этот модуль должен:

1. Приоритизировать URL для обхода
2. Определять релевантность URL для документации
3. Учитывать шаблоны URL и расширения файлов
4. Сортировать URL по приоритету

Используй следующие импорты:
import re
from urllib.parse import urlparse
import logging

Основной класс должен называться URLPrioritizer с методами get_priority(), prioritize_urls(), _matches_patterns(), _has_extension() и _get_url_depth().

___

Запрос 28: Создание resource_manager.py

Реализуй модуль r2r_scrapy/utils/resource_manager.py для проекта R2R Scrapy.

Этот модуль должен:

1. Управлять системными ресурсами для оптимальной производительности
2. Отслеживать использование CPU и памяти
3. Регулировать количество одновременных задач
4. Обрабатывать очередь задач

Используй следующие импорты:
import psutil
import time
import logging
import threading
import asyncio
from collections import deque

Основной класс должен называться ResourceManager с методами _monitor_resources(), _adjust_concurrency(), _process_queue(), _run_task(), submit_task(), get_stats() и shutdown().

___

Запрос 29: Создание quality_monitor.py

Реализуй модуль r2r_scrapy/utils/quality_monitor.py для проекта R2R Scrapy.

Этот модуль должен:

1. Отслеживать качество собранных данных
2. Собирать метрики для Prometheus
3. Предоставлять декораторы для измерения времени обработки
4. Проверять качество документов

Используй следующие импорты:
import logging
import re
import time
from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server

Основной класс должен называться QualityMonitor с методами record_page_scraped(), record_content_size(), time_processing(), record_chunk_stats(), record_r2r_response() и validate_document_quality().

___

Запрос 30: Создание version_control.py

Реализуй модуль r2r_scrapy/utils/version_control.py для проекта R2R Scrapy.

Этот модуль должен:

1. Отслеживать версии документов
2. Использовать Git для хранения истории изменений
3. Сравнивать версии документов
4. Восстанавливать предыдущие версии

Используй следующие импорты:
import os
import re
import git
import logging
import hashlib
import json
from datetime import datetime

Основной класс должен называться VersionControl с методами _derive_key(), _load_keys(), _save_keys(), add_document(), get_document_history(), get_document_version(), compare_versions(), _generate_document_id() и_calculate_document_hash().

___


Этап 10: Интеграции


___

Запрос 31: Создание github_integration.py

Реализуй модуль r2r_scrapy/integrations/github_integration.py для проекта R2R Scrapy.

Этот модуль должен:

1. Интегрироваться с GitHub API
2. Получать содержимое репозиториев
3. Получать файлы и README
4. Искать код в репозиториях

Используй следующие импорты:
import aiohttp
import asyncio
import base64
import json
import logging
from urllib.parse import quote

Основной класс должен называться GitHubIntegration с методами initialize(), close(), get_repository_contents(), get_file_content(), get_readme(), search_code(), get_repository_tree() и _get_headers().

___

Запрос 32: Создание stackoverflow_integration.py

Реализуй модуль r2r_scrapy/integrations/stackoverflow_integration.py для проекта R2R Scrapy.

Этот модуль должен:

1. Интегрироваться с Stack Overflow API
2. Искать вопросы по тегам и ключевым словам
3. Получать ответы на вопросы
4. Извлекать блоки кода из ответов

Используй следующие импорты:
import aiohttp
import asyncio
import logging
import html
import re
from urllib.parse import quote

Основной класс должен называться StackOverflowIntegration с методами initialize(), close(), search_questions(), get_question_answers(), get_question_with_answers(), search_by_tag(), _process_questions(), _process_answers(), _extract_code_blocks() и_clean_html().

___

Запрос 33: Создание wikipedia_integration.py

Реализуй модуль r2r_scrapy/integrations/wikipedia_integration.py для проекта R2R Scrapy.

Этот модуль должен:

1. Интегрироваться с Wikipedia API
2. Искать статьи по ключевым словам
3. Получать содержимое статей
4. Получать разделы статей и краткое содержание

Используй следующие импорты:
import aiohttp
import asyncio
import logging
import re
from urllib.parse import quote

Основной класс должен называться WikipediaIntegration с методами initialize(), close(), search(), get_article(), get_article_sections(), get_article_summary(),_clean_snippet() и_clean_html().

___


Этап 11: Безопасность


___

Запрос 34: Создание key_manager.py

Реализуй модуль r2r_scrapy/security/key_manager.py для проекта R2R Scrapy.

Этот модуль должен:

1. Безопасно управлять API ключами
2. Шифровать чувствительные данные
3. Хранить ключи в зашифрованном виде
4. Поддерживать ротацию ключей

Используй следующие импорты:
import os
import json
import base64
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

Основной класс должен называться KeyManager с методами _derive_key(), _load_keys(), _save_keys(), get_key(), set_key(), delete_key(), list_keys() и rotate_master_key().

___

Запрос 35: Создание secure_logger.py

Реализуй модуль r2r_scrapy/security/secure_logger.py для проекта R2R Scrapy.

Этот модуль должен:

1. Безопасно логировать события
2. Маскировать чувствительные данные в логах
3. Поддерживать различные уровни логирования
4. Сохранять логи в файлы

Используй следующие импорты:
import logging
import re
import json
import os
from datetime import datetime

Основной класс должен называться SecureLogger с методами debug(), info(), warning(), error(), critical(), _mask_sensitive_data(),_mask_json() и_mask_dict().

___


Этап 12: CLI команды


___

Запрос 36: Создание base_command.py

Реализуй модуль r2r_scrapy/cli/commands/base_command.py для проекта R2R Scrapy.

Этот модуль должен:

1. Предоставлять базовый класс для CLI команд
2. Загружать конфигурацию
3. Настраивать логирование
4. Предоставлять доступ к настройкам R2R API и Scrapy

Используй следующие импорты:
import click
import logging
import os
from r2r_scrapy.config import Config

Основной класс должен называться BaseCommand с методами get_r2r_api_settings(), get_scrapy_settings(), get_processing_settings() и get_monitoring_settings().

___

Запрос 37: Создание scrape_command.py

Реализуй модуль r2r_scrapy/cli/commands/scrape_command.py для проекта R2R Scrapy.

Этот модуль должен:

1. Реализовывать команду для сбора документации
2. Настраивать параметры сбора
3. Запускать пауков Scrapy
4. Создавать коллекции в R2R

Используй следующие импорты:
import click
import asyncio
import os
import logging
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from r2r_scrapy.cli.commands.base_command import BaseCommand
from r2r_scrapy.exporters.r2r_exporter import R2RExporter

Основной класс должен называться ScrapeCommand с методом run(). Также добавь функцию scrape() с декоратором @click.command() для регистрации в CLI.

___

Запрос 38: Создание list_collections_command.py

Реализуй модуль r2r_scrapy/cli/commands/list_collections_command.py для проекта R2R Scrapy.

Этот модуль должен:

1. Реализовывать команду для вывода списка коллекций
2. Получать коллекции из R2R API
3. Форматировать вывод в виде таблицы или JSON
4. Отображать метаданные коллекций

Используй следующие импорты:
import click
import asyncio
import os
import logging
from r2r_scrapy.cli.commands.base_command import BaseCommand
from r2r_scrapy.exporters.r2r_exporter import R2RExporter

Основной класс должен называться ListCollectionsCommand с методами _list_collections() и run(). Также добавь функцию list_collections() с декоратором @click.command() для регистрации в CLI.

___

Запрос 39: Создание create_collection_command.py

Реализуй модуль r2r_scrapy/cli/commands/create_collection_command.py для проекта R2R Scrapy.

Этот модуль должен:

1. Реализовывать команду для создания новой коллекции
2. Настраивать метаданные коллекции
3. Создавать коллекцию в R2R API
4. Отображать результат создания

Используй следующие импорты:
import click
import asyncio
import os
import logging
from r2r_scrapy.cli.commands.base_command import BaseCommand
from r2r_scrapy.exporters.r2r_exporter import R2RExporter

Основной класс должен называться CreateCollectionCommand с методами _create_collection() и run(). Также добавь функцию create_collection() с декоратором @click.command() для регистрации в CLI.

___

Запрос 40: Создание delete_collection_command.py

Реализуй модуль r2r_scrapy/cli/commands/delete_collection_command.py для проекта R2R Scrapy.

Этот модуль должен:

1. Реализовывать команду для удаления коллекции
2. Запрашивать подтверждение перед удалением
3. Удалять коллекцию из R2R API
4. Отображать результат удаления

Используй следующие импорты:
import click
import asyncio
import os
import logging
from r2r_scrapy.cli.commands.base_command import BaseCommand
from r2r_scrapy.exporters.r2r_exporter import R2RExporter

Основной класс должен называться DeleteCollectionCommand с методами _delete_collection() и run(). Также добавь функцию delete_collection() с декоратором @click.command() для регистрации в CLI.

___

Запрос 41: Создание list_documents_command.py

Реализуй модуль r2r_scrapy/cli/commands/list_documents_command.py для проекта R2R Scrapy.

Этот модуль должен:

1. Реализовывать команду для вывода списка документов
2. Фильтровать документы по коллекции
3. Поддерживать пагинацию
4. Форматировать вывод в виде таблицы или JSON

Используй следующие импорты:
import click
import asyncio
import os
import logging
from r2r_scrapy.cli.commands.base_command import BaseCommand
from r2r_scrapy.exporters.r2r_exporter import R2RExporter

Основной класс должен называться ListDocumentsCommand с методами _list_documents() и run(). Также добавь функцию list_documents() с декоратором @click.command() для регистрации в CLI.

___

Запрос 42: Создание get_document_command.py

Реализуй модуль r2r_scrapy/cli/commands/get_document_command.py для проекта R2R Scrapy.

Этот модуль должен:

1. Реализовывать команду для получения документа
2. Получать документ из R2R API
3. Форматировать вывод в различных форматах
4. Сохранять документ в файл

Используй следующие импорты:
import click
import asyncio
import os
import logging
from r2r_scrapy.cli.commands.base_command import BaseCommand
from r2r_scrapy.exporters.r2r_exporter import R2RExporter

Основной класс должен называться GetDocumentCommand с методами _get_document() и run(). Также добавь функцию get_document() с декоратором @click.command() для регистрации в CLI.

___

Запрос 43: Создание delete_document_command.py

Реализуй модуль r2r_scrapy/cli/commands/delete_document_command.py для проекта R2R Scrapy.

Этот модуль должен:

1. Реализовывать команду для удаления документа
2. Запрашивать подтверждение перед удалением
3. Удалять документ из R2R API
4. Отображать результат удаления

Используй следующие импорты:
import click
import asyncio
import os
import logging
from r2r_scrapy.cli.commands.base_command import BaseCommand
from r2r_scrapy.exporters.r2r_exporter import R2RExporter

Основной класс должен называться DeleteDocumentCommand с методами _delete_document() и run(). Также добавь функцию delete_document() с декоратором @click.command() для регистрации в CLI.

___

Запрос 44: Создание generate_report_command.py

Реализуй модуль r2r_scrapy/cli/commands/generate_report_command.py для проекта R2R Scrapy.

Этот модуль должен:

1. Реализовывать команду для генерации отчета о качестве
2. Собирать статистику по коллекциям и документам
3. Генерировать отчет в HTML или JSON формате
4. Сохранять отчет в файл

Используй следующие импорты:
import click
import asyncio
import os
import logging
import json
from datetime import datetime
from r2r_scrapy.cli.commands.base_command import BaseCommand
from r2r_scrapy.exporters.r2r_exporter import R2RExporter

Основной класс должен называться GenerateReportCommand с методами _get_collections(), _get_documents(), _generate_html_report(),_generate_json_report() и run(). Также добавь функцию generate_report() с декоратором @click.command() для регистрации в CLI.

___

Запрос 45: Создание cli/main.py

Реализуй модуль r2r_scrapy/cli/main.py для проекта R2R Scrapy.

Этот модуль должен:

1. Регистрировать все CLI команды
2. Создавать группу команд с помощью click
3. Добавлять информацию о версии
4. Предоставлять точку входа для CLI

Используй следующие импорты:
import click
from r2r_scrapy.cli.commands.scrape_command import scrape
from r2r_scrapy.cli.commands.list_collections_command import list_collections
from r2r_scrapy.cli.commands.create_collection_command import create_collection
from r2r_scrapy.cli.commands.delete_collection_command import delete_collection
from r2r_scrapy.cli.commands.list_documents_command import list_documents
from r2r_scrapy.cli.commands.get_document_command import get_document
from r2r_scrapy.cli.commands.delete_document_command import delete_document
from r2r_scrapy.cli.commands.generate_report_command import generate_report

Создай функцию cli() с декоратором @click.group() и добавь все команды с помощью cli.add_command().

___


Этап 13: Тесты


___

Запрос 46: Создание базовых тестов

Создай базовые тесты для проекта R2R Scrapy. Реализуй следующие тестовые файлы:

1. tests/test_config.py - тесты для Config
2. tests/test_processors.py - тесты для процессоров
3. tests/test_chunkers.py - тесты для чанкеров
4. tests/test_exporters.py - тесты для экспортеров

Используй pytest и unittest.mock для создания тестов. Каждый тест должен проверять основную функциональность соответствующего компонента.

___

Запрос 47: Создание тестов для CLI

Создай тесты для CLI команд проекта R2R Scrapy. Реализуй следующие тестовые файлы:

1. tests/test_cli_commands.py - тесты для CLI команд
2. tests/test_cli_main.py - тесты для основного CLI интерфейса

Используй pytest, click.testing.CliRunner и unittest.mock для создания тестов. Тесты должны проверять правильность обработки аргументов и вызова соответствующих методов.

___


Этап 14: Документация


___

Запрос 48: Создание документации по установке и быстрому старту

Создай документацию по установке и быстрому старту для проекта R2R Scrapy. Документация должна включать:

1. Требования к системе
2. Установка из PyPI
3. Установка из исходного кода
4. Установка с помощью Docker
5. Настройка R2R API
6. Первый запуск
7. Примеры базового использования

Документация должна быть в формате Markdown и должна быть понятна новым пользователям.

___

Запрос 49: Создание документации по конфигурации

Создай документацию по конфигурации для проекта R2R Scrapy. Документация должна включать:

1. Конфигурационный файл YAML
2. Переменные окружения
3. Аргументы командной строки
4. Приоритет настроек
5. Примеры конфигурации для разных сценариев

Документация должна быть в формате Markdown и должна содержать примеры для всех поддерживаемых опций.

___

Запрос 50: Создание документации по API

Создай документацию по API для проекта R2R Scrapy. Документация должна включать:

1. CLI API (все команды и их параметры)
2. Python API (основные классы и методы)
3. Примеры использования API
4. Интеграция с другими системами

Документация должна быть в формате Markdown и должна содержать примеры кода для всех основных функций.
Эти 50 запросов позволят полностью реализовать проект R2R Scrapy с помощью Claude 3.7 Sonnet, разбивая сложную задачу на управляемые части, которые модель может обрабатывать эффективно.

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Union, cast
from urllib.parse import urlparse

import scrapy
from scrapy.http import Response, TextResponse
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule

from r2r_scrapy.processors.html_processor import HTMLProcessor


class ApiDocSpider(CrawlSpider):
    """Spider для API документации и фреймворков"""

    name = "api_doc"

    def __init__(
        self,
        url: str,
        library: str = "",
        allowed_paths: Optional[str] = None,
        follow_links: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Инициализирует паук для сбора API документации.

        Args:
            url: URL для начала сканирования
            library: Название библиотеки или API
            allowed_paths: Разрешенные пути для сканирования (через запятую)
            follow_links: Следовать по ссылкам или нет
            **kwargs: Дополнительные параметры
        """
        self.start_urls = [url]
        self.library = library

        # Определение разрешенных доменов из URL
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        self.allowed_domains = [domain]

        # Установка правил для следования по ссылкам
        if follow_links:
            # Создаем различные экстракторы ссылок для разных типов документации

            # Основной экстрактор для документации API - следует за ссылками с нужными путями
            api_link_extractor = LinkExtractor(
                allow_domains=[domain],
                deny=[
                    r".*\.(css|js|jpg|jpeg|png|gif|svg|ico|xml|zip|tar|gz|woff|woff2|ttf|eot)$",
                    r".*dashboard.*",
                    r".*login.*",
                    r".*signin.*",
                    r".*account.*",
                ],
                unique=True,
            )

            # Экстрактор для навигации - более широкие селекторы
            nav_link_extractor = LinkExtractor(
                restrict_css=(
                    "nav",
                    ".nav",
                    ".navbar",
                    ".navigation",
                    ".sidebar",
                    ".menu",
                    ".toc",
                    ".site-nav",
                    ".table-of-contents",
                    ".docs-nav",
                    ".doc-nav",
                ),
                allow_domains=[domain],
                unique=True,
            )

            # Устанавливаем правила для CrawlSpider
            self.rules = (
                Rule(
                    api_link_extractor,
                    callback="parse_documentation",
                    follow=True,
                ),
                Rule(nav_link_extractor, follow=True),
            )
        else:
            self.rules = ()

        # Инициализируем процессор HTML
        self.html_processor = HTMLProcessor()

        # Если есть allowed_paths, добавим их в ограничения
        if allowed_paths:
            self.allowed_paths = [
                path.strip() for path in allowed_paths.split(",")
            ]
        else:
            self.allowed_paths = []

        # Вызываем родительский конструктор после настройки правил
        super().__init__(**kwargs)

        # Создаем логгер вместо переопределения
        self._custom_logger = logging.getLogger(self.__class__.__name__)
        self._custom_logger.info(f"Initializing {self.name} spider")
        self._custom_logger.info(f"Start URL: {url}")
        self._custom_logger.info(f"Allowed domains: {self.allowed_domains}")
        self._custom_logger.info(
            f"Spider initialization complete for {self.name}"
        )

    def _extract_library_name(self, url: str) -> str:
        """Извлекает название библиотеки из URL, если не указано"""
        domain = urlparse(url).netloc

        # Пытаемся извлечь имя библиотеки из домена
        domain_parts = domain.split(".")
        if len(domain_parts) >= 2:
            potential_names = [
                part
                for part in domain_parts
                if part not in ["com", "org", "io", "net", "docs"]
            ]
            if potential_names:
                return potential_names[0]

        path = urlparse(url).path
        path_parts = [p for p in path.split("/") if p]
        if path_parts:
            return path_parts[0]

        return domain

    def parse_start_url(self, response: Response, **kwargs: Any) -> Iterator:
        """Обработка начальной страницы"""
        self._custom_logger.info(f"Parsing start URL: {response.url}")

        # Всегда извлекаем данные из начальной страницы
        yield from self.parse_documentation(response)

        # TextResponse требуется для extract_links
        if not isinstance(response, TextResponse):
            return

        # Затем позволяем CrawlSpider обрабатывать ссылки, если есть правила
        for rule in self._rules:
            if rule.link_extractor:
                links = rule.link_extractor.extract_links(response)
                for link in links:
                    yield response.follow(link, callback=self.parse)

    def parse(self, response: Response) -> Iterator:
        """
        Базовый метод для обработки страниц
        
        Этот метод вызывается Scrapy по умолчанию и перенаправляет обработку
        на метод parse_documentation
        """
        return self.parse_documentation(response)

    def parse_documentation(self, response: Response) -> Iterator:
        """
        Обработка документации API

        Этот метод извлекает и обрабатывает данные с документационных страниц.
        Выполняет:
        1. Очистку HTML от ненужных элементов
        2. Извлечение структурированных данных документации
        3. Генерацию элементов для последующей обработки
        """
        self._custom_logger.info(f"Processing document: {response.url}")

        # Проверка URL - если имеются допустимые пути и текущий URL не соответствует - пропускаем
        if self.allowed_paths:
            parsed_url = urlparse(response.url)
            path = parsed_url.path

            path_allowed = any(
                allowed in path for allowed in self.allowed_paths
            )
            if not path_allowed:
                self._custom_logger.debug(
                    f"Skipping URL (not in allowed paths): {response.url}"
                )
                return

        # Получение заголовка страницы
        title = response.css(
            "title::text"
        ).get() or self._extract_title_from_url(response.url)

        # Обработка HTML-содержимого
        processed_result = self.html_processor.process(response.text)
        # Метод process возвращает кортеж (content, metadata)
        if isinstance(processed_result, tuple) and len(processed_result) >= 1:
            content = processed_result[0]
            page_metadata = processed_result[1] if len(processed_result) > 1 else {}
        else:
            content = processed_result
            page_metadata = {}

        # Если содержимое пустое, логируем и пропускаем
        if not content or not isinstance(content, str) or not content.strip():
            self._custom_logger.warning(
                f"Empty content after processing for URL: {response.url}"
            )
            return

        # Создаем элемент для последующей обработки в pipeline
        item = {
            "url": response.url,
            "title": title,
            "content": content,
            "content_type": "html",
            "library": self.library
            or self._extract_library_name(response.url),
            "type": "api_doc",
            "metadata": {
                "headers": self._extract_headers(response),
                "description": self._extract_description(response),
                "keywords": self._extract_keywords(response),
                "language": self._detect_language(content),
                **page_metadata  # Добавляем метаданные, полученные из HTML-процессора
            },
        }

        yield item

    def _extract_title_from_url(self, url: str) -> str:
        """Извлекает заголовок из URL, если не удалось найти в HTML"""
        path = urlparse(url).path
        path_parts = [p for p in path.split("/") if p]

        if path_parts:
            # Берем последнюю часть пути и заменяем дефисы и подчеркивания пробелами
            title = (
                path_parts[-1]
                .replace("-", " ")
                .replace("_", " ")
                .replace(".html", "")
            )
            # Делаем первую букву заглавной
            return title.capitalize()

        # Если не удалось извлечь из пути, берем домен
        domain = urlparse(url).netloc
        return domain

    def _safe_to_string(self, value: Any) -> str:
        """Безопасно преобразует значение любого типа в строку"""
        if value is None:
            return ""

        # Для строк просто возвращаем их
        if isinstance(value, str):
            return value

        # Для байтов декодируем
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8", errors="ignore")
            except Exception:
                return str(value)

        # Для списков берем первый элемент, если он есть
        if isinstance(value, list):
            if not value:
                return ""
            return self._safe_to_string(value[0])

        # Для всех остальных типов используем str()
        return str(value)

    def _extract_headers(self, response: Response) -> Dict[str, str]:
        """Извлекает заголовки HTTP из ответа"""
        result: Dict[str, str] = {}

        for key in response.headers.keys():
            if key is None:
                continue

            # Безопасно преобразуем ключ в строку
            key_str = self._safe_to_string(key)
            if not key_str:
                continue

            # Получаем значение заголовка и преобразуем его в строку
            value = response.headers.get(key)
            result[key_str] = self._safe_to_string(value)

        return result

    def _extract_description(self, response: Response) -> str:
        """Извлекает мета-описание страницы"""
        return (
            response.css('meta[name="description"]::attr(content)').get() or ""
        )

    def _extract_keywords(self, response: Response) -> List[str]:
        """Извлекает ключевые слова из мета-тегов"""
        keywords = (
            response.css('meta[name="keywords"]::attr(content)').get() or ""
        )
        return [k.strip() for k in keywords.split(",") if k.strip()]

    def _detect_language(self, content: str) -> str:
        """Определяет язык контента"""
        # Простая эвристика - проверяем наличие русских символов
        if re.search("[а-яА-Я]", content):
            return "ru"
        return "en"  # По умолчанию английский

    def _extract_section(self, response: Response) -> str:
        """Определяет раздел документации на основе URL и структуры страницы"""
        path = urlparse(response.url).path
        path_parts = path.split("/")

        # Если путь содержит явные указатели на разделы
        for section_name in ["api", "guides", "tutorial", "reference", "docs"]:
            if section_name in path_parts:
                return section_name

        # Если нет явных указателей, определяем по заголовку
        h1_text = response.css("h1::text").get() or ""
        for section_name in [
            "API",
            "Guide",
            "Tutorial",
            "Reference",
            "Documentation",
        ]:
            if section_name in h1_text:
                return section_name.lower()

        return "general"

    def _extract_last_updated(self, response: Response) -> str:
        """Извлекает дату последнего обновления страницы"""
        # Ищем даты в нескольких форматах
        for selector in [
            "time::attr(datetime)",
            'meta[name="last-modified"]::attr(content)',
            ".last-updated::text",
            ".updated-date::text",
        ]:
            date = response.css(selector).get()
            if date:
                return date.strip()

        return ""

    def _extract_version(self, response: Response) -> str:
        """Извлекает версию документируемого API или библиотеки"""
        # Общие селекторы для поиска информации о версии
        for selector in [
            ".version::text",
            ".version-info::text",
            'meta[name="version"]::attr(content)',
        ]:
            version = response.css(selector).get()
            if version:
                return version.strip()

        # Поиск в хлебных крошках
        breadcrumbs = " ".join(
            response.css(".breadcrumbs::text, .breadcrumb::text").getall()
        )
        version_match = re.search(r"v(\d+\.\d+\.?\d*)", breadcrumbs)
        if version_match:
            return version_match.group(1)

        return "latest"

    def _determine_content_type(self, response: Response) -> str:
        """Определяет тип содержимого страницы"""
        # Получаем заголовки ответа
        headers = self._extract_headers(response)
        content_type = ""

        # Ищем заголовок Content-Type в различных вариантах написания
        for header_name in ["Content-Type", "content-type", "CONTENT-TYPE"]:
            if header_name in headers:
                content_type = headers[header_name]
                break

        # Определяем тип контента по значению заголовка
        if "text/html" in content_type:
            return "html"
        elif "application/json" in content_type:
            return "json"
        elif "text/plain" in content_type:
            return "text"
        elif "application/xml" in content_type or "text/xml" in content_type:
            return "xml"
        else:
            # По умолчанию считаем, что это HTML
            return "html"

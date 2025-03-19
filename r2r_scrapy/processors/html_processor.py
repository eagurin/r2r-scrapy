import re

import html2text
from bs4 import BeautifulSoup

from r2r_scrapy.processors.code_processor import CodeProcessor


class HTMLProcessor:
    """Process HTML content"""

    def __init__(self):
        self.code_processor = CodeProcessor()
        self.html2text_converter = html2text.HTML2Text()
        self.html2text_converter.ignore_links = False
        self.html2text_converter.ignore_images = False
        self.html2text_converter.ignore_tables = False
        self.html2text_converter.body_width = 0  # No wrapping

    def process(self, response, content_css=None):
        """
        Обрабатывает HTML содержимое.
        
        Args:
            response: Объект Response или строка HTML
            content_css: CSS селектор для извлечения содержимого (опционально)
            
        Returns:
            str: Обработанное HTML содержимое, готовое для дальнейшей обработки
        """
        # Проверяем тип входного параметра и извлекаем HTML
        if content_css:
            if hasattr(response, 'css'):
                # Если передан объект Response, используем css-метод
                html_content = response.css(content_css).get(
                    default="<html><body>No content found</body></html>"
                )
            else:
                # Если передана строка, уже считаем ее HTML-содержимым
                html_content = response
        else:
            if hasattr(response, 'css'):
                # Если передан объект Response, извлекаем body
                html_content = response.css("body").get()
                if not html_content:
                    # Если body не найден, берем весь HTML
                    html_content = response.text
            else:
                # Если передана строка, используем ее напрямую
                html_content = response

        # Parse HTML
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove navigation, sidebars, footers, ads
        for element in soup.select(
            "nav, .sidebar, .navigation, footer, .footer, .menu, .ads, .advertisement"
        ):
            if element:
                element.decompose()

        # Process code blocks
        code_blocks = soup.find_all(["pre", "code"])
        for block in code_blocks:
            code_text = block.get_text()
            language = None

            # Try to detect language from class
            if block.get("class"):
                for cls in block.get("class"):
                    if cls.startswith("language-") or cls.startswith("lang-"):
                        language = cls.replace("language-", "").replace(
                            "lang-", ""
                        )
                        break

            # Process code
            processed_code = self.code_processor.process_code(
                code_text, language
            )

            # Replace original code with processed version if needed
            # For now, we'll keep the original code in the HTML

        # Extract metadata
        metadata = self.extract_metadata(response, soup)

        # Convert HTML to Markdown
        markdown_content = self.html2text_converter.handle(str(soup))

        # Clean up markdown content
        cleaned_content = self.clean_markdown(markdown_content)

        return cleaned_content, metadata

    def extract_metadata(self, response, soup):
        """Extract metadata from HTML"""
        metadata = {}

        # Extract title
        if soup.title:
            metadata["title"] = soup.title.get_text()
        elif response and hasattr(response, "css"):
            metadata["title"] = response.css("title::text").get()

        # Extract meta description
        description = soup.find("meta", attrs={"name": "description"})
        if description:
            metadata["description"] = description.get("content", "")

        # Extract meta keywords
        keywords = soup.find("meta", attrs={"name": "keywords"})
        if keywords:
            metadata["keywords"] = [
                k.strip() for k in keywords.get("content", "").split(",")
            ]

        # Extract canonical URL
        canonical = soup.find("link", attrs={"rel": "canonical"})
        if canonical:
            metadata["canonical_url"] = canonical.get("href", "")

        # Extract Open Graph metadata
        og_title = soup.find("meta", attrs={"property": "og:title"})
        if og_title:
            metadata["og_title"] = og_title.get("content", "")

        og_description = soup.find(
            "meta", attrs={"property": "og:description"}
        )
        if og_description:
            metadata["og_description"] = og_description.get("content", "")

        og_image = soup.find("meta", attrs={"property": "og:image"})
        if og_image:
            metadata["og_image"] = og_image.get("content", "")

        # Extract headings
        headings = []
        for level in range(1, 7):
            for heading in soup.find_all(f"h{level}"):
                headings.append(
                    {"level": level, "text": heading.get_text().strip()}
                )

        metadata["headings"] = headings

        # Extract links
        links = []
        for link in soup.find_all("a"):
            href = link.get("href")
            if href:
                links.append({"text": link.get_text(), "url": href})

        metadata["links"] = links

        return metadata

    def clean_markdown(self, markdown):
        """Clean up markdown content"""
        # Remove excessive newlines
        cleaned = re.sub(r"\n{3,}", "\n\n", markdown)

        # Fix code block formatting
        cleaned = re.sub(r"```\s+", "```\n", cleaned)
        cleaned = re.sub(r"\s+```", "\n```", cleaned)

        # Fix list formatting
        cleaned = re.sub(r"(\n[*-]\s+[^\n]+)(\n[^\n*-])", r"\1\n\2", cleaned)

        return cleaned

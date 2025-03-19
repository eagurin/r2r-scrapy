import os
from r2r_scrapy.config import Config

# Load configuration
config = Config(os.environ.get('R2R_SCRAPY_CONFIG', 'config.yaml'))

# Scrapy settings
BOT_NAME = 'r2r_scrapy'

SPIDER_MODULES = ['r2r_scrapy.spiders']
NEWSPIDER_MODULE = 'r2r_scrapy.spiders'

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Configure maximum concurrent requests
CONCURRENT_REQUESTS = config.get('scrapy.concurrent_requests', 16)
CONCURRENT_REQUESTS_PER_DOMAIN = config.get('scrapy.concurrent_requests_per_domain', 8)
DOWNLOAD_DELAY = config.get('scrapy.download_delay', 0.5)

# User agent
USER_AGENT = config.get('scrapy.user_agent', 'R2R Scrapy/1.0 (+https://github.com/eagurin/r2r-scrapy)')

# Enable or disable cookies
COOKIES_ENABLED = False

# Configure item pipelines
ITEM_PIPELINES = {
    'r2r_scrapy.pipelines.preprocessing_pipeline.PreprocessingPipeline': 300,
    'r2r_scrapy.pipelines.content_pipeline.ContentPipeline': 400,
    'r2r_scrapy.pipelines.chunking_pipeline.ChunkingPipeline': 500,
    'r2r_scrapy.pipelines.r2r_pipeline.R2RPipeline': 600,
}

# Enable and configure HTTP caching
HTTPCACHE_ENABLED = True
HTTPCACHE_EXPIRATION_SECS = 86400  # 1 day
HTTPCACHE_DIR = 'httpcache'
HTTPCACHE_IGNORE_HTTP_CODES = [500, 502, 503, 504, 400, 401, 403, 404]
HTTPCACHE_STORAGE = 'scrapy.extensions.httpcache.FilesystemCacheStorage'

# JavaScript rendering settings (if enabled)
if config.get('scrapy.javascript_rendering', False):
    SPLASH_URL = config.get('scrapy.splash_url', 'http://localhost:8050')
    DOWNLOADER_MIDDLEWARES = {
        'scrapy_splash.SplashCookiesMiddleware': 723,
        'scrapy_splash.SplashMiddleware': 725,
        'scrapy.downloadermiddlewares.httpcompression.HttpCompressionMiddleware': 810,
    }
    SPIDER_MIDDLEWARES = {
        'scrapy_splash.SplashDeduplicateArgsMiddleware': 100,
    }
    DUPEFILTER_CLASS = 'scrapy_splash.SplashAwareDupeFilter'

# Monitoring settings
if config.get('monitoring.enabled', False):
    PROMETHEUS_PORT = config.get('monitoring.prometheus_port', 9090)
    QUALITY_THRESHOLD = config.get('monitoring.quality_threshold', 0.8)
    ALERT_ON_ERROR = config.get('monitoring.alert_on_error', True)

# R2R API settings
R2R_API_URL = config.get('r2r.api_url')
R2R_API_KEY = config.get('r2r.api_key')
R2R_BATCH_SIZE = config.get('r2r.batch_size', 10)
R2R_MAX_CONCURRENCY = config.get('r2r.max_concurrency', 5)

# Processing settings
DEFAULT_CHUNKING_STRATEGY = config.get('processing.default_chunking_strategy', 'semantic')
CHUNK_SIZE = config.get('processing.chunk_size', 800)
CHUNK_OVERLAP = config.get('processing.chunk_overlap', 150)
PRESERVE_CODE_BLOCKS = config.get('processing.preserve_code_blocks', True)
EXTRACT_METADATA = config.get('processing.extract_metadata', True)

# Security settings
SECURE_MODE = config.get('security.secure_mode', True)
API_KEY_ENCRYPTION = config.get('security.api_key_encryption', True)
SECURE_LOGGING = config.get('security.secure_logging', True)

# Export settings
EXPORT_FORMAT = config.get('export.format', 'json')
EXPORT_COMPRESSION = config.get('export.compression', None)
EXPORT_BATCH_SIZE = config.get('export.batch_size', 100)

# Integration settings
GITHUB_TOKEN = config.get('integrations.github_token')
STACKOVERFLOW_KEY = config.get('integrations.stackoverflow_key')
WIKIPEDIA_USER_AGENT = config.get('integrations.wikipedia_user_agent', USER_AGENT)

# Custom settings
settings = {k: v for k, v in locals().items() if k.isupper()}

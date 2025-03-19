"""
R2R Scrapy Middleware Components
"""

from r2r_scrapy.middleware.javascript_middleware import JavaScriptMiddleware
from r2r_scrapy.middleware.rate_limiter import RateLimiter

__all__ = [
    'JavaScriptMiddleware',
    'RateLimiter',
]

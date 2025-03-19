"""
R2R Scrapy Processing Pipelines
"""

from r2r_scrapy.pipelines.preprocessing_pipeline import PreprocessingPipeline
from r2r_scrapy.pipelines.content_pipeline import ContentPipeline
from r2r_scrapy.pipelines.chunking_pipeline import ChunkingPipeline
from r2r_scrapy.pipelines.r2r_pipeline import R2RPipeline

__all__ = [
    'PreprocessingPipeline',
    'ContentPipeline',
    'ChunkingPipeline',
    'R2RPipeline',
]

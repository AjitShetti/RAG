"""Re-exports for the OpenSearch service package."""

from .index_mapping import INDEX_NAME, PAPER_INDEX_MAPPING
from .service import OpenSearchService

__all__ = ["OpenSearchService", "INDEX_NAME", "PAPER_INDEX_MAPPING"]

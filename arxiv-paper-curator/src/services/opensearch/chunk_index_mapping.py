"""OpenSearch mapping definition for chunk-level index with kNN vector support."""

from ...config import settings

CHUNK_INDEX_NAME = "paper_chunks_v1"

CHUNK_INDEX_MAPPING = {
    "settings": {
        "index": {
            "knn": True,
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "analysis": {
            "analyzer": {
                "scientific_english": {
                    "type": "english",
                    "stopwords": "_english_",
                }
            }
        },
    },
    "mappings": {
        "properties": {
            "chunk_id": {
                "type": "keyword",
            },
            "paper_id": {
                "type": "keyword",
            },
            "section_name": {
                "type": "keyword",
            },
            "chunk_index": {
                "type": "integer",
            },
            "text": {
                "type": "text",
                "analyzer": "scientific_english",
                "term_vector": "with_positions_offsets",
            },
            "embedding": {
                "type": "knn_vector",
                "dimension": settings.embeddings_dimensions,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "nmslib",
                },
            },
            "title": {
                "type": "text",
                "analyzer": "scientific_english",
                "fields": {
                    "raw": {"type": "keyword"},
                },
            },
            "authors": {
                "type": "keyword",
            },
            "category": {
                "type": "keyword",
            },
            "published_date": {
                "type": "date",
                "format": "yyyy-MM-dd||strict_date_optional_time||epoch_millis",
            },
            "pdf_url": {
                "type": "keyword",
                "index": False,
            },
            "parse_status": {
                "type": "keyword",
            },
        }
    },
}

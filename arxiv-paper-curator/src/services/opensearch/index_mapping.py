"""OpenSearch index mapping definition for arXiv papers.

Optimized for BM25 keyword search over scientific text.
Uses the `english` analyzer for stem-matching and stop-word filtering.
Fields boosted in multi_match: title > abstract > section_headings > section_bodies.
Category and published_date use exact types (keyword / date) for fast filtering.
"""

INDEX_NAME = "papers_v1"

PAPER_INDEX_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
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
            "arxiv_id": {
                "type": "keyword",
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
            "abstract": {
                "type": "text",
                "analyzer": "scientific_english",
            },
            "section_headings": {
                "type": "text",
                "analyzer": "scientific_english",
            },
            "section_bodies": {
                "type": "text",
                "analyzer": "scientific_english",
                "term_vector": "with_positions_offsets",
            },
            "full_text": {
                "type": "text",
                "analyzer": "scientific_english",
                "term_vector": "with_positions_offsets",
            },
            "pdf_url": {
                "type": "keyword",
                "index": False,
            },
            "published_date": {
                "type": "date",
                "format": "yyyy-MM-dd||strict_date_optional_time||epoch_millis",
            },
            "category": {
                "type": "keyword",
            },
            "parse_status": {
                "type": "keyword",
            },
        }
    },
}

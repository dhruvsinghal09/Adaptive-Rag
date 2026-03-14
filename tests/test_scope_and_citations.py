import unittest
from types import SimpleNamespace

from src.rag.citations import extract_citations, render_citations
from src.rag.scope import get_collection_name


class ScopeAndCitationsTests(unittest.TestCase):
    def test_collection_name_is_scoped_and_sanitized(self):
        name = get_collection_name("Team A", "User/Session#1")
        self.assertEqual(name, "adaptive_rag__team_a__user_session_1")

    def test_extract_citations_deduplicates(self):
        docs = [
            SimpleNamespace(metadata={"document_id": "d1", "chunk_id": 0, "filename": "a.pdf", "page": 1}),
            SimpleNamespace(metadata={"document_id": "d1", "chunk_id": 0, "filename": "a.pdf", "page": 1}),
            SimpleNamespace(metadata={"document_id": "d1", "chunk_id": 1, "filename": "a.pdf", "page": 2}),
        ]
        citations = extract_citations(docs)
        self.assertEqual(len(citations), 2)
        self.assertEqual(citations[0]["filename"], "a.pdf")

    def test_render_citations(self):
        citations = [{"filename": "a.pdf", "page": 1, "chunk_id": 2}]
        rendered = render_citations(citations)
        self.assertIn("Sources:", rendered)
        self.assertIn("a.pdf, page 1, chunk 2", rendered)


if __name__ == "__main__":
    unittest.main()

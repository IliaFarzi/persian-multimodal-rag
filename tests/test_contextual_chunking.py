from dataclasses import dataclass, field
import unittest

from chunking import ContextualScientificChunker


@dataclass
class Region:
    region_index: int
    caption: str
    region_type: str = "image"

    @property
    def label(self):
        labels = {
            "image": "توضیح تصویر",
            "table": "توضیح جدول",
            "formula": "توضیح فرمول",
        }
        return labels.get(self.region_type, "توضیح بخش تصویری")

    def inline_text(self):
        return f"[{self.label}: {self.caption}]" if self.caption else ""


@dataclass
class Paragraph:
    paragraph_index: int
    text: str
    heading_text: str = "Methods"
    page_number: int = 1
    is_heading: bool = False
    regions: list[Region] = field(default_factory=list)

    @property
    def ocr_text(self):
        return self.text

    @property
    def sort_key(self):
        return (self.paragraph_index, self.paragraph_index, 0)


class ContextualScientificChunkerTest(unittest.TestCase):
    def test_short_anchor_expands_with_neighbor_context(self):
        chunker = ContextualScientificChunker(target_min=120, hard_max=500)
        paragraphs = [
            Paragraph(1, "Previous paragraph discusses amyloid markers."),
            Paragraph(2, "Short anchor."),
            Paragraph(3, "Next paragraph discusses tau and MRI measures."),
        ]

        chunks = chunker.build_page_chunks(paragraphs, file_name="paper.pdf")
        anchor_chunk = next(
            chunk for chunk in chunks if chunk["paragraph_index"] == 2
        )

        self.assertEqual(anchor_chunk["paragraph_span"], "1-3")
        self.assertIn("Previous paragraph", anchor_chunk["display_text"])
        self.assertIn("Short anchor", anchor_chunk["display_text"])
        self.assertIn("Next paragraph", anchor_chunk["display_text"])
        self.assertIn("File / سند: paper.pdf", anchor_chunk["retrieval_text"])
        self.assertIn("Heading / عنوان: Methods", anchor_chunk["retrieval_text"])

    def test_long_anchor_splits_and_keeps_heading_context(self):
        chunker = ContextualScientificChunker(target_max=120, hard_max=180)
        long_text = (
            "This sentence describes cortical atrophy and MRI biomarkers. " * 8
        ).strip()
        paragraphs = [Paragraph(1, long_text, heading_text="Results")]

        chunks = chunker.build_page_chunks(paragraphs, file_name="paper.pdf")

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(chunk["paragraph_span"], "1")
            self.assertIn("Heading / عنوان: Results", chunk["retrieval_text"])
            self.assertIn("Page / صفحه: 1", chunk["retrieval_text"])

    def test_unrelated_caption_is_excluded_from_anchor_chunk(self):
        chunker = ContextualScientificChunker(target_min=120, hard_max=500)
        paragraphs = [
            Paragraph(
                1,
                "A different section with a figure.",
                heading_text="Background",
                regions=[Region(4, "unrelated architecture diagram")],
            ),
            Paragraph(2, "Anchor paragraph about diagnostic accuracy.", "Methods"),
        ]

        chunks = chunker.build_page_chunks(paragraphs, file_name="paper.pdf")
        anchor_chunk = next(
            chunk for chunk in chunks if chunk["paragraph_index"] == 2
        )

        self.assertNotIn("unrelated architecture", anchor_chunk["display_text"])
        self.assertNotIn("unrelated architecture", anchor_chunk["retrieval_text"])
        self.assertEqual(anchor_chunk["region_indices"], [])

    def test_visual_only_region_creates_retrievable_chunk(self):
        chunker = ContextualScientificChunker()
        paragraphs = [
            Paragraph(
                5,
                "",
                heading_text="Results",
                regions=[Region(7, "AUC is 0.91 for the classifier", "table")],
            )
        ]

        chunks = chunker.build_page_chunks(paragraphs, file_name="paper.pdf")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["paragraph_span"], "5")
        self.assertEqual(chunks[0]["region_indices"], [7])
        self.assertEqual(chunks[0]["ocr_text"], "")
        self.assertIn("AUC is 0.91", chunks[0]["image_caption"])
        self.assertIn("Visual caption", chunks[0]["retrieval_text"])


if __name__ == "__main__":
    unittest.main()

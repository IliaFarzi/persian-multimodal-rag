import unittest

from layout_extraction import LayoutSegment, PageLayout, VisualRegion


class PageLayoutTest(unittest.TestCase):
    def test_heading_is_propagated_and_region_attaches_below_paragraph(self):
        heading = LayoutSegment(
            kind="text",
            bbox=(50, 20, 250, 34),
            text="1 Introduction",
            block_num=1,
            par_num=1,
            line_num=1,
        )
        body = LayoutSegment(
            kind="text",
            bbox=(50, 70, 390, 92),
            text="This paragraph describes the scientific method in detail.",
            block_num=2,
            par_num=1,
            line_num=1,
        )
        region = VisualRegion(
            document_id="doc-1",
            page_number=1,
            region_index=2,
            region_type="image",
            bbox=(55, 98, 360, 180),
            crop_path="/tmp/region.png",
            prompt="",
            caption="a related figure",
        )
        page_layout = PageLayout(
            page_number=1,
            text_segments=[heading, body],
            region_segments=[],
            regions=[region],
            page_bbox=(0, 0, 600, 800),
        )

        paragraphs = page_layout.paragraphs()
        body_paragraph = next(
            paragraph for paragraph in paragraphs if body in paragraph.lines
        )

        self.assertTrue(paragraphs[0].is_heading)
        self.assertEqual(body_paragraph.heading_text, "1 Introduction")
        self.assertEqual(body_paragraph.regions, [region])
        self.assertEqual(region.paragraph_index, body_paragraph.paragraph_index)

    def test_unrelated_region_becomes_visual_only_paragraph(self):
        body = LayoutSegment(
            kind="text",
            bbox=(50, 70, 300, 92),
            text="This paragraph is in the left column.",
            block_num=1,
            par_num=1,
            line_num=1,
        )
        region = VisualRegion(
            document_id="doc-1",
            page_number=1,
            region_index=8,
            region_type="table",
            bbox=(430, 400, 560, 470),
            crop_path="/tmp/table.png",
            prompt="",
            caption="an unrelated table",
        )
        page_layout = PageLayout(
            page_number=1,
            text_segments=[body],
            region_segments=[],
            regions=[region],
            page_bbox=(0, 0, 600, 800),
        )

        paragraphs = page_layout.paragraphs()
        visual_only = next(
            paragraph for paragraph in paragraphs if paragraph.regions == [region]
        )

        self.assertEqual(visual_only.lines, [])
        self.assertEqual(region.paragraph_index, visual_only.paragraph_index)


if __name__ == "__main__":
    unittest.main()

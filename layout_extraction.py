import contextlib
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import fitz
import pytesseract
from pytesseract import Output


REGION_PROMPTS = {
    "image": "این بخش یک تصویر یا نمودار داخل سند است. محتوای قابل مشاهده را برای بازیابی متنی توصیف کن.",
    "table": "این بخش یک جدول داخل سند است. ساختار، عنوان‌ها و مقادیر مهم را خلاصه کن.",
    "formula": "این بخش یک فرمول یا رابطه ریاضی داخل سند است. نمادها و مفهوم کلی را توصیف کن.",
}

REGION_LABELS = {
    "image": "توضیح تصویر",
    "table": "توضیح جدول",
    "formula": "توضیح فرمول",
}


@dataclass
class VisualRegion:
    document_id: str
    page_number: int
    region_index: int
    region_type: str
    bbox: tuple[float, float, float, float]
    crop_path: str
    prompt: str
    paragraph_index: int | None = None
    caption: str = ""

    @property
    def label(self):
        return REGION_LABELS.get(self.region_type, "توضیح بخش تصویری")

    def inline_text(self):
        if not self.caption:
            return ""
        return f"[{self.label}: {self.caption}]"

    def to_record(self):
        return {
            "document_id": self.document_id,
            "page_number": self.page_number,
            "region_index": self.region_index,
            "region_type": self.region_type,
            "bbox": json.dumps(self.bbox),
            "crop_path": self.crop_path,
            "prompt": self.prompt,
            "paragraph_index": self.paragraph_index,
            "caption": self.caption,
        }


@dataclass
class LayoutSegment:
    kind: str
    bbox: tuple[float, float, float, float]
    text: str = ""
    region_index: int | None = None
    block_num: int | None = None
    par_num: int | None = None
    line_num: int | None = None
    paragraph_index: int | None = None

    @property
    def sort_key(self):
        x0, y0, _, _ = self.bbox
        return round(y0 / 8), y0, x0


@dataclass
class ParagraphLayout:
    page_number: int
    bbox: tuple[float, float, float, float]
    lines: list[LayoutSegment] = field(default_factory=list)
    regions: list[VisualRegion] = field(default_factory=list)
    paragraph_index: int = 0
    is_heading: bool = False
    heading_text: str = ""

    @property
    def sort_key(self):
        x0, y0, _, _ = self.bbox
        return round(y0 / 8), y0, x0

    @property
    def ocr_text(self):
        return "\n".join(line.text.strip() for line in self.lines if line.text.strip())

    @property
    def image_caption_text(self):
        return "\n".join(
            region.inline_text() for region in self.regions if region.caption.strip()
        )

    @property
    def region_indices(self):
        return [region.region_index for region in self.regions]

    def inline_text(self):
        region_by_index = {region.region_index: region for region in self.regions}
        segments = list(self.lines)
        segments.extend(
            LayoutSegment(
                kind="region",
                bbox=region.bbox,
                region_index=region.region_index,
            )
            for region in self.regions
        )

        parts = []
        for segment in sorted(segments, key=lambda seg: seg.sort_key):
            if segment.kind == "text" and segment.text.strip():
                parts.append(segment.text.strip())
            elif segment.kind == "region":
                region = region_by_index.get(segment.region_index)
                if region and region.inline_text():
                    parts.append(region.inline_text())
        return "\n".join(parts)


@dataclass
class PageLayout:
    page_number: int
    text_segments: list[LayoutSegment]
    region_segments: list[LayoutSegment]
    regions: list[VisualRegion]
    segments: list[LayoutSegment] = field(default_factory=list)
    page_bbox: tuple[float, float, float, float] | None = None

    @property
    def ocr_text(self):
        return "\n".join(seg.text for seg in self.text_segments if seg.text.strip())

    @property
    def image_caption_text(self):
        return "\n".join(
            region.inline_text() for region in self.regions if region.caption.strip()
        )

    def inline_text(self):
        parts = []
        for paragraph in self.paragraphs():
            inline_text = paragraph.inline_text()
            if inline_text.strip():
                parts.append(inline_text)
        return "\n\n".join(parts)

    def paragraphs(self):
        paragraphs = self._text_paragraphs()
        paragraphs.sort(key=lambda paragraph: paragraph.sort_key)
        self._annotate_headings(paragraphs)

        for region in self.regions:
            paragraph = self._best_related_paragraph(region, paragraphs)
            if paragraph is None:
                paragraph = ParagraphLayout(
                    page_number=self.page_number,
                    bbox=region.bbox,
                    regions=[region],
                )
                paragraphs.append(paragraph)
            else:
                paragraph.regions.append(region)
                paragraph.bbox = self._union_bbox([paragraph.bbox, region.bbox])

        paragraphs.sort(key=lambda paragraph: paragraph.sort_key)
        self._propagate_heading_text(paragraphs)
        for index, paragraph in enumerate(paragraphs, start=1):
            paragraph.paragraph_index = index
            for line in paragraph.lines:
                line.paragraph_index = index
            for region in paragraph.regions:
                region.paragraph_index = index

        return paragraphs

    def _text_paragraphs(self):
        grouped = {}
        fallback_index = 0
        for segment in sorted(self.text_segments, key=lambda seg: seg.sort_key):
            if not segment.text.strip():
                continue
            if segment.block_num is not None and segment.par_num is not None:
                key = ("ocr", segment.block_num, segment.par_num)
            else:
                key = ("fallback", fallback_index)
                fallback_index += 1
            grouped.setdefault(key, []).append(segment)

        paragraphs = []
        for lines in grouped.values():
            bbox = self._union_bbox([line.bbox for line in lines])
            paragraphs.append(
                ParagraphLayout(
                    page_number=self.page_number,
                    bbox=bbox,
                    lines=sorted(lines, key=lambda line: line.sort_key),
                )
            )
        return paragraphs

    def _annotate_headings(self, paragraphs):
        for index, paragraph in enumerate(paragraphs):
            previous_paragraph = paragraphs[index - 1] if index > 0 else None
            next_paragraph = (
                paragraphs[index + 1] if index + 1 < len(paragraphs) else None
            )
            paragraph.is_heading = self._is_heading_paragraph(
                paragraph,
                previous_paragraph,
                next_paragraph,
            )
        self._propagate_heading_text(paragraphs)

    def _propagate_heading_text(self, paragraphs):
        current_heading = ""
        for paragraph in sorted(paragraphs, key=lambda item: item.sort_key):
            if paragraph.is_heading:
                current_heading = self._clean_text(paragraph.ocr_text)
            paragraph.heading_text = current_heading

    def _is_heading_paragraph(self, paragraph, previous_paragraph, next_paragraph):
        text = self._clean_text(paragraph.ocr_text)
        if not text:
            return False
        if re.fullmatch(r"[\d۰-۹\s.\-_/]+", text):
            return False

        words = re.findall(r"[\w\u0600-\u06FF]+", text)
        if len(words) > 14 or len(text) > 120:
            return False
        if len(paragraph.lines) > 2:
            return False
        numbered_heading = bool(
            re.match(r"^\s*(?:\d+(?:\.\d+)*|[۰-۹]+(?:\.[۰-۹]+)*)\s+\S", text)
        )
        if not numbered_heading and self._looks_sentence_like(text, words):
            return False

        _, page_top, _, page_bottom = self.page_bbox or (0, 0, 0, 1000)
        page_height = max(page_bottom - page_top, 1)
        near_top = paragraph.bbox[1] <= page_top + page_height * 0.18

        previous_gap = 999
        if previous_paragraph is not None:
            previous_gap = max(0, paragraph.bbox[1] - previous_paragraph.bbox[3])
        next_gap = 0
        if next_paragraph is not None:
            next_gap = max(0, next_paragraph.bbox[1] - paragraph.bbox[3])

        visually_separated = previous_gap >= 10 or next_gap >= 8
        section_boundary = previous_paragraph is None or previous_gap >= 16

        return numbered_heading or (visually_separated and (near_top or section_boundary))

    def _looks_sentence_like(self, text, words):
        sentence_endings = ".؟?؛;:"
        if text[-1:] in sentence_endings:
            return True
        if len(words) >= 9 and re.search(r"[،,]", text):
            return True
        sentence_mark_count = len(re.findall(r"[.!؟?؛;]", text))
        return sentence_mark_count > 1

    def _clean_text(self, text):
        return re.sub(r"\s+", " ", text or "").strip()

    def _best_related_paragraph(self, region, paragraphs):
        overlap_candidates = []
        below_candidates = []
        for paragraph in paragraphs:
            if paragraph.is_heading:
                continue
            relationship = self._region_paragraph_relationship(
                region.bbox,
                paragraph.bbox,
            )
            if relationship is None:
                continue
            kind, score = relationship
            if kind == "overlap":
                overlap_candidates.append((score, paragraph))
            elif kind == "below":
                below_candidates.append((score, paragraph))

        if overlap_candidates:
            return min(overlap_candidates, key=lambda item: item[0])[1]
        if below_candidates:
            return min(below_candidates, key=lambda item: item[0])[1]
        return None

    def _region_paragraph_relationship(self, region_bbox, paragraph_bbox):
        vertical_overlap = self._overlap_length(
            region_bbox[1],
            region_bbox[3],
            paragraph_bbox[1],
            paragraph_bbox[3],
        )
        horizontal_overlap = self._overlap_length(
            region_bbox[0],
            region_bbox[2],
            paragraph_bbox[0],
            paragraph_bbox[2],
        )
        region_height = max(region_bbox[3] - region_bbox[1], 1)
        paragraph_height = max(paragraph_bbox[3] - paragraph_bbox[1], 1)
        region_width = max(region_bbox[2] - region_bbox[0], 1)
        paragraph_width = max(paragraph_bbox[2] - paragraph_bbox[0], 1)

        vertical_overlap_ratio = vertical_overlap / min(region_height, paragraph_height)
        horizontal_overlap_ratio = horizontal_overlap / min(region_width, paragraph_width)
        horizontal_gap = self._axis_gap(
            region_bbox[0],
            region_bbox[2],
            paragraph_bbox[0],
            paragraph_bbox[2],
        )
        same_column = horizontal_overlap_ratio >= 0.25 or horizontal_gap <= 18

        if not same_column:
            return None

        if vertical_overlap_ratio >= 0.2:
            distance = self._center_distance(region_bbox, paragraph_bbox)
            return ("overlap", distance - (vertical_overlap_ratio * 100))

        if region_bbox[1] < paragraph_bbox[3]:
            return None

        vertical_gap = region_bbox[1] - paragraph_bbox[3]
        max_near_gap = max(36, min(region_height * 0.35, paragraph_height * 1.5))
        if vertical_gap <= max_near_gap:
            return ("below", self._center_distance(region_bbox, paragraph_bbox) + vertical_gap)

        return None

    def _center_distance(self, first, second):
        first_x = (first[0] + first[2]) / 2
        first_y = (first[1] + first[3]) / 2
        second_x = (second[0] + second[2]) / 2
        second_y = (second[1] + second[3]) / 2
        return ((first_x - second_x) ** 2 + (first_y - second_y) ** 2) ** 0.5

    def _overlap_length(self, first_start, first_end, second_start, second_end):
        return max(0, min(first_end, second_end) - max(first_start, second_start))

    def _axis_gap(self, first_start, first_end, second_start, second_end):
        if first_end < second_start:
            return second_start - first_end
        if second_end < first_start:
            return first_start - second_end
        return 0

    def _union_bbox(self, bboxes):
        return (
            min(bbox[0] for bbox in bboxes),
            min(bbox[1] for bbox in bboxes),
            max(bbox[2] for bbox in bboxes),
            max(bbox[3] for bbox in bboxes),
        )


class PDFLayoutExtractor:
    def __init__(
        self,
        pdf_path,
        document_id,
        region_root="app_data/regions",
        crop_zoom=2.5,
    ):
        self.pdf_path = str(pdf_path)
        self.document_id = document_id
        self.region_root = Path(region_root)
        self.crop_zoom = crop_zoom
        self.doc = fitz.open(self.pdf_path)

    def close(self):
        self.doc.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    @property
    def page_count(self):
        return self.doc.page_count

    def extract_page(self, page_number, page_image_path):
        page = self.doc[page_number - 1]
        raw_regions = self._detect_regions(page)
        regions = []
        region_segments = []

        for region_index, (region_type, rect) in enumerate(raw_regions):
            crop_path = self._save_region_crop(page, page_number, region_index, rect)
            region = VisualRegion(
                document_id=self.document_id,
                page_number=page_number,
                region_index=region_index,
                region_type=region_type,
                bbox=tuple(round(value, 2) for value in rect),
                crop_path=str(crop_path),
                prompt=REGION_PROMPTS[region_type],
            )
            regions.append(region)
            region_segments.append(
                LayoutSegment(
                    kind="region",
                    bbox=region.bbox,
                    region_index=region.region_index,
                )
            )

        text_segments = self._ocr_line_segments(page, page_image_path)
        segments = text_segments + region_segments
        return PageLayout(
            page_number=page_number,
            text_segments=text_segments,
            region_segments=region_segments,
            regions=regions,
            segments=segments,
            page_bbox=tuple(page.rect),
        )

    def _detect_regions(self, page):
        candidates = []
        page_rect = page.rect

        candidates.extend(self._table_regions(page, page_rect))
        candidates.extend(self._formula_regions(page, page_rect))
        candidates.extend(self._image_regions(page, page_rect))

        priority = {"table": 0, "formula": 1, "image": 2}
        candidates.sort(key=lambda item: (priority[item[0]], item[1].y0, item[1].x0))

        accepted = []
        for region_type, rect in candidates:
            if any(self._overlap_ratio(rect, other_rect) > 0.65 for _, other_rect in accepted):
                continue
            accepted.append((region_type, rect))

        accepted.sort(key=lambda item: (round(item[1].y0 / 8), item[1].y0, item[1].x0))
        return accepted

    def _image_regions(self, page, page_rect):
        regions = []
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 1:
                continue
            rect = fitz.Rect(block["bbox"])
            if self._is_meaningful_rect(rect, page_rect, min_area_ratio=0.004):
                regions.append(("image", rect))
        return regions

    def _table_regions(self, page, page_rect):
        regions = []
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                tables = page.find_tables().tables
        except Exception:
            return regions

        for table in tables:
            rect = fitz.Rect(table.bbox)
            if self._is_meaningful_rect(
                rect,
                page_rect,
                min_width=70,
                min_height=28,
                min_area_ratio=0.004,
                max_area_ratio=0.45,
            ):
                regions.append(("table", rect))
        return regions

    def _formula_regions(self, page, page_rect):
        regions = []
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                text = "".join(
                    span.get("text", "") for span in line.get("spans", [])
                ).strip()
                if not self._is_formula_like(text):
                    continue
                rect = fitz.Rect(line["bbox"])
                if self._is_meaningful_rect(
                    rect,
                    page_rect,
                    min_width=35,
                    min_height=8,
                    min_area_ratio=0.0003,
                    max_area_ratio=0.12,
                ):
                    regions.append(("formula", rect))
        return regions

    def _ocr_line_segments(self, page, image_path):
        image = cv2.imread(str(image_path))
        if image is None:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        data = pytesseract.image_to_data(
            gray,
            lang="fas+eng",
            output_type=Output.DICT,
        )
        grouped = {}
        for index, text in enumerate(data.get("text", [])):
            text = text.strip()
            if not text:
                continue
            key = (
                data["block_num"][index],
                data["par_num"][index],
                data["line_num"][index],
            )
            grouped.setdefault(key, []).append(index)

        image_height, image_width = image.shape[:2]
        scale_x = page.rect.width / image_width
        scale_y = page.rect.height / image_height
        segments = []

        for indices in grouped.values():
            words = [data["text"][idx].strip() for idx in indices if data["text"][idx].strip()]
            if not words:
                continue

            x0 = min(data["left"][idx] for idx in indices) * scale_x
            y0 = min(data["top"][idx] for idx in indices) * scale_y
            x1 = max(data["left"][idx] + data["width"][idx] for idx in indices) * scale_x
            y1 = max(data["top"][idx] + data["height"][idx] for idx in indices) * scale_y
            segments.append(
                LayoutSegment(
                    kind="text",
                    bbox=(x0, y0, x1, y1),
                    text=" ".join(words),
                    block_num=data["block_num"][indices[0]],
                    par_num=data["par_num"][indices[0]],
                    line_num=data["line_num"][indices[0]],
                )
            )

        if not segments:
            text = pytesseract.image_to_string(gray, lang="fas+eng").strip()
            if text:
                segments.append(
                    LayoutSegment(
                        kind="text",
                        bbox=tuple(page.rect),
                        text=text,
                    )
                )

        return segments

    def _save_region_crop(self, page, page_number, region_index, rect):
        page_dir = self.region_root / self.document_id
        page_dir.mkdir(parents=True, exist_ok=True)
        padded = fitz.Rect(rect)
        padded.x0 = max(page.rect.x0, padded.x0 - 4)
        padded.y0 = max(page.rect.y0, padded.y0 - 4)
        padded.x1 = min(page.rect.x1, padded.x1 + 4)
        padded.y1 = min(page.rect.y1, padded.y1 + 4)

        target = page_dir / f"page_{page_number:04d}_region_{region_index:04d}.png"
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(self.crop_zoom, self.crop_zoom),
            clip=padded,
            alpha=False,
        )
        pixmap.save(str(target))
        return target

    def _is_meaningful_rect(
        self,
        rect,
        page_rect,
        min_width=45,
        min_height=24,
        min_area_ratio=0.002,
        max_area_ratio=0.45,
    ):
        if rect.width < min_width or rect.height < min_height:
            return False
        area_ratio = rect.get_area() / page_rect.get_area()
        return min_area_ratio <= area_ratio <= max_area_ratio

    def _overlap_ratio(self, first, second):
        intersection = first & second
        if intersection.is_empty:
            return 0.0
        return intersection.get_area() / min(first.get_area(), second.get_area())

    def _is_formula_like(self, text):
        if len(text) < 3:
            return False
        if re.search(r"doi|download|http|www", text, flags=re.IGNORECASE):
            return False

        strong_symbols = set("=×÷^√∑∫≤≥≈≠±→←∆")
        weak_symbols = set("+-*/")
        strong_count = sum(1 for char in text if char in strong_symbols)
        weak_count = sum(1 for char in text if char in weak_symbols)
        has_digit = bool(re.search(r"\d|[۰-۹]", text))
        has_latin = bool(re.search(r"[A-Za-z]", text))

        if strong_count >= 1 and (has_digit or has_latin):
            return True
        if weak_count >= 2 and has_digit and has_latin:
            return True
        if re.search(r"[A-Za-z]\s*[=<>]\s*[\w\d۰-۹]", text):
            return True
        return False

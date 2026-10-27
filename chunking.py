import re


class PersianChunker:
    def __init__(self, max_len=300):
        self.max_len = max_len

    def chunk(self, text):
        sentences = re.split(r'[.!؟]', text)

        chunks = []
        current = ""

        for s in sentences:
            if len(current) + len(s) < self.max_len:
                current += s + " "
            else:
                chunks.append(current.strip())
                current = s

        if current:
            chunks.append(current.strip())

        return chunks


class ContextualScientificChunker:
    """Build layout-aware BM25 chunks around paragraph anchors."""

    def __init__(
        self,
        target_min=700,
        target_max=1000,
        hard_max=1400,
        min_useful=250,
    ):
        self.target_min = target_min
        self.target_max = target_max
        self.hard_max = hard_max
        self.min_useful = min_useful

    def build_page_chunks(self, paragraphs, file_name):
        ordered = sorted(
            paragraphs,
            key=lambda paragraph: (
                getattr(paragraph, "page_number", 0),
                getattr(paragraph, "paragraph_index", 0) or 0,
                getattr(paragraph, "sort_key", (0, 0, 0)),
            ),
        )
        chunks = []

        for index, paragraph in enumerate(ordered):
            if self._is_heading(paragraph) and not self._caption_lines(paragraph):
                continue
            if not self._has_text(paragraph):
                continue

            if self._is_visual_only(paragraph):
                chunks.append(
                    self._make_chunk(
                        file_name=file_name,
                        anchor=paragraph,
                        paragraphs=[paragraph],
                    )
                )
                continue

            anchor_text = self._paragraph_text(paragraph)
            if len(anchor_text) > self.hard_max:
                for text_part in self._split_long_text(anchor_text):
                    chunks.append(
                        self._make_chunk(
                            file_name=file_name,
                            anchor=paragraph,
                            paragraphs=[paragraph],
                            anchor_text_override=text_part,
                        )
                    )
                continue

            context_paragraphs = self._context_window(ordered, index)
            chunks.append(
                self._make_chunk(
                    file_name=file_name,
                    anchor=paragraph,
                    paragraphs=context_paragraphs,
                )
            )

        return [chunk for chunk in chunks if chunk["retrieval_text"].strip()]

    def _context_window(self, ordered, anchor_index):
        anchor = ordered[anchor_index]
        selected = {anchor_index}

        if self._content_length([anchor]) >= self.target_min:
            return [anchor]

        for neighbor_index in (anchor_index - 1, anchor_index + 1):
            if neighbor_index < 0 or neighbor_index >= len(ordered):
                continue

            neighbor = ordered[neighbor_index]
            if self._is_heading(neighbor):
                continue
            if self._is_visual_only(neighbor):
                continue
            if not self._same_context(anchor, neighbor):
                continue

            candidate_indices = selected | {neighbor_index}
            candidate = [ordered[index] for index in sorted(candidate_indices)]
            if self._content_length(candidate) <= self.hard_max:
                selected = candidate_indices

        return [ordered[index] for index in sorted(selected)]

    def _make_chunk(
        self,
        file_name,
        anchor,
        paragraphs,
        anchor_text_override=None,
    ):
        heading_text = self._heading_text(anchor)
        page_number = getattr(anchor, "page_number", None)
        paragraph_span = self._paragraph_span(paragraphs)
        ocr_text = self._ocr_text(paragraphs, anchor, anchor_text_override)
        caption_text = self._caption_text(paragraphs)
        display_text = self._display_text(
            heading_text,
            paragraphs,
            anchor,
            anchor_text_override,
        )
        retrieval_text = self._retrieval_text(
            file_name=file_name,
            page_number=page_number,
            paragraph_span=paragraph_span,
            heading_text=heading_text,
            ocr_text=ocr_text,
            caption_text=caption_text,
        )

        return {
            "page_number": page_number,
            "paragraph_index": getattr(anchor, "paragraph_index", None),
            "paragraph_span": paragraph_span,
            "heading_text": heading_text,
            "display_text": display_text,
            "retrieval_text": retrieval_text,
            "ocr_text": ocr_text,
            "image_caption": caption_text,
            "combined_text": retrieval_text,
            "region_indices": self._region_indices(paragraphs),
        }

    def _display_text(
        self,
        heading_text,
        paragraphs,
        anchor,
        anchor_text_override=None,
    ):
        parts = []
        if heading_text:
            parts.append(heading_text)

        for paragraph in paragraphs:
            if paragraph is anchor and anchor_text_override is not None:
                paragraph_text = anchor_text_override
            else:
                paragraph_text = self._paragraph_text(paragraph)
            if paragraph_text:
                parts.append(paragraph_text)
            parts.extend(self._caption_lines(paragraph))

        return self._normalize_blocks(parts)

    def _retrieval_text(
        self,
        file_name,
        page_number,
        paragraph_span,
        heading_text,
        ocr_text,
        caption_text,
    ):
        parts = [
            f"File / سند: {file_name}",
            f"Page / صفحه: {page_number}",
        ]
        if paragraph_span:
            parts.append(f"Paragraph / پاراگراف: {paragraph_span}")
        if heading_text:
            parts.append(f"Heading / عنوان: {heading_text}")
        if ocr_text:
            parts.append(f"OCR text / متن: {ocr_text}")
        if caption_text:
            parts.append(f"Visual caption / توضیح بصری: {caption_text}")
        return self._normalize_blocks(parts)

    def _split_long_text(self, text):
        sentences = self._split_sentences(text)
        chunks = []
        current = ""

        for sentence in sentences:
            if not sentence:
                continue
            if len(sentence) > self.target_max:
                if current:
                    chunks.append(current.strip())
                    current = ""
                chunks.extend(self._split_oversized_sentence(sentence))
                continue
            candidate = f"{current} {sentence}".strip()
            if current and len(candidate) > self.target_max:
                chunks.append(current.strip())
                current = sentence
            else:
                current = candidate

        if current:
            chunks.append(current.strip())
        return chunks or [text]

    def _split_sentences(self, text):
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!؟?؛;])\s+|\n+", text)
            if sentence.strip()
        ]

    def _split_oversized_sentence(self, sentence):
        words = sentence.split()
        chunks = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if current and len(candidate) > self.target_max:
                chunks.append(current)
                current = word
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks

    def _same_context(self, first, second):
        return (
            getattr(first, "page_number", None) == getattr(second, "page_number", None)
            and self._heading_text(first) == self._heading_text(second)
        )

    def _content_length(self, paragraphs):
        return len(self._display_text("", paragraphs, anchor=None))

    def _ocr_text(self, paragraphs, anchor, anchor_text_override=None):
        parts = []
        for paragraph in paragraphs:
            if paragraph is anchor and anchor_text_override is not None:
                text = anchor_text_override
            else:
                text = self._paragraph_text(paragraph)
            if text:
                parts.append(text)
        return self._normalize_blocks(parts)

    def _caption_text(self, paragraphs):
        return self._normalize_blocks(
            line for paragraph in paragraphs for line in self._caption_lines(paragraph)
        )

    def _caption_lines(self, paragraph):
        lines = []
        for region in getattr(paragraph, "regions", []) or []:
            inline_text = ""
            if hasattr(region, "inline_text"):
                inline_text = region.inline_text()
            elif getattr(region, "caption", ""):
                label = getattr(region, "label", "توضیح بخش تصویری")
                inline_text = f"[{label}: {region.caption}]"
            inline_text = self._normalize_text(inline_text)
            if inline_text:
                lines.append(inline_text)
        return lines

    def _region_indices(self, paragraphs):
        indices = []
        for paragraph in paragraphs:
            for region in getattr(paragraph, "regions", []) or []:
                region_index = getattr(region, "region_index", None)
                if region_index is not None:
                    indices.append(region_index)
        return sorted(set(indices))

    def _paragraph_span(self, paragraphs):
        indices = sorted(
            {
                getattr(paragraph, "paragraph_index", None)
                for paragraph in paragraphs
                if getattr(paragraph, "paragraph_index", None) is not None
            }
        )
        if not indices:
            return None
        if len(indices) == 1:
            return str(indices[0])
        return f"{indices[0]}-{indices[-1]}"

    def _has_text(self, paragraph):
        return bool(self._paragraph_text(paragraph) or self._caption_lines(paragraph))

    def _is_visual_only(self, paragraph):
        return not self._paragraph_text(paragraph) and bool(
            getattr(paragraph, "regions", []) or []
        )

    def _is_heading(self, paragraph):
        return bool(getattr(paragraph, "is_heading", False))

    def _heading_text(self, paragraph):
        return self._normalize_text(getattr(paragraph, "heading_text", "") or "")

    def _paragraph_text(self, paragraph):
        return self._normalize_text(getattr(paragraph, "ocr_text", "") or "")

    def _normalize_blocks(self, parts):
        return "\n\n".join(
            part for part in (self._normalize_text(part) for part in parts) if part
        )

    def _normalize_text(self, text):
        return re.sub(r"[ \t\r\f\v]+", " ", str(text or "")).strip()

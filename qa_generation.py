import re
import logging

from vgen_runtime import get_vgen_runtime
from config import VGEN_MODEL_PATH


class FarsiQAGenerator:
    def __init__(
        self,
        runtime=None,
        model_path=VGEN_MODEL_PATH,
        max_new_tokens=256,
        max_chunks=5,
        max_chunk_chars=900,
    ):
        self.runtime = runtime or get_vgen_runtime(model_path)
        self._uses_default_runtime = runtime is None
        self.max_new_tokens = max_new_tokens
        self.max_chunks = max_chunks
        self.max_chunk_chars = max_chunk_chars
        self.last_error = None

    def generate_answer(self, question, evidence):
        self.last_error = None
        useful_evidence = self._useful_evidence(evidence)
        if not useful_evidence:
            return ""

        try:
            return self._generate_answer(question, useful_evidence)
        except Exception as exc:
            self.last_error = str(exc)
            logging.getLogger(__name__).exception("Answer generation failed")
            return ""

    def _generate_answer(
        self,
        question,
        evidence,
    ):
        if self._uses_default_runtime:
            from runtime_lifecycle import activate_vgen_runtime
            activate_vgen_runtime(self.runtime)
        messages = self.build_messages(question, evidence)
        answer = self.runtime.generate(
            messages,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        return self._clean_answer(answer)

    def build_messages(self, question, evidence):
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "تو یک دستیار پرسش‌وپاسخ علمی فارسی هستی. "
                            "فقط بر اساس شواهد بازیابی‌شده پاسخ بده."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": self.build_prompt(question, evidence)}],
            },
        ]

    def build_prompt(self, question, evidence):
        evidence_blocks = []
        for index, item in enumerate(evidence[: self.max_chunks], start=1):
            source = self._format_source(item)
            text = self._chunk_text(item)
            evidence_blocks.append(f"[{index}] منبع: {source}\nمتن: {text}")

        return (
            "تو یک دستیار پرسش‌وپاسخ علمی فارسی هستی.\n"
            "فقط بر اساس متن‌های بازیابی‌شده پاسخ بده.\n"
            "اگر پاسخ دقیق در متن‌ها وجود ندارد، صریح بگو که بر اساس شواهد موجود پاسخ قطعی پیدا نشد.\n"
            "از دانش بیرونی، حدس، یا اطلاعاتی که در متن نیست استفاده نکن.\n"
            "پاسخ را روان، علمی، کوتاه و فارسی بنویس.\n"
            "اصطلاحات تخصصی انگلیسی، اعداد، واحدها، نام ژن‌ها، نام داروها، فرمول‌ها و اختصارها را همان‌طور که در متن آمده‌اند حفظ کن.\n"
            "اگر شواهد متناقض یا ناکافی هستند، همین را توضیح بده.\n"
            "در پایان، منابع استفاده‌شده را با شماره شواهد ذکر کن.\n\n"
            f"پرسش:\n{normalize_whitespace(question)}\n\n"
            "شواهد بازیابی‌شده:\n"
            f"{chr(10).join(evidence_blocks)}\n\n"
            "پاسخ فارسی:"
        )

    def _useful_evidence(self, evidence):
        return [
            item
            for item in evidence
            if (
                item.get("match_count", 0) > 0
                or item.get("retrieval_mode") == "simsearch"
                or item.get("semantic_score") is not None
            )
            and (item.get("display_text") or item.get("combined_text"))
        ][: self.max_chunks]

    def _format_source(self, item):
        source = f'{item["file_name"]}، صفحه {item["page_number"]}'
        paragraph_span = item.get("paragraph_span") or item.get("paragraph_index")
        if paragraph_span:
            source += f"، پاراگراف {paragraph_span}"
        return source

    def _chunk_text(self, item):
        text = item.get("display_text") or item.get("combined_text") or ""
        return truncate_text(normalize_whitespace(text), self.max_chunk_chars)

    def _clean_answer(self, answer):
        answer = re.sub(r"\s+", " ", str(answer or "")).strip()
        for marker in ("پاسخ فارسی:", "پاسخ:"):
            if marker in answer:
                answer = answer.split(marker)[-1].strip()
        return answer.strip(" \n\t:：-")


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def truncate_text(text, max_chars):
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"

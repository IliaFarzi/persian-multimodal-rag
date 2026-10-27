import re
import logging

from PIL import Image

from vgen_runtime import get_vgen_runtime
from config import VGEN_MODEL_PATH


class FarsiCaptioner:
    def __init__(self, runtime=None, model_path=VGEN_MODEL_PATH, max_new_tokens=80):
        self.runtime = runtime or get_vgen_runtime(model_path)
        self._uses_default_runtime = runtime is None
        self.max_new_tokens = max_new_tokens
        self.last_error = None

    def generate_caption(self, image_path, region_type=None, prompt=None):
        self.last_error = None
        try:
            return self._generate_caption(image_path, region_type, prompt)
        except Exception as exc:
            self.last_error = str(exc)
            logging.getLogger(__name__).exception("Caption generation failed for %s", image_path)
            return ""

    def _generate_caption(
        self,
        image_path,
        region_type=None,
        prompt=None,
    ):
        if self._uses_default_runtime:
            from runtime_lifecycle import activate_vgen_runtime
            activate_vgen_runtime(self.runtime)
        prepare = getattr(self.runtime, "prepare", None)
        if prepare is not None:
            prepare()
        image = self._load_image(image_path)
        messages = self._build_messages(image, region_type, prompt)
        caption = self.runtime.generate(
            messages,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        return self._clean_caption(caption)

    def _load_image(self, image_path):
        with Image.open(image_path) as image:
            return image.convert("RGB").copy()

    def _build_messages(self, image, region_type=None, prompt=None):
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "تو یک مدل بینایی-زبانی برای شرح کوتاه بخش‌های "
                            "سند علمی هستی. فقط به فارسی پاسخ بده."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": self._caption_prompt(region_type, prompt),
                    },
                ],
            },
        ]

    def _caption_prompt(self, region_type=None, prompt=None):
        type_instructions = {
            "image": "این بخش یک تصویر یا نمودار است. محتوای قابل مشاهده را کوتاه توصیف کن.",
            "table": "این بخش یک جدول است. ساختار، برچسب‌ها و مقادیر مهم را کوتاه خلاصه کن.",
            "formula": "این بخش یک فرمول است. نمادها و مفهوم کلی رابطه را کوتاه توضیح بده.",
        }
        instruction = type_instructions.get(
            region_type,
            "این بخش تصویری از یک سند علمی است. محتوای مهم آن را کوتاه توصیف کن.",
        )
        if prompt:
            instruction = f"{instruction}\nراهنمای تکمیلی: {prompt}"
        return (
            f"{instruction}\n"
            "خروجی فقط یک یا دو جمله کوتاه فارسی باشد. اعداد، برچسب‌ها و اصطلاحات "
            "انگلیسی مهم را همان‌طور که دیده می‌شوند حفظ کن."
        )

    def _clean_caption(self, caption):
        caption = re.sub(r"\s+", " ", str(caption or "")).strip()
        return caption.strip(" \n\t:：-")

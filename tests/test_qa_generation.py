import unittest

from captioning import FarsiCaptioner
from qa_generation import FarsiQAGenerator
from vgen_runtime import reset_vgen_runtime


class FakeRuntime:
    def __init__(self, answer="پاسخ فارسی: نتیجه نهایی"):
        self.answer = answer
        self.calls = []
        self.dtype = None

    def generate(self, messages, max_new_tokens, do_sample=False, dtype_candidates=()):
        self.calls.append(
            {
                "messages": messages,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "dtype_candidates": dtype_candidates,
            }
        )
        return self.answer

    def reset_model(self):
        self.dtype = None


class FarsiQAGeneratorTest(unittest.TestCase):
    def tearDown(self):
        reset_vgen_runtime()

    def test_captioner_and_qa_generator_share_default_runtime(self):
        reset_vgen_runtime()

        captioner = FarsiCaptioner()
        qa_generator = FarsiQAGenerator()

        self.assertIs(captioner.runtime, qa_generator.runtime)

    def test_prompt_contains_grounding_rules_and_numbered_evidence(self):
        generator = FarsiQAGenerator(runtime=FakeRuntime())
        prompt = generator.build_prompt(
            "اثر دارو روی AUC چیست؟",
            [
                {
                    "file_name": "paper.pdf",
                    "page_number": 2,
                    "paragraph_span": "7",
                    "match_count": 2,
                    "display_text": "AUC پس از درمان به 0.91 رسید.",
                }
            ],
        )

        self.assertIn("فقط بر اساس متن‌های بازیابی‌شده پاسخ بده", prompt)
        self.assertIn("از دانش بیرونی، حدس", prompt)
        self.assertIn("اثر دارو روی AUC چیست؟", prompt)
        self.assertIn("[1] منبع: paper.pdf، صفحه 2، پاراگراف 7", prompt)
        self.assertIn("AUC پس از درمان به 0.91 رسید", prompt)
        self.assertIn("پاسخ فارسی:", prompt)

    def test_generate_answer_uses_runtime_and_cleans_prompt_echo(self):
        runtime = FakeRuntime(answer="پاسخ فارسی: نتیجه نهایی بر اساس [1].")
        generator = FarsiQAGenerator(runtime=runtime, max_new_tokens=123)

        answer = generator.generate_answer(
            "نتیجه چیست؟",
            [
                {
                    "file_name": "paper.pdf",
                    "page_number": 1,
                    "paragraph_index": 3,
                    "match_count": 1,
                    "display_text": "نتیجه اصلی افزایش دقت بود.",
                }
            ],
        )

        self.assertEqual(answer, "نتیجه نهایی بر اساس [1].")
        self.assertEqual(runtime.calls[0]["max_new_tokens"], 123)
        self.assertFalse(runtime.calls[0]["do_sample"])
        self.assertEqual(runtime.calls[0]["messages"][1]["role"], "user")

    def test_generate_answer_returns_empty_without_matching_evidence(self):
        runtime = FakeRuntime()
        generator = FarsiQAGenerator(runtime=runtime)

        answer = generator.generate_answer(
            "نتیجه چیست؟",
            [
                {
                    "file_name": "paper.pdf",
                    "page_number": 1,
                    "match_count": 0,
                    "display_text": "متن نامرتبط.",
                }
            ],
        )

        self.assertEqual(answer, "")
        self.assertEqual(runtime.calls, [])


if __name__ == "__main__":
    unittest.main()

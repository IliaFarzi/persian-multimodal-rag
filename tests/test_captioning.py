import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from PIL import Image

from captioning import FarsiCaptioner
from vgen_runtime import VGenRuntime, reset_vgen_runtime
from config import VGEN_MODEL_PATH


class GemmaCaptionerTest(unittest.TestCase):
    def setUp(self):
        reset_vgen_runtime()
        self.addCleanup(reset_vgen_runtime)
        cuda_available_patch = patch(
            "vgen_runtime.torch.cuda.is_available",
            return_value=True,
        )
        bf16_supported_patch = patch(
            "vgen_runtime.torch.cuda.is_bf16_supported",
            return_value=True,
        )
        mps_available_patch = patch(
            "vgen_runtime.torch.backends.mps.is_available",
            return_value=False,
        )
        capability_patch = patch(
            "vgen_runtime.torch.cuda.get_device_capability",
            return_value=(8, 0),
        )
        cuda_available_patch.start()
        bf16_supported_patch.start()
        mps_available_patch.start()
        capability_patch.start()
        self.addCleanup(cuda_available_patch.stop)
        self.addCleanup(bf16_supported_patch.stop)
        self.addCleanup(mps_available_patch.stop)
        self.addCleanup(capability_patch.stop)

    def test_uses_local_vgen_model_and_region_prompt(self):
        processor = self._processor(" :  یک   جدول آزمایش است. - ")
        model = self._model(torch.tensor([[1, 2, 3, 4]]))

        _, model_class = self._patch_gemma(processor, model)
        captioner = FarsiCaptioner()
        caption = captioner.generate_caption(
            self._image_path(),
            region_type="table",
            prompt="برچسب‌های جدول را بخوان.",
        )

        self.assertEqual(caption, "یک جدول آزمایش است.")
        model_class.from_pretrained.assert_called_once()
        self.assertEqual(
            model_class.from_pretrained.call_args.kwargs["local_files_only"],
            True,
        )
        self.assertEqual(model_class.from_pretrained.call_args.args[0], str(VGEN_MODEL_PATH))

        messages = processor.apply_chat_template.call_args.args[0]
        prompt_text = messages[1]["content"][1]["text"]
        self.assertIn("این بخش یک جدول است", prompt_text)
        self.assertIn("برچسب‌های جدول را بخوان", prompt_text)

    def test_loads_fp4_quantized_model_by_default(self):
        processor = self._processor("یک تصویر علمی.")
        model = self._model(torch.tensor([[1, 2, 3]]))

        _, model_class = self._patch_gemma(processor, model)
        captioner = FarsiCaptioner()
        caption = captioner.generate_caption(self._image_path(), region_type="image")

        self.assertEqual(caption, "یک تصویر علمی.")
        kwargs = model_class.from_pretrained.call_args.kwargs
        quantization_config = kwargs["quantization_config"]
        self.assertTrue(quantization_config.load_in_4bit)
        self.assertEqual(quantization_config.bnb_4bit_quant_type, "fp4")
        self.assertEqual(quantization_config.bnb_4bit_compute_dtype, torch.bfloat16)
        self.assertFalse(quantization_config.llm_int8_enable_fp32_cpu_offload)
        self.assertEqual(kwargs["device_map"], {"": "cuda:0"})
        self.assertTrue(kwargs["low_cpu_mem_usage"])
        self.assertEqual(
            kwargs["attn_implementation"],
            {"text_config": "sdpa", "vision_config": "sdpa"},
        )
        self.assertNotIn("dtype", kwargs)
        self.assertNotIn("torch_dtype", kwargs)

    def test_does_not_retry_full_precision_when_quantized_load_fails(self):
        processor = self._processor("unused")
        model = self._model(torch.tensor([[1, 2, 3]]))

        _, model_class = self._patch_gemma(processor, model)
        model_class.from_pretrained.side_effect = RuntimeError("quantization failed")
        captioner = FarsiCaptioner()
        caption = captioner.generate_caption(self._image_path(), region_type="image")

        self.assertEqual(caption, "")
        model_class.from_pretrained.assert_called_once()

    def test_refuses_cpu_vgen_load(self):
        processor = self._processor("unused")
        model = self._model(torch.tensor([[1, 2, 3]]))

        with patch("vgen_runtime.torch.cuda.is_available", return_value=False), patch(
            "vgen_runtime.torch.backends.mps.is_available", return_value=False
        ):
            processor_class, model_class = self._patch_gemma(processor, model)
            captioner = FarsiCaptioner()
            caption = captioner.generate_caption(self._image_path(), region_type="image")

        self.assertEqual(caption, "")
        processor_class.from_pretrained.assert_not_called()
        model_class.from_pretrained.assert_not_called()

    def test_uses_float16_compute_when_cuda_bf16_is_not_supported(self):
        processor = self._processor("یک تصویر علمی.")
        model = self._model(torch.tensor([[1, 2, 3]]))

        with patch("vgen_runtime.torch.cuda.get_device_capability", return_value=(7, 5)):
            _, model_class = self._patch_gemma(processor, model)
            runtime = VGenRuntime()
            caption = runtime.generate(
                self._messages(),
                max_new_tokens=10,
            )

        self.assertEqual(caption, "یک تصویر علمی.")
        model.to.assert_not_called()
        kwargs = model_class.from_pretrained.call_args.kwargs
        self.assertEqual(kwargs["device_map"], {"": "cuda:0"})
        self.assertEqual(
            kwargs["quantization_config"].bnb_4bit_compute_dtype,
            torch.float16,
        )
        model.model.vision_tower.to.assert_called_once_with(dtype=torch.float16)
        model.model.multi_modal_projector.to.assert_called_once_with(
            dtype=torch.float16
        )

    def test_generation_failure_returns_empty_caption(self):
        processor = self._processor("unused")
        model = self._model(torch.tensor([[1, 2, 3]]))
        model.generate.side_effect = RuntimeError("generation failed")

        self._patch_gemma(processor, model)
        captioner = FarsiCaptioner()
        caption = captioner.generate_caption(self._image_path(), region_type="formula")

        self.assertEqual(caption, "")

    def _processor(self, decoded_text):
        processor = MagicMock()
        processor.apply_chat_template.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "pixel_values": torch.zeros((1, 3, 8, 8)),
        }
        processor.decode.return_value = decoded_text
        return processor

    def _model(self, generated):
        model = MagicMock()
        model.device = torch.device("cpu")
        model.to.return_value = model
        model.generate.return_value = generated
        return model

    def _patch_gemma(self, processor, model):
        processor_patch = patch("vgen_runtime.AutoProcessor")
        model_patch = patch("vgen_runtime.Gemma3ForConditionalGeneration")
        processor_class = processor_patch.start()
        model_class = model_patch.start()
        self.addCleanup(processor_patch.stop)
        self.addCleanup(model_patch.stop)
        processor_class.from_pretrained.return_value = processor
        model_class.from_pretrained.return_value = model
        return processor_class, model_class

    def _image_path(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "crop.png"
        Image.new("RGB", (8, 8), color="white").save(path)
        return path

    def _messages(self):
        return [
            {"role": "system", "content": [{"type": "text", "text": "system"}]},
            {"role": "user", "content": [{"type": "text", "text": "user"}]},
        ]


if __name__ == "__main__":
    unittest.main()

import threading
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration

from device_memory import release_device_memory
from config import VGEN_MODEL_PATH


_VGEN_RUNTIME = None
_VGEN_RUNTIME_LOCK = threading.Lock()


class VGenRuntime:
    def __init__(self, model_path=VGEN_MODEL_PATH, quantization="fp4"):
        self.model_path = str(Path(model_path))
        self.quantization = quantization
        self.processor = None
        self.model = None
        self.dtype = None
        self.device = self._select_device()
        self._lock = threading.RLock()

    def generate(
        self,
        messages,
        max_new_tokens,
        do_sample=False,
        dtype_candidates=None,
    ):
        with self._lock:
            self._ensure_model()
            cpu_inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = self._move_inputs_to_model(cpu_inputs)
            del cpu_inputs
            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode(), self._attention_context():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    use_cache=True,
                )

            generated = generated[0][input_len:]
            decoded = self._decode(generated)
            del generated, inputs
            return decoded

    def prepare(self):
        with self._lock:
            self._ensure_model()

    def reset_model(self):
        with self._lock:
            self._reset_model()

    def _ensure_model(self):
        if self.model is not None and self.processor is not None:
            return

        try:
            if self.quantization == "fp4":
                self._require_cuda_for_quantized_model()
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                local_files_only=True,
            )
            self.dtype = self._select_compute_dtype()
            self.model = self._load_model()
            self.model.eval()
        except Exception as exc:
            self._reset_model()
            raise RuntimeError(f"Could not load local Gemma 3 VGen model at {self.model_path}: {exc}") from exc

    def _load_model(self):
        if self.quantization == "fp4":
            return self._load_fp4_model()
        if self.quantization in ("none", None):
            return self._load_full_precision_model(self.dtype)
        raise ValueError(f"Unsupported VGen quantization: {self.quantization}")

    def _load_fp4_model(self):
        self._require_cuda_for_quantized_model()
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_compute_dtype=self.dtype,
            llm_int8_enable_fp32_cpu_offload=False,
        )
        model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_path,
            local_files_only=True,
            quantization_config=quantization_config,
            device_map=self._device_map(),
            low_cpu_mem_usage=True,
            attn_implementation={
                "text_config": "sdpa",
                "vision_config": "sdpa",
            },
        )
        self._prepare_vision_modules(model)
        self._assert_model_not_offloaded(model)
        return model

    def _load_full_precision_model(self, dtype):
        try:
            model = Gemma3ForConditionalGeneration.from_pretrained(
                self.model_path,
                local_files_only=True,
                dtype=dtype,
                low_cpu_mem_usage=True,
            )
        except TypeError:
            model = Gemma3ForConditionalGeneration.from_pretrained(
                self.model_path,
                local_files_only=True,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
        return self._move_model_to_device(model)

    def _reset_model(self):
        model = self.model
        processor = self.processor
        self.processor = None
        self.model = None
        self.dtype = None
        del model, processor
        release_device_memory(self.device)

    def _select_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _select_compute_dtype(self):
        if self.device.type == "cuda":
            capability = torch.cuda.get_device_capability(self.device)
            if capability[0] >= 8 and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32

    def _device_map(self):
        index = 0 if self.device.index is None else self.device.index
        return {"": f"cuda:{index}"}

    def _attention_context(self):
        if self.device.type != "cuda":
            return nullcontext()
        return sdpa_kernel(
            [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
            set_priority=True,
        )

    def _prepare_vision_modules(self, model):
        if self.device.type != "cuda" or self.dtype != torch.float16:
            return

        multimodal_model = getattr(model, "model", None)
        for name in ("vision_tower", "multi_modal_projector"):
            module = getattr(multimodal_model, name, None)
            if module is not None:
                module.to(dtype=torch.float16)

    def _require_cuda_for_quantized_model(self):
        if self.device.type != "cuda":
            raise RuntimeError(
                "VGen FP4 mode requires CUDA. Refusing to load Gemma on CPU/MPS."
            )

    def _assert_model_not_offloaded(self, model):
        device_map = getattr(model, "hf_device_map", None) or {}
        offloaded = {
            name: device
            for name, device in device_map.items()
            if str(device).startswith("cpu") or str(device) == "disk"
        }
        if offloaded:
            raise RuntimeError(
                "VGen model was offloaded outside CUDA despite FP4 mode: "
                f"{offloaded}"
            )

    def _move_model_to_device(self, model):
        if self.device.type == "cpu":
            return model
        return model.to(self.device)

    def _move_inputs_to_model(self, inputs):
        device = self._model_device()
        if hasattr(inputs, "to"):
            try:
                return inputs.to(device=device, dtype=self.dtype)
            except TypeError:
                return inputs.to(device)

        moved = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                if value.is_floating_point():
                    moved[key] = value.to(device=device, dtype=self.dtype)
                else:
                    moved[key] = value.to(device=device)
            else:
                moved[key] = value
        return moved

    def _model_device(self):
        device = getattr(self.model, "device", None)
        if device is not None:
            return device
        return next(self.model.parameters()).device

    def _decode(self, generated):
        if hasattr(self.processor, "decode"):
            return self.processor.decode(generated, skip_special_tokens=True)
        return self.processor.tokenizer.decode(generated, skip_special_tokens=True)


def get_vgen_runtime(model_path=VGEN_MODEL_PATH):
    global _VGEN_RUNTIME
    normalized_model_path = str(Path(model_path))
    with _VGEN_RUNTIME_LOCK:
        if _VGEN_RUNTIME is None or _VGEN_RUNTIME.model_path != normalized_model_path:
            _VGEN_RUNTIME = VGenRuntime(model_path=normalized_model_path)
        return _VGEN_RUNTIME


def reset_vgen_runtime():
    global _VGEN_RUNTIME
    with _VGEN_RUNTIME_LOCK:
        if _VGEN_RUNTIME is not None:
            _VGEN_RUNTIME.reset_model()
        _VGEN_RUNTIME = None


def release_vgen_runtime():
    with _VGEN_RUNTIME_LOCK:
        if _VGEN_RUNTIME is not None:
            _VGEN_RUNTIME.reset_model()

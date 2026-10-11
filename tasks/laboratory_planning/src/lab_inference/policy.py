"""Frozen policy generation for final plans and JEPA candidate selection."""
from pathlib import Path

from .postprocess import cap_completion, preprocess, repair, signature_map, stable_seed
from .prompts import build_messages


class PlanningPolicy:
    def __init__(self, base_model, adapter, *, family="qwen3", device="cuda:0", dtype="bf16"):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

        self.torch = torch
        self.device = device
        self.family = family
        self.adapter_name = Path(adapter).name
        precision = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
        options = {"dtype": precision, "device_map": {"": device}, "trust_remote_code": False}
        if family == "qwen3":
            self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=False)
            base = AutoModelForCausalLM.from_pretrained(base_model, **options)
        elif family == "qwen35":
            self.processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=False)
            if not self.processor.chat_template and self.processor.tokenizer.chat_template:
                self.processor.chat_template = self.processor.tokenizer.chat_template
            self.tokenizer = self.processor.tokenizer
            base = AutoModelForImageTextToText.from_pretrained(base_model, **options)
        else:
            raise ValueError("Policy family must be qwen3 or qwen35")
        self.model = PeftModel.from_pretrained(base, adapter, is_trainable=False).requires_grad_(False).eval()

    def predict(self, task):
        """Generate the final M_SFT100_E2 plan with signature repair and cap13."""
        if self.family != "qwen3":
            raise ValueError("Final-policy inference uses the qwen3 family")
        encoded = self.tokenizer.apply_chat_template(
            build_messages(task), tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        length = ids.shape[-1]
        if length + 2048 > int(getattr(self.model.config, "max_position_embeddings", 32768)):
            raise ValueError("Input exceeds model context window")
        with self.torch.inference_mode():
            generated = self.model.generate(
                ids.to(self.device), do_sample=False, max_new_tokens=2048,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        raw = self.tokenizer.decode(generated[0, length:], skip_special_tokens=True)
        fixed, repair_detail = repair(raw, signature_map(task["action_pool"]))
        completion, cap_detail = cap_completion(fixed, 13)
        return {"id": task["id"], "completion": completion, "raw_completion": raw,
                "repair": repair_detail, "cap": cap_detail,
                "model": self.adapter_name, "max_new_tokens": 2048}

    def candidates(self, task, seed):
        """Generate one greedy plan and three sampled plans with the P4 policy."""
        if self.family != "qwen35":
            raise ValueError("JEPA candidate generation uses the qwen35 family")
        prompt = self.processor.apply_chat_template(
            build_messages(task), tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.processor(text=[prompt], return_tensors="pt").to(self.device)
        length = inputs["input_ids"].shape[-1]
        with self.torch.inference_mode():
            greedy = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=2048,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        task_seed = stable_seed(seed, task["id"])
        self.torch.manual_seed(task_seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(task_seed)
        with self.torch.inference_mode():
            sampled = self.model.generate(
                **inputs, do_sample=True, num_return_sequences=3, temperature=0.7, top_p=0.95,
                max_new_tokens=2048, pad_token_id=self.tokenizer.eos_token_id,
            )
        candidates = []
        for index, tokens in enumerate([greedy[0], *sampled]):
            suffix = tokens[length:]
            raw = self.tokenizer.decode(suffix, skip_special_tokens=True)
            candidates.append({"candidate_id": f"a{index + 1}", "raw_completion": raw,
                               "completion": preprocess(raw, task["action_pool"]),
                               "output_tokens": len(suffix), "hit_token_limit": len(suffix) >= 2048,
                               "mode": "greedy" if index == 0 else "sampled_batch"})
        return candidates

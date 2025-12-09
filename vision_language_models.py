from abc import ABC, abstractmethod
import torch
import gc
import os
from transformers import (
    BitsAndBytesConfig,
    AutoProcessor,
    Blip2ForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    PaliGemmaForConditionalGeneration,
    LlavaForConditionalGeneration,
)


class BaseVisionLanguageModel(ABC):
    def __init__(self, max_new_tokens=100, quantization=None):
        self.quantization = quantization
        self.model = None
        self.processor = None
        self.max_new_tokens = max_new_tokens

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def generate(self, question, image=None):
        pass

    @abstractmethod
    def form_prompt(self, question, image):
        pass

    def load_model_and_processor_helper(
        self, model_loader, processor_loader=AutoProcessor
    ):
        bnb_config = self.make_bnb_config()
        kwargs = (
            {"quantization_config": bnb_config}
            if bnb_config
            else {"dtype": torch.bfloat16}
        )

        self.processor = processor_loader.from_pretrained(self.model_id)
        self.model = model_loader.from_pretrained(self.model_id, **kwargs)

        if not bnb_config:
            self.model = self.model.to("cuda")

    def cleanup(self):
        """Explicit safe cleanup."""
        try:
            if self.model is not None:
                del self.model
                self.model = None

            if self.processor is not None:
                del self.processor
                self.processor = None

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        except Exception:
            pass

    def make_bnb_config(self):
        if self.quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=torch.bfloat16,
            )
        elif self.quantization == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=torch.bfloat16,
            )
        else:
            return None

    def calculate_model_size_in_bytes(self):
        total_size = 0
        if self.model:
            for param in self.model.parameters():
                total_size += param.numel() * param.element_size()
        return total_size

    def calculate_model_parameters(self):
        total_params = 0
        if self.model:
            for param in self.model.parameters():
                total_params += param.numel()
        return total_params


class Blip2Wrapper(BaseVisionLanguageModel):
    model_id = "Salesforce/blip2-opt-2.7b"

    def load(self):
        self.load_model_and_processor_helper(Blip2ForConditionalGeneration)

    def form_prompt(self, question):
        return f"Question: {question} Answer:"

    def generate(self, image, question):
        prompt = self.form_prompt(question)
        inputs = self.processor(image, text=prompt, return_tensors="pt").to(
            self.model.device
        )

        self.model.eval()
        ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        out = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        return out.split("Answer:")[-1].strip()


class QwenWrapper(BaseVisionLanguageModel):
    model_id = "Qwen/Qwen2-VL-2B-Instruct"

    def load(self):
        self.load_model_and_processor_helper(Qwen2VLForConditionalGeneration)

    def form_prompt(self, question, image):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        return messages

    def generate(self, image, question):
        messages = self.form_prompt(question, image)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text], images=[image], videos=None, return_tensors="pt"
        ).to(self.model.device)

        out_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        trimmed = out_ids[0][inputs.input_ids[0].shape[0] :]
        return self.processor.decode(trimmed, skip_special_tokens=True)


class PaliGemmaWrapper(BaseVisionLanguageModel):
    model_id = "google/paligemma-3b-mix-224"

    def load(self):
        self.load_model_and_processor_helper(PaliGemmaForConditionalGeneration)

    def form_prompt(self, question, image):
        return f"answer en <image> {question}"

    def generate(self, image, question):
        prompt = self.form_prompt(question, image)

        inputs = (
            self.processor(text=prompt, images=image, return_tensors="pt")
            .to(torch.bfloat16)
            .to(self.model.device)
        )

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
            generation = generation[0][input_len:]
            return self.processor.decode(generation, skip_special_tokens=True).strip()


class LlavaWrapper(BaseVisionLanguageModel):
    model_id = "llava-hf/llava-1.5-7b-hf"

    def load(self):
        self.load_model_and_processor_helper(LlavaForConditionalGeneration)

    def form_prompt(self, question, image):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image"},
                ],
            },
        ]

    def generate(self, image, question):
        prompt = self.form_prompt(question, image)
        prompt = self.processor.apply_chat_template(prompt, add_generation_prompt=True)
        inputs = (
            self.processor(images=image, text=prompt, return_tensors="pt")
            .to(torch.bfloat16)
            .to(self.model.device)
        )

        output = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
        )
        output = self.processor.decode(output[0][2:], skip_special_tokens=True)
        return output.strip()


class CustomLlavaWrapper(LlavaWrapper):
    # Base model ID to use for processor (unchanged preprocessing)
    BASE_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
    
    def __init__(self, max_new_tokens=100, quantization=None, custom_model_path=None):
        super().__init__(max_new_tokens, quantization)
        if custom_model_path:
            self.custom_model_path = custom_model_path
    
    def load(self):
        """
        Load custom pruned model weights but use base model's processor.
        Since preprocessing didn't change, we use the standard processor.
        """
        bnb_config = self.make_bnb_config()
        kwargs = (
            {"quantization_config": bnb_config}
            if bnb_config
            else {"dtype": torch.bfloat16}
        )

        # Load processor from base model (preprocessing unchanged)
        self.processor = AutoProcessor.from_pretrained(self.BASE_MODEL_ID)
        
        # Load the pruned model from custom path
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.custom_model_path, **kwargs
        )

        if not bnb_config:
            self.model = self.model.to("cuda")



WRAPPERS = {
    "blip2": Blip2Wrapper,
    "qwen": QwenWrapper,
    "paligemma": PaliGemmaWrapper,
    "llava": LlavaWrapper,
}


def load_model(model_name, max_new_tokens=100, quantization=None):
    """
    Load a vision-language model by name.

    For custom LLaVA models, use the format: "llava:/path/to/model" or "llava:model_name"
    - If model_name (after "llava:") is a relative path, it will be resolved from the pruned_models directory
    - If it's an absolute path, it will be used as-is
    For standard models, use the name from WRAPPERS.
    """
    # Check if it's a custom LLaVA model path
    if model_name.startswith("llava:"):
        custom_path = model_name.split(":", 1)[1]
        
        # If it's not an absolute path, assume it's in the pruned_models directory
        if not os.path.isabs(custom_path):
            # Get the directory where this file is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Construct path to pruned_models directory (assuming it's at project root)
            pruned_models_dir = os.path.join(current_dir, "pruned_models")
            custom_path = os.path.join(pruned_models_dir, custom_path)
        
        # Verify the model directory exists
        if not os.path.exists(custom_path):
            custom_path = "CrystalRaindropsFall/" + model_name.split(":", 1)[1]
        
        obj = CustomLlavaWrapper(
            max_new_tokens, quantization, custom_model_path=custom_path
        )
        obj.load()
        return obj

    # Standard model loading
    cls = WRAPPERS.get(model_name)
    if cls is None:
        raise ValueError(
            f"Unknown model {model_name}. Available: {list(WRAPPERS.keys())}"
        )

    obj = cls(max_new_tokens, quantization)
    obj.load()
    return obj  # contains both model + processor


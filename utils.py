import os
import pickle
from datasets import load_dataset
import torch
from transformers import (
    AutoProcessor,
    Blip2ForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    PaliGemmaForConditionalGeneration,
    LlavaForConditionalGeneration,
    BitsAndBytesConfig,
)


def get_filename_to_full_path_mapping(image_dir):
    filename_to_path = {}
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                filename_to_path[file] = os.path.join(os.path.abspath(root), file)
    return filename_to_path


def load_train_dataset():
    # Get the directory where utils.py is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    mapping_file = os.path.join(data_dir, "image_filenames_mapping.pkl")

    if os.path.exists(mapping_file):
        mapping = pickle.load(open(mapping_file, "rb"))
    else:
        mapping = {}
        # Assuming images dirs are images1 to images7 inside data
        for i in range(1, 8):
            directory = os.path.join(data_dir, f"images{i}")
            if os.path.exists(directory):
                mapping.update(get_filename_to_full_path_mapping(directory))

        # Only save if we found something or if directory exists (to avoid errors if data is missing)
        if os.path.exists(data_dir):
            pickle.dump(mapping, open(mapping_file, "wb"))

    def convert_image_filenames_to_rel_paths(row, mapping=mapping):
        return {"image": mapping.get(row["image"], row["image"])}

    return load_dataset("ChongyanChen/VQAonline", split="train").map(
        convert_image_filenames_to_rel_paths, fn_kwargs={"mapping": mapping}
    )


def form_message(model, messages, image_paths, **kwargs):
    formatted_messages = []
    for message, image_path in zip(messages, image_paths):
        formatted_message = {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": message},
            ],
        }
        if model == "qwen":
            formatted_message["content"][0].extend(kwargs)
        formatted_messages.append(formatted_message)

    return formatted_messages


def load_model(model_name, quantization):
    MODELS = {
        "blip2": "Salesforce/blip2-opt-2.7b",
        "qwen": "Qwen/Qwen2-VL-2B-Instruct",
        "paligemma": "google/paligemma-3b-mix-224",
        "llava": "llava-hf/llava-1.5-7b-hf",
    }
    model_id = MODELS[model_name]

    bnb_config = None
    if quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
    elif quantization == "8bit":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16,
        )

    processor = AutoProcessor.from_pretrained(model_id)
    kwargs = {"quantization_config": bnb_config}

    if not bnb_config:
        kwargs["dtype"] = torch.bfloat16

    if model_name == "blip2":
        model = Blip2ForConditionalGeneration.from_pretrained(model_id, **kwargs)
    elif model_name == "qwen":
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
    elif model_name == "paligemma":
        model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, **kwargs)
    elif model_name == "llava":
        model = LlavaForConditionalGeneration.from_pretrained(model_id, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if quantization not in ["8bit", "4bit"]:
        model = model.to(torch.device("cuda"))

    return model, processor


def generate_answer(model, processor, image, question, model_name):
    model.eval()
    device = model.device

    if model_name == "blip2":
        prompt = f"Question: {question} Answer:"
        inputs = processor(image, text=prompt, return_tensors="pt").to(device)
        generated_ids = model.generate(**inputs, max_new_tokens=100)
        generated_output = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
        if "Answer:" in generated_output:
            generated_output = generated_output.split("Answer:", 1)[1].strip()
        else:
            generated_output = generated_output.strip()
        return generated_output

    elif model_name == "qwen":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info_qwen(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)
        generated_ids = model.generate(**inputs, max_new_tokens=100)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    elif model_name == "smolvlm":
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": question}],
            }
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)
        generated_ids = model.generate(**inputs, max_new_tokens=100)
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ].strip()

    elif model_name == "paligemma":
        prompt = f"answer en <image> {question}"
        inputs = (
            processor(text=prompt, images=image, return_tensors="pt")
            .to(torch.bfloat16)
            .to(device)
        )
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            generation = generation[0][input_len:]
            return processor.decode(generation, skip_special_tokens=True).strip()
    elif model_name == "llava":
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = (
            processor(images=image, text=prompt, return_tensors="pt")
            .to(torch.bfloat16)
            .to(device)
        )
        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        output = processor.decode(output[0][2:], skip_special_tokens=True)
        return output.strip()


def process_vision_info_qwen(messages):
    image_inputs = []
    video_inputs = []
    for message in messages:
        if message["role"] == "user":
            for content in message["content"]:
                if content["type"] == "image":
                    image_inputs.append(content["image"])
                elif content["type"] == "video":
                    video_inputs.append(content["video"])
    return image_inputs, video_inputs if video_inputs else None

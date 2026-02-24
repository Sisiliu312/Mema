import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import math
import re

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
from PIL import Image


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    return split_list(lst, n)[k]


def parse_abcd(output: str):
    """Return 0/1/2/3 for A/B/C/D, or None if not found."""
    out = output.strip().upper()
    m = re.search(r"\b([ABCD])\b", out)
    if m:
        return "ABCD".index(m.group(1))
    m = re.search(r"[\(\[]\s*([ABCD])\s*[\)\]]", out)
    if m:
        return "ABCD".index(m.group(1))
    m = re.search(r"(?:ANSWER|OPTION)\s*[:：]?\s*([ABCD])", out)
    if m:
        return "ABCD".index(m.group(1))
    return None


class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, conv_mode):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]

        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, conv_mode, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, conv_mode)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)


def eval_model(args):
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # auto switch for plain models
    conv_mode = args.conv_mode
    if "plain" in model_name and "finetune" not in model_name.lower() and "mmtag" not in conv_mode:
        conv_mode = conv_mode + "_mmtag"
        print(f"[auto] plain model detected, switching conv_mode -> {conv_mode}")

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    image_folder = os.path.expanduser(args.image_folder)
    missing = [q for q in questions if not os.path.exists(os.path.join(image_folder, q["image"]))]
    if missing:
        missing_files = set(q["image"] for q in missing)
        print(f"[WARN] {len(missing)} questions reference missing images (e.g. {list(missing_files)[:3]}), skipping.")
        questions = [q for q in questions if os.path.exists(os.path.join(image_folder, q["image"]))]
    if not questions:
        raise SystemExit("No questions left after filtering missing images. Check --image-folder and data.")

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True) if os.path.dirname(answers_file) else None
    ans_file = open(answers_file, "w")

    mc_results = []

    data_loader = create_data_loader(questions, image_folder, tokenizer, image_processor, model.config, conv_mode)

    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device="cuda", non_blocking=True)

        # MC 推荐确定性生成
        do_sample = False if args.force_no_sample else (True if args.temperature > 0 else False)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True),
                image_sizes=image_sizes,
                do_sample=do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # keep your original dump jsonl
        ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    "prompt": cur_prompt,
                    "text": outputs,
                    "answer_id": ans_id,
                    "model_id": model_name,
                    "metadata": {},
                },
                ensure_ascii=False,
            )
            + "\n"
        )

        # Visual7W MC export
        if args.mc_out:
            qa_id = line.get("qa_id", idx)
            question = line.get("question", cur_prompt)
            choices = line["choices"]  # len=4
            pred_idx = parse_abcd(outputs)
            if pred_idx is None:
                pred_idx = 0  # fallback
            pred_text = choices[pred_idx]

            mc_results.append(
                {
                    "qa_id": qa_id,
                    "question": question,
                    "candidates": [{"answer": pred_text}],
                }
            )

    ans_file.close()

    if args.mc_out:
        mc_out = os.path.expanduser(args.mc_out)
        os.makedirs(os.path.dirname(mc_out), exist_ok=True) if os.path.dirname(mc_out) else None
        with open(mc_out, "w") as f:
            json.dump(mc_results, f, ensure_ascii=False)
        print(f"[MC] saved: {mc_out}  (#samples={len(mc_results)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--mc_out", type=str, default="my_llava_v7w_mc.json",
                        help="If set, also dump Visual7W-toolkit MC json")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--force_no_sample", action="store_true",
                        help="force do_sample=False (recommended for MC)")

    args = parser.parse_args()
    eval_model(args)

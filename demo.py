from peft import PeftModel
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import PIL
import numpy as np
import cv2
import itertools

from training_toolkit.common.tokenization_utils.segmentation import (
    SegmentationTokenizer,
)

from training_toolkit.common.tokenization_utils.json import (
    JSONTokenizer,
)

MODEL_ID = "google/paligemma-3b-mix-224"
SEG_CHECKPOINT_PATH = "notebooks/paligemma_2024-08-06_09-05-06"
JSON_CHECKPOINT_PATH = "notebooks/paligemma_2024-08-13_07-21-14/checkpoint-240"

# 1. Load the base model straight from the hub
base_model = PaliGemmaForConditionalGeneration.from_pretrained(MODEL_ID)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# 2. Load the first adapter
model = PeftModel.from_pretrained(
    base_model, SEG_CHECKPOINT_PATH, adapter_name="segmentation"
).cuda()

# 3. Load the second adapter
model.load_adapter(JSON_CHECKPOINT_PATH, adapter_name="json")

# 4. Prepare utility classes to process inputs and outputs
segmentation_tokenizer = SegmentationTokenizer()
json_tokenizer = JSONTokenizer(processor)

# Coming straight from the segmentation notebook

class_names = {
    "carton",
    "foam",
    "food",
    "general",
    "glass",
    "metal",
    "paper",
    "plastic",
    "special",
}
PROMPT = "segment " + " ; ".join(class_names)


def generate_segmentations(image, temperature):
    if image is None:
        return
    # Prepare segmentation inputs
    inputs = processor(images=image, text=PROMPT).to(model.device)

    # Enable segmentation adapter
    model.set_adapter("segmentation")
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=temperature > 0,
        temperature=float(temperature),
    )

    # Post process segmentation outputs
    image_token_index = model.config.image_token_index
    num_image_tokens = len(generated_ids[generated_ids == image_token_index])
    num_text_tokens = len(processor.tokenizer.encode(PROMPT))
    num_prompt_tokens = num_image_tokens + num_text_tokens + 2

    generated_text = processor.batch_decode(
        generated_ids[:, num_prompt_tokens:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    w, h, _ = image.shape

    # Reconstruct the segmentation mask
    generated_segmentation = segmentation_tokenizer.decode(generated_text, h, w)
    return generated_segmentation


def annotate_image(image, generated_segmentation):
    # print('generated_segmentation', generated_segmentation)
    if image is None or generated_segmentation is None:
        return None
    generated_segmentation_out = [
        (x["mask"], f"{i}:{(x['name'] if x['name'] else 'None')}")
        for i, x in enumerate(generated_segmentation)
    ]
    return (image, generated_segmentation_out)


with open("notebooks/korea_summary.txt", "r") as f:
    rules = f.read()

from string import Template

PREFIX_TEMPLATE = Template(
    "For every object outlined in the image, here're their detected classes: $items. "
    "For every outlined item, extract JSON with a more accurate label, "
    "as well as disposal directions based on local rules. "
    "The local rules are as follows: $rules"
)


def generate_json(image, generated_segmentation, temperature):
    items = [
        {
            "item_id": i,
            "class": seg["name"].strip() if seg["name"] is not None else "None",
        }
        for i, seg in enumerate(generated_segmentation)
    ]

    prompt = PREFIX_TEMPLATE.substitute(items=items, rules=rules)
    # Enable the JSON adapter
    model.set_adapter("json")

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=temperature > 0,
        temperature=float(temperature),
    )

    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    generated_json = json_tokenizer.decode(generated_text)
    return generated_json


import gradio as gr


def gallery_select(img_gallery, select_data: gr.SelectData):
    # print(select_data.value)
    return select_data.value["image"]["path"]


with gr.Blocks() as demo:
    generated_segmentation = gr.State()

    # gr.HTML('''<h1 align="center">Trash Detection and Sorting Helper</h1>''')

    with gr.Row():
        img_input = gr.Image(label="Upload Image or Select Below")
        img_output = gr.AnnotatedImage(label="Detected Objects")
    with gr.Row():
        img_gallery = gr.Gallery(
            [f"./assets/trash{i}.jpg" for i in range(1, 5)],
            allow_preview=False,
            label="Select an image",
            interactive=True,
            object_fit="cover",
        )
        json_output = gr.JSON(label="Additional Instructions")
    with gr.Row():
        temp_segm = gr.Slider(
            0,
            2,
            value=1,
            label="Segmentation generation temperature",
            info="Choose between 0 and 2 (default 1)",
        )
        temp_json = gr.Slider(
            0,
            2,
            value=1,
            label="JSON generation temperature",
            info="Choose between 0 and 2 (default 1)",
        )
    with gr.Row():
        regenerate_btn = gr.Button("Generate outputs")

    img_gallery.select(gallery_select, img_gallery, img_input)
    upload_event = img_input.change(
        generate_segmentations, [img_input, temp_segm], generated_segmentation
    )
    upload_event.success(
        annotate_image, [img_input, generated_segmentation], img_output
    )
    upload_event.success(
        generate_json, [img_input, generated_segmentation, temp_json], json_output
    )

    click_event = regenerate_btn.click(
        generate_segmentations, [img_input, temp_segm], generated_segmentation
    )
    click_event.success(annotate_image, [img_input, generated_segmentation], img_output)
    click_event.success(
        generate_json, [img_input, generated_segmentation, temp_json], json_output
    )

# demo.launch()
demo.launch(share=True)

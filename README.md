# ♻️ Trash Sorting Assistant

Sorting trash may sound like a thing you only need to figure out once.

That is, until you visit another city. Or your local waste disposal rules change. Or you find yourself having to dispose of shrimp or something tricky like that.

The point is, let's have PaliGemma do it instead.

## Description

This demo showcases the way you can train adapters using the 🦾 **Training Toolkit** to get a single 3B LLM to do a complex task.



There're three notebooks walking you through the process of setting it up:

1. `train_segmentation.ipynb`: Training a segmentation adapter to locate and highlight trash items on the image. 
2. `train_json.ipynb`: Training a RAG / structured output adapter to cross-reference the output of the first adapter with local waste disposal rules and produce instructions.
3. `assemble_peft.ipynb`: Loading the model with both adapters and setting up an example workflow.

You can also run `demo.py` to get the aformentioned workflow wrapped in a simple GUI.

## 🦾 Training Toolkit

We built this library to simplify the process of training adapters for multimodal LLMs.

It comes with presets to train LLMs to do segmentation, extraction and other tasks on images or video.

Take a look at the repo [here](https://github.com/tensorsense/training_toolkit).

## Checkpoints

In case you just want to run the demo to see what it does, use these checkpoints:

- https://huggingface.co/koolkittykat/paligemma_trash_json_adapter
- https://huggingface.co/koolkittykat/paligemma_trash_segm_adapter
"""
RunPod serverless handler — SDXL-Turbo (fast/cheap tier)
"""

import base64
import io

import torch
from diffusers import AutoPipelineForText2Image
import runpod

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe.to("cuda")


def image_to_base64(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def handler(event):
    job_input = event.get("input", {})
    prompt = job_input.get("prompt")
    if not prompt:
        return {"error": "No 'prompt' provided in input."}

    steps = int(job_input.get("num_inference_steps", 2))
    guidance = float(job_input.get("guidance_scale", 0.0))

    image = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
    ).images[0]

    return {"image_base64": image_to_base64(image), "model": "sdxl-turbo"}


runpod.serverless.start({"handler": handler})

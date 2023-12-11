import torch
import jax
import numpy as np
import jax.numpy as jnp
from PIL import Image
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from diffusers import StableDiffusionPipeline
from diffusers import FlaxStableDiffusionImg2ImgPipeline
from pathlib import Path
import sys

# Pop up a dir.
sys.path.append(str(Path(__file__).absolute().parent.parent))
# Import from parent directory.
from config_util import GetModelConfig
# Go back to file path.
sys.path.pop()


DEVICE = GetModelConfig("Device")
def StableDiffusionV1_5():
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(DEVICE)

    prompt = "Merlin, the legendary wizard of Arthurian lore, is an enigmatic and wise figure whose long, flowing beard and robes exude an aura of ancient mysticism. With eyes that gleam with arcane knowledge, he wields a magical staff, channeling the forces of nature and the cosmos to shape destiny itself, embodying the timeless archetype of the sagacious and powerful sorcerer."
    image = pipe(prompt).images[0]
    SaveImage(image, "StableDiffusionV1_5")

def RealisticVisionV6B1():
    pipe = StableDiffusionPipeline.from_pretrained("SG161222/Realistic_Vision_V6.0_B1_noVAE", torch_dtype=torch.float16).to(DEVICE)

    prompt = "Merlin, the legendary wizard of Arthurian lore, is an enigmatic and wise figure whose long, flowing beard and robes exude an aura of ancient mysticism. With eyes that gleam with arcane knowledge, he wields a magical staff, channeling the forces of nature and the cosmos to shape destiny itself, embodying the timeless archetype of the sagacious and powerful sorcerer."
    image = pipe(prompt).images[0]
    SaveImage(image, "RealisticVisionV6B1")

def FlaxStableDiffusionV1_4Im2Im():
    def create_key(seed=0):
        return jax.random.PRNGKey(seed)
    rng = create_key(0)

    ref_img = GetImage("FlaxStableDiffusionV1_4Im2Im", "merlin_ref").convert("RGB")
    ref_img = ref_img.resize((768, 512))

    prompts = "Merlin holding up three fingers in one hand and a spoon in the other."

    pipeline, params = FlaxStableDiffusionImg2ImgPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        revision="flax",
        dtype=jnp.bfloat16,
    )

    num_samples = jax.device_count()
    rng = jax.random.split(rng, jax.device_count())
    prompt_ids, processed_image = pipeline.prepare_inputs(
        prompt=[prompts] * num_samples, image=[ref_img] * num_samples
    )
    p_params = replicate(params)
    prompt_ids = shard(prompt_ids)
    processed_image = shard(processed_image)

    output = pipeline(
        prompt_ids=prompt_ids,
        image=processed_image,
        params=p_params,
        prng_seed=rng,
        strength=0.75,
        num_inference_steps=20,
        jit=True,
        height=512,
        width=768,
    ).images

    output_images = pipeline.numpy_to_pil(np.asarray(output.reshape((num_samples,) + output.shape[-3:])))
    SaveImage(output_images[0], "FlaxStableDiffusionV1_4Im2Im")


def GetImage(modelName, imageName):
    return Image.open(f"images/{modelName}/{imageName}.jpeg")

def SaveImage(image, modelName):
    image.save(f"images/{modelName}/output.png")


if __name__ == "__main__":
    FlaxStableDiffusionV1_4Im2Im()

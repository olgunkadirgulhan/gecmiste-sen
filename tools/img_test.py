"""CPU görsel testi: DreamShaper 8 (SD1.5, CreativeML OpenRAIL-M: ticari kullanım serbest) + LCM-LoRA (4-6 adım)."""
import sys, time, torch
from diffusers import StableDiffusionPipeline, LCMScheduler

STYLE = ("flat 2D storybook illustration, warm muted vintage colors, clean outlines, soft lighting, "
         "detailed background, cinematic wide shot, no text, no watermark")
NEG = "photo, realistic, text, watermark, logo, blurry, deformed, ugly"
PROMPTS = [
    "1960s Istanbul Sirkeci train station platform at dawn, old green passenger train, steam, crowds with suitcases",
    "1960s German car factory interior, assembly line, workers in overalls, morning light through tall windows",
    "small Anatolian village in 1964, stone houses, dirt road, minaret, poplar trees, golden hour",
    "1980s Turkish living room, cathode tube TV, lace doilies, carpet, tea glasses on a tray, evening lamp light",
]
pipe = StableDiffusionPipeline.from_pretrained("Lykon/dreamshaper-8", torch_dtype=torch.float32, safety_checker=None)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
pipe.fuse_lora()
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
torch.set_num_threads(4)
for i, p in enumerate(PROMPTS):
    t = time.time()
    img = pipe(f"{p}, {STYLE}", negative_prompt=NEG, num_inference_steps=6, guidance_scale=1.5,
               width=768, height=432, generator=torch.Generator().manual_seed(100 + i)).images[0]
    img.save(f"out/img{i}.png")
    print(f"img{i} {time.time() - t:.1f}s", flush=True)

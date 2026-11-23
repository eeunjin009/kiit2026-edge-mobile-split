import os, sys, torch
from diffusers import StableDiffusionPipeline, PNDMScheduler
from compression.utils.misc import get_file_list_from_csv                 # hybrid_batch.py:28 과 같은 CSV 읽기
from compression.utils.seed import set_random_seed                        # hybrid_batch.py:29
csv, n, seed, gs, out, lat_out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5], sys.argv[6]
set_random_seed(seed)                                                     # hybrid_batch.py:104
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16,
                                               safety_checker=None).to("cuda")  # inference_pipeline.py:135-163 (모두 fp16), :205 safety off
pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)          # inference_pipeline.py:199-202 (--use_pndm_solver)
pipe.set_progress_bar_config(disable=True)                                # :186
LAT, dec = [], pipe.vae.decode                                           # run_split.py와 같은 지점: 최종 디코드 입력 기록, 값은 그대로 통과
pipe.vae.decode = lambda z, *a, **k: (LAT.append(z.float().cpu()), dec(z, *a, **k))[1]
g = torch.Generator(device="cuda").manual_seed(seed)                      # :192, 프롬프트 사이에 이어서 씀 (generate가 같은 generator 재사용 :226)
os.makedirs(out, exist_ok=True)
for name, prompt in [r[:2] for r in get_file_list_from_csv(csv)[:n]]:     # hybrid_batch.py:183-200, batch_sz 1
    img = pipe(prompt=[prompt], negative_prompt=None, num_inference_steps=25, height=512, width=512, generator=g,
               guidance_scale=gs, num_images_per_prompt=1).images[0]      # inference_pipeline.py:218-230
    img.save(os.path.join(out, name))                                     # hybrid_batch.py:229 (같은 .jpg 이름 → 같은 JPEG 저장)
torch.save(LAT, lat_out)

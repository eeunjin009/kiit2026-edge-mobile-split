# 재사용: hybrid_batch.py 전체(CLI·생성 루프·저장), 감싼 두 메서드도 원래 함수를 먼저 호출.
import os
import runpy
import sys
import torch
import compression.hybrid_sd.inference_pipeline as ip
from mobile_slot import build_qnn, MobileSlot           # 같은 bridge/ 폴더 (sys.path[0])

_orig_step = ip.HybridInferencePipeline.get_step_config
def _get_step_config(self, args):
    total, cfg = _orig_step(self, args)
    if args.use_pndm_solver:
        cfg["step"][total] = cfg["step"][total - 1]
    return total, cfg                                   # total(=25) 그대로 → num_inference_steps 25 (inference_pipeline.py:223)
ip.HybridInferencePipeline.get_step_config = _get_step_config

SLOT, LAT = [], []
_orig_params = ip.HybridInferencePipeline.get_sdm_params
def _get_sdm_params(self):
    s = _orig_params(self)
    if os.environ.get("D5_LATENT_OUT"):
        dec = self.pipe.vae.decode
        self.pipe.vae.decode = lambda z, *a, **k: (LAT.append(z.float().cpu()), dec(z, *a, **k))[1]
    if self.args.steps[1] == 0:
        return s
    self.pipe.unets[1].to("cpu")                        # 교체 전 FP16 U-Net을 GPU에서 내림 (:160 루프 변수 unet이 참조를 쥐고 있어 del로는 해제 안 됨)
    torch.cuda.empty_cache()
    self.pipe.unets[0].to("cpu"); torch.cuda.empty_cache()   # Mobile 생성 중 최대 12.2GiB → Edge U-Net(약 1.6GiB)을 잠시 내림 (pipe.unet과 같은 객체, inference_pipeline.py:184)
    SLOT.append(MobileSlot(build_qnn(os.environ["D5_CKPT"], os.environ["D5_UNET_DIR"]), os.environ["D5_SLOT_LOG"]))
    torch.cuda.empty_cache(); self.pipe.unets[0].to(self.device)   # Edge 복귀 (self.device: inference_pipeline.py:120)
    self.pipe.unets[1] = SLOT[0]                        # unets는 같은 리스트 객체 (inference_pipeline.py:187)
    return s                                            # params 문자열은 교체 전 값 (원래 함수 결과 그대로)
ip.HybridInferencePipeline.get_sdm_params = _get_sdm_params

sys.argv = [os.environ["D5_HYBRID_BATCH"]] + sys.argv[1:]   # <Hybrid-SD>/examples/hybrid_sd/hybrid_batch.py, 원본 CLI 인자 그대로
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
finally:
    SLOT and SLOT[0].dump()                             # 호출별 지연·t·act_k·GPU 피크 JSON 1개 (Mobile을 올린 실행만)
    LAT and torch.save(LAT, os.environ["D5_LATENT_OUT"])

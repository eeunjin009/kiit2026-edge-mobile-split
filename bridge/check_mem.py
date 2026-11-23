# 재사용: inference_qmodel.py:73-103 절차·get_qmodel 그대로. 새로 쓰는 이유: 원본은 VAE·텍스트 인코더·배치 4를 함께 올려 단독 측정 불가.
import json, sys, types, torch
from quant.quant_layer import Scaler; from quant.load_qmodel_util import get_qmodel   # inference_qmodel.py:8,12
from diffusers_rewrite import UNet2DConditionModel                                   # utils.py:14
from mobile_slot import fold_weights                                                 # 같은 bridge/ 폴더 (sys.path[0])
def rec(tag, **kw):  # 할당량, 장치 전체 사용량(mem_get_info, nvidia-smi와 같은 장치 수준), 호스트 RSS/피크(/proc/self/status)
    free, tot = torch.cuda.mem_get_info(); st = dict(l.split(":", 1) for l in open("/proc/self/status"))
    print("D5JSON:" + json.dumps(dict(tag=tag, alloc=torch.cuda.memory_allocated(), max_alloc=torch.cuda.max_memory_allocated(),
          dev_used=tot - free, dev_total=tot, rss=st["VmRSS"].strip(), hwm=st["VmHWM"].strip(), **kw)), flush=True)
unet = UNet2DConditionModel.from_pretrained(sys.argv[2], subfolder="unet").to("cuda"); rec("unet_fp32_gpu")   # utils.py:49
qnn = get_qmodel("sd", types.SimpleNamespace(unet=unet), sys.argv[1],
                 {"bits": 4, "channel_wise": True, "scaler": Scaler.MINMAX}, True,                              # inference_qmodel.py:73-76, :89
                 {"bits": 8, "channel_wise": False, "scaler": Scaler.MINMAX, "leaf_param": True},               # :78-81
                 {"softmax_a_bit": 8, "t2i_log_quant": False, "t2i_real_time": False, "t2i_start_peak": False, "log_max_1": False},  # :83-87
                 False, num_inference_steps=25, time_aware_aqtizer=True); rec("a_after_ckpt_load")               # :90-92
qnn = qnn.float().cuda(); qnn.disable_out_quantization(); rec("b_after_cuda")                                   # :96-103
torch.manual_seed(0); x = torch.randn(2, 4, 64, 64, device="cuda"); e = torch.randn(2, 77, 768, device="cuda")    # 배치 2 = CFG 1프롬프트 (pipeline_stable_diffusion.py:701)
t = torch.tensor(961, device="cuda"); outs = {}
def fwd(name, sl):   # 기존 루프 본문 그대로 (fold 전·후에 재사용)
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.no_grad():
            outs[name] = qnn(x[sl], t, encoder_hidden_states=e[sl])[0]                                          # QuantModel 호출 규약 (quant_model.py:113)
        rec("c_forward_" + name, finite=bool(torch.isfinite(outs[name]).all()))
    except torch.cuda.OutOfMemoryError as err:
        rec("c_forward_%s_OOM" % name, err=str(err)[:300]); torch.cuda.empty_cache()
fwd("b1_row0", slice(0, 1)); fwd("b1_row1", slice(1, 2))                          # a. fold 전 배치 1 (기존과 같음)
fold_weights(qnn); rec("e_after_fold")                                            # b. 가중치 양자화 미리 계산 (mobile_slot.py)
fwd("f1_row0", slice(0, 1)); fwd("f1_row1", slice(1, 2))                          # c. fold 후 같은 입력 x/e/t
if len(outs) == 4:  # fold 전후 출력 차이 (0이어야 같은 계산)
    rec("fold_diff", row0=float((outs["f1_row0"] - outs["b1_row0"]).abs().max()), row1=float((outs["f1_row1"] - outs["b1_row1"]).abs().max()))
fwd("b2", slice(0, 2))                                                            # d. fold 후 배치 2 (OOM 처리 기존 그대로)
if len(outs) == 5:  # 배치 행 독립성: 배치 2의 각 행 vs 같은 입력의 배치 1 (fold 후)
    rec("row_independence", row0=float((outs["b2"][0] - outs["f1_row0"][0]).abs().max()), row1=float((outs["b2"][1] - outs["f1_row1"][0]).abs().max()))

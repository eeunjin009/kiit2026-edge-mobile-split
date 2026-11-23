# 사용: dgq <ckpt> <out.pt> (DGQ 환경, cwd에 ./pretrained) | slot <ckpt> <pretrained_dir> <out.pt> (Hybrid 환경) | cmp <dgq.pt> <slot.pt>
import json, sys, torch
mode, TS = sys.argv[1], (961, 921, 881, 1)
if mode == "cmp":
    a, b = torch.load(sys.argv[2]), torch.load(sys.argv[3])
    for t in TS:  # a: fp32 원본 출력, b: MobileSlot 출력(fp16로 되돌린 값). 같은 dtype(fp16)으로 맞춰 비교 + fp32 기준 차이도 기록
        d16, d32 = (a[t].half().float() - b[t].float()).abs(), (a[t] - b[t].float()).abs()
        print("D5JSON:" + json.dumps(dict(t=t, max_abs_fp16=float(d16.max()), max_abs_vs_fp32=float(d32.max()),
              rel_l2_vs_fp32=float(d32.norm() / a[t].norm()), finite=bool(torch.isfinite(b[t]).all()))))
    sys.exit(0)
torch.manual_seed(0); x = torch.randn(2, 4, 64, 64).half().cuda(); e = torch.randn(2, 77, 768).half().cuda()  # Hybrid가 주는 fp16 입력 (inference_pipeline.py:163)
torch.manual_seed(42)                                                                          # get_qmodel 더미 입력 randn(load_qmodel_util.py:46) 고정
if mode == "dgq":
    from utils import prepare_pipe                                                             # inference_qmodel.py:10 (DGQ/src는 PYTHONPATH, ./pretrained는 cwd 링크)
    from quant.quant_layer import Scaler; from quant.load_qmodel_util import get_qmodel
    pipe = prepare_pipe("sd")                                                                  # inference_qmodel.py:60
    torch.manual_seed(42)
    qnn = get_qmodel("sd", pipe, sys.argv[2], {"bits": 4, "channel_wise": True, "scaler": Scaler.MINMAX}, True,
                     {"bits": 8, "channel_wise": False, "scaler": Scaler.MINMAX, "leaf_param": True},
                     {"softmax_a_bit": 8, "t2i_log_quant": False, "t2i_real_time": False, "t2i_start_peak": False, "log_max_1": False},
                     False, num_inference_steps=25, time_aware_aqtizer=True)                    # :73-92
    qnn.float(); pipe.unet = qnn.cuda(); pipe.unet.disable_out_quantization()                  # :96-103
    call = lambda t: pipe.unet(x.float(), torch.tensor(t).cuda(), encoder_hidden_states=e.float(), return_dict=False)[0]  # 파이프라인의 U-Net 호출 형태
else:
    from mobile_slot import build_qnn, MobileSlot
    slot = MobileSlot(build_qnn(sys.argv[2], sys.argv[3]), "/dev/null")
    call = lambda t: slot(x, torch.tensor(t).cuda(), encoder_hidden_states=e)[0]
with torch.no_grad():
    torch.save({t: call(t).cpu() for t in TS}, sys.argv[-1])

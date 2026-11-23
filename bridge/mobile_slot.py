# bridge/mobile_slot.py — Mobile Device(DGQ W4A8 양자화 U-Net)를 Hybrid unets[1] 자리에 끼우는 얇은 래퍼.
# 재사용: U-Net 생성·양자화·체크포인트 로드는 DGQ get_qmodel(quant/load_qmodel_util.py:28-72)과 inference_qmodel.py:73-103 순서 그대로.
import json
from types import SimpleNamespace

import torch
from quant.quant_layer import Scaler, QuantLayer, StraightThrough   # inference_qmodel.py:8; QuantLayer·StraightThrough는 fold_weights용 (quant_layer.py:14, :577)
from quant.load_qmodel_util import get_qmodel           # inference_qmodel.py:12
from diffusers_rewrite import UNet2DConditionModel      # utils.py:14 (prepare_pipe가 쓰는 재작성 U-Net 클래스)


def fold_weights(qnn):
    with torch.no_grad():
        for m in qnn.modules():
            if isinstance(m, QuantLayer) and m.use_wq:
                m.w.data = m.wqtizer(m.w)
                m.original_w = m.w.data
                m.wqtizer = StraightThrough()
    torch.cuda.empty_cache()


def build_qnn(ckpt, pretrained_dir):
    unet = UNet2DConditionModel.from_pretrained(pretrained_dir, subfolder="unet").to("cuda")
    wq_params = {"bits": 4, "channel_wise": True, "scaler": Scaler.MINMAX}
    aq_params = {"bits": 8, "channel_wise": False, "scaler": Scaler.MINMAX, "leaf_param": True}
    softmax_aq_params = {"softmax_a_bit": 8, "t2i_log_quant": False, "t2i_real_time": False,
                         "t2i_start_peak": False, "log_max_1": False}
    qnn = get_qmodel("sd", SimpleNamespace(unet=unet), ckpt, wq_params, True, aq_params, softmax_aq_params,
                     False, num_inference_steps=25, time_aware_aqtizer=True)
    qnn.float()
    qnn = qnn.cuda()
    qnn.disable_out_quantization()
    fold_weights(qnn)
    return qnn


class MobileSlot:
    def __init__(self, qnn, log_path):
        self.qnn, self.log_path, self.calls = qnn, log_path, []
        self.n_act = sum(1 for k in qnn.ckpt if k.startswith("act_"))
        self.key = next(k for k in qnn.ckpt["act_0"] if k.endswith(".delta"))
        self.probe = qnn.get_submodule(self.key[:-len(".delta")])

    def __call__(self, sample, timestep, encoder_hidden_states=None, **kwargs):
        dtype = sample.dtype
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        out = self.qnn(sample.float(), timestep, encoder_hidden_states=encoder_hidden_states.float())
        end.record()
        torch.cuda.synchronize()
        d = self.probe.delta.detach().cpu()
        self.calls.append({"t": int(timestep.item()), "ms": start.elapsed_time(end),
                           "k_formula": int((1000 - timestep.item()) // 40),
                           "k_loaded": [j for j in range(self.n_act)
                                        if torch.equal(d, self.qnn.ckpt["act_%d" % j][self.key].detach().cpu())]})
        return (out[0].to(dtype),)

    def dump(self):
        with open(self.log_path, "w") as f:
            json.dump({"calls": self.calls, "probe_key": self.key, "n_act": self.n_act,
                       "max_memory_allocated": torch.cuda.max_memory_allocated()}, f)

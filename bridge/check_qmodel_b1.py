import sys, torch
from pytorch_lightning import seed_everything                             # inference_qmodel.py:6
from utils import prepare_pipe                                            # :10 (DGQ/src는 PYTHONPATH)
from quant.quant_layer import Scaler; from quant.load_qmodel_util import get_qmodel  # :8, :12
prompt = sys.argv[3] if len(sys.argv) > 3 else "a painting of a virus monster playing guitar"   # :24 기본값
pipe = prepare_pipe("sd")                                                 # :60
seed_everything(42)                                                       # :63 (--seed 기본 42); FP 이미지(:70)는 만들지 않음
qnn = get_qmodel("sd", pipe, sys.argv[1], {"bits": 4, "channel_wise": True, "scaler": Scaler.MINMAX}, True,
                 {"bits": 8, "channel_wise": False, "scaler": Scaler.MINMAX, "leaf_param": True},
                 {"softmax_a_bit": 8, "t2i_log_quant": False, "t2i_real_time": False, "t2i_start_peak": False, "log_max_1": False},
                 False, num_inference_steps=25, time_aware_aqtizer=True)  # :73-92
qnn.float(); pipe.unet = qnn.cuda(); pipe.unet.disable_out_quantization()  # :96-103
pipe.safety_checker = None; pipe.requires_safety_checker = False          # :104-105
seed_everything(42)                                                       # :47 inference() 안의 seed
pipe([prompt], num_inference_steps=25)[0][0].save(sys.argv[2])            # :51-53, 배치만 1
print("D5JSON:{\"max_memory_allocated\": %d}" % torch.cuda.max_memory_allocated())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 사용(Kaggle):  !python d9_kaggle.py        → 단일 세션, 실행 순서 S0 S1 S2 S3 S4 S6 S5 S8 S7. S1이 OOM/FAIL이면 Mobile 필요 항목만 SKIPPED(우회 없음).
#                옵션: --skip S3,S4   --redo S1   --ckpt <경로>   --budget-min 140 (넘으면 남은 명령을 실행하지 않음)
# 산출물: /kaggle/working/d9_out/ (d9_report.md, d9_state.json, logs/<단계>_<이름>.log, 이미지·JSON)
import argparse, datetime, json, os, shlex, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(globals().get("__file__") or os.path.join(os.getcwd(), "d9_kaggle.py")))
BR = os.path.join(HERE, "bridge")
ROOT, HF = "/kaggle/working", "/tmp/hf"
OUT = ROOT + "/d9_out"
LOGS = OUT + "/logs"
DGQ, HYB, VENV = ROOT + "/DGQ", ROOT + "/Hybrid-SD", ROOT + "/venv"
PY, SYS = VENV + "/bin/python", sys.executable
PRE = DGQ + "/pretrained/stable-diffusion-v1-4"
META = ROOT + "/data/mscoco_val2014_30k/metadata.csv"
META_SUB = OUT + "/meta_n200.csv"                          # META에서 bridge/make_subset.py가 만든 부분집합 (생성·CLIP 입력)
SUB_SEED, SUB_SKIP = 2026, 50                              # 선택 해시 시드, 앞 50행(N=50 예비 실험 범위)은 제외
SD14_LINK = ROOT + "/models/sd14_mobile_slot"
RUN = OUT + "/run"                                 # cwd: ./pretrained -> DGQ/pretrained (utils.py:18 규약)
RUN_INF = OUT + "/run_infer"                       # 원본 inference_qmodel.py의 tmp_*.png 저장 위치(cwd, inference_qmodel.py:53)
CKPT_DEFAULT = "/kaggle/input/dgq-sd14-w4a8g1-merged/cali_ckpt_activation_w4a8g1.pth_merged"
CKPT = CKPT_DEFAULT                                # main()에서 --ckpt로 바뀜
DGQ_URL, DGQ_SHA = "https://github.com/ugonfor/DGQ", "f449f6a"
HYB_URL, HYB_SHA = "https://github.com/bytedance/Hybrid-SD", "dcac636"
META_URL = "https://netspresso-research-code-release.s3.us-east-2.amazonaws.com/data/mscoco_val2014_30k/metadata.csv"  # get_mscoco_files.sh
PINS = ("torch==2.2.2+cu121 torchvision==0.17.2+cu121 'numpy<2' diffusers==0.27.0 huggingface_hub==0.24.7 transformers==4.44.2 "
        "peft==0.2.0 scikit-learn==1.5.1 pandas==2.2.1 pytorch_lightning einops accelerate safetensors pyyaml pynvml "
        "torch-fidelity==0.3.0 pytorch-fid==0.3.0 open_clip_torch==3.3.0")   # open_clip_torch는 예비 실험(N=50)과 같은 버전으로 고정
INFER_PROMPT = "a painting of a virus monster playing guitar"          # inference_qmodel.py:24
D6_N = int(os.environ.get("D9_N", "200"))                         # 구성당 이미지 수 = META_SUB 행 수 (환경변수 D9_N)
D9_ALL = {"B0": "25,0", "Q5": "20,5", "Q10": "15,10", "Q15": "10,15", "Q20": "5,20", "Q25": "0,25"}   # 구성 이름 -> --steps
D9_TAG = os.environ.get("D9_TAG", "X")                            # 세션 이름 (결과 파일 이름에 쓰임)
D6_CFGS = [(c, D9_ALL[c]) for c in os.environ.get("D9_CFGS", "B0,Q20,Q25").split(",")]   # 이 세션이 만들 구성, 실행 순서 고정 (환경변수 D9_CFGS)
assert D6_CFGS[0][0] == "B0" and len(D6_CFGS) > 1, D6_CFGS         # B0가 맨 앞(PSNR 기준), Mobile 구성이 1개 이상
D6_Q = [c for c, st in D6_CFGS if st.split(",")[1] != "0"]        # Mobile을 쓰는 구성 (n_mobile > 0)
STEPS = ["S0", "SUB", "REG"] + [c for c, _ in D6_CFGS] + ["PSNR", "BASE", "CLIP", "S7", "PACK"]   # D9 실행 순서. 진단 단계 S1~S6·S8은 제외
CORE = ["S0", "SUB", "S2", "S3", "S4"]                             # 이 중 FAIL이면 뒤 단계 SKIPPED (S7·PACK은 항상 실행)
MOBILE = {"q_5_20", "routing_q", "2_dgq", "2_slot", "2_cmp", "4_actk"}   # S1(Mobile) 실패 시 SKIPPED할 명령 (S8은 단계 전체)
MOBILE |= {"%s_%s" % (k, c) for k in ("gen", "routing", "info", "slot", "psnr", "stat", "clip", "clippi") for c in D6_Q}   # 첫 Q 구성이 Mobile 로드 전에 실패하면 뒤 Q 명령 SKIPPED

NEED = ("mobile_slot.py", "run_split.py", "check_img.py", "make_subset.py", "make_shift.py", "clip_per_image.py", "pack_summary.py")   # 필수 bridge 7개
EXPECTED_SIZE = 6883907740

COMMON = ("HF_HOME=%s PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONBREAKPOINT=0 CUDA_VISIBLE_DEVICES=0 DIFFUSERS_REWRITE=sd "
          "TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled" % HF)
DGQ_ENV = COMMON + " PYTHONPATH=%s/diffusers/src:%s:%s/src" % (DGQ, DGQ, DGQ)   # DGQ 규약 + src (포크 diffusers 0.26.0, d0 Gate1-D)
HYB_ENV = COMMON + " PYTHONPATH=%s:%s" % (HYB, DGQ)                  # Hybrid 규약 (venv diffusers 0.27.0, d0 Gate1-C) + DGQ quant


def gpu(cmd, tag):
    """명령 실행 중 nvidia-smi 사용량을 0.5초마다 기록하고 최댓값을 출력, /usr/bin/time -v가 있으면 호스트 최대 RSS도."""
    smi = "%s/smi_%s.txt" % (OUT, tag)
    return ("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > %s & SMI=$!; "
            "T=$([ -x /usr/bin/time ] && echo '/usr/bin/time -v'); $T bash -c %s; rc=$?; kill $SMI; "
            "echo \"nvidia-smi peak MiB: $(sort -n %s | tail -1)\"; exit $rc" % (smi, shlex.quote(cmd), smi))


def split(steps, tag, out, data=META_SUB, n=2, seed=1234, gs=7, latent=""):
    """bridge/run_split.py로 원본 hybrid_batch.py 실행 (CLI 인자는 원본 그대로)."""
    env = "%s D5_CKPT=%s D5_UNET_DIR=%s D5_SLOT_LOG=%s/slot.json D5_HYBRID_BATCH=%s/examples/hybrid_sd/hybrid_batch.py%s" % (
        HYB_ENV, CKPT, PRE, out, HYB, (" D5_LATENT_OUT=" + latent) if latent else "")
    return gpu("mkdir -p %s && cd %s && %s %s -u %s/run_split.py --model_id CompVis/stable-diffusion-v1-4 %s --steps %s "
               "--use_pndm_solver --img_sz 512 --img_resz 256 --guidance_scale %s --batch_sz 1 --seed %d --data_list %s "
               "--max_n_files %d --output_dir %s" % (out, OUT, env, PY, BR, SD14_LINK, steps, gs, seed, data, n, out), tag)


def preflight(br, ckpt, default_ckpt, expected_size):
    """S0 전에 bridge 7개(이름)·ckpt 존재·크기를 확인. 반환 (ok, 출력줄 리스트)."""
    import hashlib                                                            # sha256 앞 12자리 출력용
    sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()[:12]     # 작은 .py만 읽음 (ckpt는 해시 안 함)
    why, out = [], []                                                         # 실패 사유 / 통과 시 출력줄
    miss = [n for n in NEED if not os.path.isfile(os.path.join(br, n))]       # 검사1: 이름으로 확인
    if miss:
        why.append("bridge(%s)에 없음: %s" % (br, ", ".join(miss)))           # 없는 이름 나열
    if not os.path.isfile(ckpt):                                              # 검사2: ckpt 존재
        why.append("ckpt 없음: %s" % ckpt)
    elif os.path.getsize(ckpt) != expected_size:                              # 검사3: 크기
        msg = "ckpt 크기 %d != 기대 %d (%s)" % (os.path.getsize(ckpt), expected_size, ckpt)
        if ckpt == default_ckpt:
            why.append(msg)                                                   # 기본 경로: 불일치는 FAIL
        else:
            out.append("경고(--ckpt): " + msg)                                # --ckpt 사용: 경고만
    if why:
        return False, why + ["힌트: bridge를 /kaggle/working/bridge 에 복사했는지, 같은 세션인지 확인"]   # 해결 힌트 한 줄
    out += ["%s  %s" % (sha(os.path.join(br, n)), n) for n in NEED]          # 통과: 7개 sha256 앞 12자리
    me = os.path.join(os.path.dirname(br), "d9_kaggle.py")                    # 이 스크립트 = HERE/d9_kaggle.py (BR = HERE/bridge)
    out.append("%s  d9_kaggle.py" % sha(me) if os.path.isfile(me) else "스크립트가 디스크에 없음(셀 실행)")
    return True, out


def log(step, name):
    return "%s/%s_%s.log" % (LOGS, step, name)


def routing(step, name):   # "Step:{i} Unet Model is:{name} {index}" (pipeline_stable_diffusion.py:707) 줄을 모델별로 셈
    return ("grep -c 'Unet Model is' %s; grep -o 'Unet Model is:[^ ]* [0-9]*' %s | sort | uniq -c; grep 'len(timesteps)' %s | head -2; "
            "echo \"invalid value 경고: $(grep -c 'invalid value' %s)\"" % ((log(step, name),) * 4))


def plan():
    """단계별 (이름, bash 명령, timeout초, 실패 허용) 목록."""
    q_csv = OUT + "/meta_q.csv"
    me = (" %s/d9_kaggle.py" % HERE) if os.path.isfile(HERE + "/d9_kaggle.py") else ""
    infer_png = "%s/tmp_sd_%s_0_w4a8g1.png" % (RUN_INF, INFER_PROMPT.replace(" ", "_"))   # inference_qmodel.py:53, :108
    dd = lambda c: "%s/d9_%s" % (OUT, c)                                 # D6 결과 폴더 (<dir>/im512, im256, log, model_info.json — D5와 같은 규칙)
    d6 = {c: [("gen_" + c, split(st, "d9_" + c, dd(c), n=D6_N), 14400, False),   # 공통 인자는 split() 기본값 그대로 (seed 1234, gs 7, META)
              ("routing_" + c, routing(c, "gen_" + c), 60, True),          # Edge/Mobile 호출 수, len(timesteps), 'invalid value' 수
              ("info_" + c, "%s -c \"import json, os; d = '%s'; n5, n2 = len(os.listdir(d + '/im512')), len(os.listdir(d + '/im256')); "
                            "lat = json.load(open(d + '/model_info.json'))['latency']; "
                            "print('D5JSON:' + json.dumps(dict(cfg='%s', N=%d, n512=n5, n256=n2, sec_per_image=lat%s))); assert n5 == n2 == %d\""
                            % (PY, dd(c), c, D6_N, ", edge_call_upper_s=lat / 26" if c == "B0" else "", D6_N), 60, False)]   # N 확인; B0는 26호출(PNDM) 전부 Edge → ÷26 = Edge 호출 지연 상한
          + ([("slot_" + c, "%s -c \"import json; c = json.load(open('%s/slot.json')); ms = [x['ms'] for x in c['calls']]; print('D5JSON:' + json.dumps(dict("
                            "cfg='%s', mobile_calls=len(ms), mobile_mean_ms=sum(ms) / len(ms), k_match=sum(x['k_formula'] in x['k_loaded'] for x in c['calls']), "
                            "max_memory_allocated=c['max_memory_allocated'])))\"" % (PY, dd(c), c), 60, False)] if c in D6_Q else [])   # mobile_slot.py 기록 요약
          for c, st in D6_CFGS}
    d6["PSNR"] = [x for c in D6_Q for x in (                              # B0 대비, 파일 이름으로 짝 (check_img.py:8), im512 = 모델 원출력
        ("psnr_" + c, "%s %s/check_img.py %s/im512 %s/im512" % (PY, BR, dd("B0"), dd(c)), 600, False),
        ("stat_" + c, "%s -c \"import json, os, statistics as s; r = [json.loads(l[7:]) for l in open('%s') if l.startswith('D5JSON:')]; "
                      "r = [dict(file=os.path.basename(x['a']), psnr=x['psnr']) for x in r if 'psnr' in x]; json.dump(r, open('%s/psnr_vs_B0.json', 'w'), indent=1); "
                      "v = [x['psnr'] for x in r if x['psnr'] is not None]; print('D5JSON:' + json.dumps(dict(cfg='%s', ref='B0', n=len(v), ddof=0, mean=s.mean(v), "
                      "std=s.pstdev(v), min=min(v), per_image='%s/psnr_vs_B0.json'))); assert len(v) == %d\""
                      % (PY, log("PSNR", "psnr_" + c), dd(c), c, dd(c), D6_N), 60, False))]   # check_img per-image 값 원본 저장 + 평균·표준편차(ddof=0)·최솟값
    d6["CLIP"] = [("disk", "df -h %s /tmp; du -sh %s 2>/dev/null" % (ROOT, HF), 60, True)] + [x for c, _ in D6_CFGS for x in (   # ViT-g-14 가중치 약 5.1GiB가 HF_HOME(/tmp/hf)로
        ("clip_" + c, gpu("cd %s && %s %s %s/src/eval_clip_score.py --data_list %s --img_dir %s/im256 --num_imgs %d --save_txt %s/clip.txt"
                          % (RUN, DGQ_ENV, PY, DGQ, META_SUB, dd(c), D6_N, dd(c)), "clip_" + c), 3600, False),       # 원본 호출만(평균), 기본 im256
        ("clippi_" + c, gpu("cd %s && %s %s %s/clip_per_image.py --dgq_src %s/src --data_list %s --img_dir %s/im256 --num_imgs %d --out %s/clip_per_image.json"
                            % (RUN, DGQ_ENV, PY, BR, DGQ, META_SUB, dd(c), D6_N, dd(c)), "clippi_" + c), 3600, False))]   # 같은 계산, 이미지별 점수
    d6["SUB"] = [("subset", "%s %s/make_subset.py %s %s %d %d %d && head -3 %s && wc -l %s"
                  % (PY, BR, META, META_SUB, D6_N, SUB_SEED, SUB_SKIP, META_SUB, META_SUB), 300, False)]       # 무작위 부분집합
    d6["REG"] = [("gen_reg", split("25,0", "d9_reg", OUT + "/d9_reg", data=META, n=2), 1800, False),          # 예비 실험과 같은 B0 앞 2행 재현 (같은 환경이면 mean_a가 같음)
                 ("routing_reg", routing("REG", "gen_reg"), 60, True),
                 ("img_reg", "%s %s/check_img.py %s/d9_reg/im512 %s/d9_reg/im512" % (PY, BR, OUT, OUT), 120, False),
                 ("cmp_reg", "%s -c \"import json, os; r = [json.loads(l[7:]) for l in open('%s') if l.startswith('D5JSON:')]; r = [x for x in r if 'mean_a' in x]; "
                             "e = [113.76008860270183, 118.2762082417806]; print('D5JSON:' + json.dumps([dict(file=os.path.basename(x['a']), mean_a=x['mean_a'], "
                             "expected=y, equal=x['mean_a'] == y) for x, y in zip(r, e)])); assert len(r) == 2\"" % (PY, log("REG", "img_reg")), 60, False)]
    d6["BASE"] = [("shift", "%s %s/make_shift.py %s/im512 %s/d9_base/im512 %d" % (PY, BR, dd("B0"), OUT, D6_N // 2), 300, False),   # PSNR 기준선 = 서로 다른 캡션 쌍
                  ("psnr_base", "%s %s/check_img.py %s/im512 %s/d9_base/im512" % (PY, BR, dd("B0"), OUT), 600, False),
                  ("stat_base", "%s -c \"import json, os, statistics as s; r = [json.loads(l[7:]) for l in open('%s') if l.startswith('D5JSON:')]; "
                                "r = [dict(file=os.path.basename(x['a']), psnr=x['psnr']) for x in r if 'psnr' in x]; json.dump(r, open('%s/d9_base/psnr_unrelated.json', 'w'), indent=1); "
                                "v = [x['psnr'] for x in r if x['psnr'] is not None]; print('D5JSON:' + json.dumps(dict(cfg='B0_shift', ref='B0', n=len(v), ddof=0, mean=s.mean(v), "
                                "std=s.pstdev(v), min=min(v), max=max(v), per_image='%s/d9_base/psnr_unrelated.json'))); assert len(v) == %d\""
                                % (PY, log("BASE", "psnr_base"), OUT, OUT, D6_N), 60, False)]
    d6["PACK"] = [("pack", "%s %s/pack_summary.py %s %s %d %s" % (PY, BR, OUT, D9_TAG, D6_N, ",".join(c for c, _ in D6_CFGS)), 600, False)]   # 올릴 파일 2개로 묶기
    return {
        "S0": [
            ("disk", "df -h %s /tmp; du -sh %s 2>/dev/null" % (ROOT, ROOT), 300, False),
            ("gpu", "nvidia-smi", 60, False),
            ("ram", "free -m; grep MemTotal /proc/meminfo", 60, False),
            ("ckpt", "ls -la %s; ls -la $(dirname %s)" % (CKPT, CKPT), 60, False),
            ("clone_hyb", "test -d %s/.git || (git clone %s %s && git -C %s -c advice.detachedHead=false checkout %s); "
                          "git -C %s rev-parse HEAD" % (HYB, HYB_URL, HYB, HYB, HYB_SHA, HYB), 900, False),
            ("clone_dgq", "test -d %s/.git || (git clone %s %s && git -C %s -c advice.detachedHead=false checkout %s); "
                          "git -C %s rev-parse HEAD" % (DGQ, DGQ_URL, DGQ, DGQ, DGQ_SHA, DGQ), 900, False),
            ("venv", "test -x %s || (%s -m pip install -q uv && %s -m uv venv --python 3.9 --seed %s && %s -m uv pip install "
                     "--python %s %s --extra-index-url https://download.pytorch.org/whl/cu121 --index-strategy unsafe-best-match); "
                     "%s -c 'import torch, diffusers; print(torch.__version__, diffusers.__version__, torch.cuda.is_available())'"
                     % (PY, SYS, SYS, VENV, SYS, PY, PINS, PY), 2400, False),
            ("rundirs", "mkdir -p %s/pretrained %s %s && ln -sfn %s/pretrained %s/pretrained && ln -sfn %s/pretrained %s/pretrained"
                        % (DGQ, RUN, RUN_INF, DGQ, RUN, DGQ, RUN_INF), 60, False),
            ("pretrained", "test -d %s/unet || (cd %s && %s %s -c \"from utils import prepare_pipe; prepare_pipe('sd')\"); "
                           "ls %s %s/unet" % (PRE, RUN, DGQ_ENV, PY, PRE, PRE), 3600, False),   # utils.py:16-23 원본 그대로
            ("metadata", "test -s %s || (mkdir -p $(dirname %s) && curl -sSfL %s -o %s); head -3 %s; wc -l %s"
                         % (META, META, META_URL, META, META, META), 300, False),
            ("sd14_link", "%s %s -c \"from diffusers import DiffusionPipeline as P; print(P.download('CompVis/stable-diffusion-v1-4'))\" "
                          "| tail -1 > %s/sd14_path.txt && mkdir -p %s/models && ln -sfn $(cat %s/sd14_path.txt) %s && ls -la %s/"
                          % (HYB_ENV, PY, OUT, ROOT, OUT, SD14_LINK, SD14_LINK), 3600, False),
            ("q_csv", "printf 'file_name,text\\nq0.jpg,%s\\n' > %s; cat %s" % (INFER_PROMPT, q_csv, q_csv), 60, False),
            ("disk_after", "df -h %s /tmp; du -sh %s %s %s %s %s 2>/dev/null" % (ROOT, ROOT, VENV, DGQ, OUT, HF), 600, False),
        ],
        "S1": [("mobile_mem", gpu("cd %s && %s %s %s/check_mem.py %s %s" % (RUN, DGQ_ENV, PY, BR, CKPT, PRE), "s1"), 3600, False)],
        "S2": [("record", "nvidia-smi --query-gpu=name,memory.total --format=csv; echo 'S2 판정은 S1 수치로 사람이 한다 (d5는 기록만)'", 60, False)],
        "S3": [
            ("unet_load_027", "cd %s && %s %s -c \"import diffusers; from diffusers_rewrite import UNet2DConditionModel as U; "
                              "u = U.from_pretrained('%s', subfolder='unet'); print(diffusers.__version__, diffusers.__file__, "
                              "type(u).__module__, sum(p.numel() for p in u.parameters()))\"" % (OUT, HYB_ENV, PY, PRE), 1200, False),
            ("import_027", "cd %s && %s %s -c \"import diffusers, quant.load_qmodel_util, compression.hybrid_sd.inference_pipeline; "
                           "print('ok', diffusers.__version__, diffusers.__file__)\"" % (OUT, HYB_ENV, PY), 600, False),
            ("model_names", "echo stable-diffusion-v1-4 $(basename %s)   # hybrid_batch.py:110 model_name = 경로 마지막 이름" % SD14_LINK, 60, False),
        ],
        "S4": [
            ("files", "wc -l %s/*.py%s; sha256sum %s/*.py" % (BR, me, BR), 60, False),
            ("orig_clean", "git -C %s status --porcelain; echo '[DGQ end]'; git -C %s status --porcelain; echo '[Hybrid end]'" % (DGQ, HYB), 60, False),
        ],
        "S6": [   # 우선순위: B0 → Q(5,20) → B0 재실행
            ("b0_run1", split("25,0", "s5_b0", OUT + "/s5_b0", latent=OUT + "/lat_b0.pt"), 3600, False),   # B0: Mobile 교체 안 함 (S5a가 이 출력 사용)
            ("routing_b0_1", routing("S6", "b0_run1"), 60, True),
            ("q_5_20", split("5,20", "s6_q", OUT + "/s6_q"), 3600, False),
            ("routing_q", routing("S6", "q_5_20"), 60, True),
            ("b0_run2", split("25,0", "s6_b0_2", OUT + "/s6_b0_2"), 3600, False),
            ("routing_b0_2", routing("S6", "b0_run2"), 60, True),
            ("b0_repeat", "%s %s/check_img.py %s/s5_b0/im512 %s/s6_b0_2/im512" % (PY, BR, OUT, OUT), 300, False),
            ("meta", "for d in s5_b0 s6_b0_2 s6_q; do echo $d; cat %s/$d/model_info.json; echo; done; %s -c \"import json; c = json.load(open('%s/s6_q/slot.json')); "
                     "ms = [x['ms'] for x in c['calls']]; print('mobile calls', len(ms), 'mean ms', sum(ms) / len(ms) if ms else None, "
                     "'max_memory_allocated', c['max_memory_allocated'])\"" % (OUT, PY, OUT), 120, True),   # slot.json은 Mobile을 올린 실행만
            ("disk", "du -sh %s %s %s %s 2>/dev/null; df -h %s" % (OUT, VENV, DGQ, HF, ROOT), 600, False),
        ],
        "S5": [   # 나머지 동등성 (S6 뒤 실행, 실패는 기록만)
            ("1ii_ref", gpu("cd %s && %s %s %s/check_ref_sd.py %s 2 1234 7 %s/s5_ref %s/lat_ref.pt" % (OUT, HYB_ENV, PY, BR, META, OUT, OUT), "s5_ref"), 3600, False),
            ("1ii_cmp", "%s %s/check_img.py %s/s5_b0/im512 %s/s5_ref" % (PY, BR, OUT, OUT), 300, False),
            ("1ii_latent", "%s -c \"import json, torch; a = torch.load('%s/lat_b0.pt'); b = torch.load('%s/lat_ref.pt'); "   # 최종 디코드 입력 비교
                           "[print('D5JSON:' + json.dumps(dict(i=i, shape=list(x.shape), max_abs=float((x - y).abs().max()), "
                           "rel_l2=float((x - y).norm() / y.norm())))) for i, (x, y) in enumerate(zip(a, b))]; print('n', len(a), len(b))\""
                           % (PY, OUT, OUT), 300, False),
            ("2_dgq", gpu("cd %s && %s %s %s/check_mobile_eq.py dgq %s %s/eq_dgq.pt" % (RUN, DGQ_ENV, PY, BR, CKPT, OUT), "s5_eq_dgq"), 3600, False),
            ("2_slot", gpu("cd %s && %s %s %s/check_mobile_eq.py slot %s %s %s/eq_slot.pt" % (OUT, HYB_ENV, PY, BR, CKPT, PRE, OUT), "s5_eq_slot"), 3600, False),
            ("2_cmp", "%s %s %s/check_mobile_eq.py cmp %s/eq_dgq.pt %s/eq_slot.pt" % (HYB_ENV, PY, BR, OUT, OUT), 300, False),
            ("4_actk", "%s -c \"import json; [print(c['t'], c['k_formula'], c['k_loaded'], round(c['ms'], 1)) "
                       "for c in json.load(open('%s/s6_q/slot.json'))['calls']]\"" % (PY, OUT), 60, False),
            ("1i_orig_25_0", gpu("mkdir -p %s/s5_orig && cd %s && %s %s -u %s/examples/hybrid_sd/hybrid_batch.py --model_id "
                                 "CompVis/stable-diffusion-v1-4 %s --steps 25,0 --use_pndm_solver --img_sz 512 --img_resz 256 "
                                 "--guidance_scale 7 --batch_sz 1 --seed 1234 --data_list %s --max_n_files 1 --output_dir %s/s5_orig"
                                 % (OUT, OUT, HYB_ENV, PY, HYB, SD14_LINK, META, OUT), "s5_orig"), 3600, True),
            ("1i_keyerror", "grep -n 'KeyError' %s | tail -3" % log("S5", "1i_orig_25_0"), 60, True),
        ],
        "S8": [   # 선택 항목 (예산이 남을 때만, 실패는 앞 단계에 영향 없음)
            ("3a_orig_infer", gpu("cd %s && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True %s %s -u %s/src/inference_qmodel.py "
                                  "--cali_ckpt %s --wq 4 --use_aq --aq 8 --time_aware_aqtizer < /dev/null"   # README 5단계, 원본 그대로
                                  % (RUN_INF, DGQ_ENV, PY, DGQ, CKPT), "s5_infer"), 3600, True),
            ("3b_b1", gpu("cd %s && %s %s %s/check_qmodel_b1.py %s %s/s5_b1.png" % (RUN, DGQ_ENV, PY, BR, CKPT, OUT), "s5_b1"), 3600, False),
            ("3c_hyb_m26", split("0,25", "s5_m26", OUT + "/s5_m26", data=q_csv, n=1, seed=42, gs=7.5), 3600, False),  # 원본 guidance 기본 7.5
            ("3d_cmp", "ls %s/*.png; %s %s/check_img.py %s %s/s5_b1.png; %s %s/check_img.py %s/s5_b1.png %s/s5_m26/im512/q0.jpg"
                       % (RUN_INF, PY, BR, infer_png, OUT, PY, BR, OUT, OUT), 300, True),
        ],
        "S7": [
            ("new_code", "wc -l %s/*.py%s; sha256sum %s/*.py%s" % (BR, me, BR, me), 60, False),
            ("orig_clean", "git -C %s status --porcelain; echo '[DGQ end]'; git -C %s status --porcelain; echo '[Hybrid end]'" % (DGQ, HYB), 60, False),
        ],
        **d6,                                                           # D6 단계: 구성별 생성, PSNR, CLIP
    }


def run(step, name, cmd, timeout, env):
    os.makedirs(LOGS, exist_ok=True)
    lp = log(step, name)
    t0 = time.time()
    with open(lp, "w", encoding="utf-8") as f:
        f.write("$ %s\n" % cmd)
        f.flush()
        try:
            rc = subprocess.run(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                env=env, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout(%ss)" % timeout
    text = open(lp, encoding="utf-8", errors="replace").read()
    lines = text.splitlines()
    return {"name": name, "cmd": cmd, "rc": rc, "secs": round(time.time() - t0, 1), "log": lp, "tail": lines[-40:],
            "d5json": [l for l in lines if l.startswith("D5JSON:") or l.startswith("nvidia-smi peak")
                       or "Maximum resident set size" in l]}


def report(state, meta):
    L = ["# D5 실행 기록 (d9_kaggle.py, 판정 없음 — 수치는 출력 그대로)", "",
         "- 시각 %s, 인자 `%s`, 무시한 인자 `%s`, ckpt `%s`, 시간 예산 %s분, 경과 %.1f분" % (
             meta["time"], meta["argv"], meta["unknown"], CKPT, meta["budget"], meta["elapsed"]), "",
         "- 원본 파일 직접 수정 0건. 실행 중 래핑(bridge/run_split.py): `HybridInferencePipeline.get_step_config`(키 25 추가, U4), "
         "`HybridInferencePipeline.get_sdm_params`(원래 함수 후 unets[1]→MobileSlot, **n_mobile=0이면 교체 안 함**; S5a에서만 vae.decode 입력 기록). "
         "환경변수만: PYTHONBREAKPOINT=0, S5 3a_orig_infer에서 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
         "- **B0(`25,0`)는 Mobile을 올리지 않으므로 Mobile 경로를 검증하지 않는다.**",
         "- S5a(1ii_*): Hybrid B0 vs diffusers StableDiffusionPipeline — 최종 latent 차이와 이미지 PSNR 기록. "
         "S5b(2_*): 같은 코드를 부르는 입출력 변환 확인.",
         "- D6: 구성 %s, N=%d, 공통 인자 D5와 같음. PSNR = B0 im512 대비(B0 자신은 ∞, 기준점; n·ddof=0 명시, per-image는 d9_<구성>/psnr_vs_B0.json), "
         "CLIP = DGQ eval_clip_score.py im256. CLIP 단계 FAIL = CLIP SKIPPED(앞 결과 영향 없음)." % (D6_CFGS, D6_N),
         "- D9(N=%d, 세션 %s): 구성 %s. 부분집합 = metadata.csv에서 앞 %d행을 뺀 나머지 중 sha256(시드 %d + file_name)이 작은 %d행 (bridge/make_subset.py, 행 번호는 meta_n200.csv.idx.json). "
         "PSNR 기준선(BASE) = B0 im512를 파일 이름 순으로 %d칸 순환 이동해 짝지은 서로 다른 캡션 쌍. REG = D5와 같은 앞 2행 B0 재현 값. "
         "CLIP: clip_<구성> = 원본 평균, clippi_<구성> = 같은 계산의 이미지별 점수." % (D6_N, D9_TAG, [c for c, _ in D6_CFGS], SUB_SKIP, SUB_SEED, D6_N, D6_N // 2),
         "- Edge 호출별 지연은 측정 코드 없음. info_B0의 edge_call_upper_s = B0 sec_per_image ÷ 26 = Edge 호출 지연 상한(전체 파이프라인 포함: "
         "텍스트 인코딩·VAE·리사이즈·저장 포함).",
         "- 같은 세션에서 Run All을 두 번 누르지 말 것 (실행 중인 GPU 프로세스와 겹침).", "",
         "| 단계 | 상태 | 비고 |", "|---|---|---|"]
    for s in STEPS:
        st = state.get(s, {})
        L.append("| %s | %s | %s |" % (s, st.get("status", "-"), st.get("note", "")))
    for s in STEPS:
        for r in state.get(s, {}).get("cmds", []):
            L += ["", "## %s %s — rc=%s, %ss" % (s, r["name"], r["rc"], r["secs"]), "```", r["cmd"][:1500], "```"]
            if r["d5json"]:
                L += ["측정 줄:", "```"] + r["d5json"][:60] + ["```"]
            L += ["끝 40줄 (전체 `%s`):" % r["log"], "```"] + r["tail"] + ["```"]
    L += ["", "---", "이 보고서를 그대로 Claude에게 붙여 달라."]
    return "\n".join(L)


def main():
    global CKPT
    t_start = time.time()
    p = argparse.ArgumentParser(description="D5 오케스트레이터 (단일 세션 S0~S7)")
    p.add_argument("--budget-min", "--budget_min", dest="budget_min", type=float, default=360,
                   help="전체 시간 예산(분). 넘으면 남은 명령을 실행하지 않음 (기본 %(default)s)")
    p.add_argument("--skip", default="", help="건너뛸 단계, 쉼표 구분 (예: S3,S4)")
    p.add_argument("--redo", default="", help="이전 OK여도 다시 할 단계, 쉼표 구분 (예: S1)")
    p.add_argument("--ckpt", default=CKPT_DEFAULT, help="병합 체크포인트 경로 (기본 %(default)s)")
    args, unknown = p.parse_known_args()       # 노트북 셀 실행 시 ipykernel 인자 무시
    CKPT = args.ckpt
    print("HERE=%s\nBR=%s" % (HERE, BR), flush=True)
    print("D9 tag=%s n=%d cfgs=%s subset=%s" % (D9_TAG, D6_N, ",".join(c for c, _ in D6_CFGS), META_SUB), flush=True)
    ok, pf = preflight(BR, CKPT, CKPT_DEFAULT, EXPECTED_SIZE)
    if not ok:
        print("PREFLIGHT FAIL: " + "\n  ".join(pf), flush=True)              # 보고서·상태 파일 없이 종료
        return 2
    print("PREFLIGHT OK\n  " + "\n  ".join(pf), flush=True)                 # 통과: sha 줄 보여 주고 기존 흐름
    skip = {s.strip().upper() for s in args.skip.split(",") if s.strip()}
    redo = {s.strip().upper() for s in args.redo.split(",") if s.strip()}
    os.makedirs(LOGS, exist_ok=True)
    sp = OUT + "/d9_state.json"
    try:
        state = json.load(open(sp))
    except (OSError, ValueError):
        state = {}
    env = dict(os.environ)
    try:
        from kaggle_secrets import UserSecretsClient   # HF 토큰은 환경변수로만 전달 (명령 문자열·로그에 안 남김)
        env.setdefault("HF_TOKEN", UserSecretsClient().get_secret("HF_TOKEN"))
    except Exception:  # noqa: BLE001 — SD v1.4는 gated 아님, 토큰 없어도 진행
        pass
    steps = plan()
    core_fail = mobile_fail = None
    for s in STEPS:
        prev = str(state.get(s, {}).get("status", ""))
        if s in skip:
            state[s] = {"status": "SKIPPED(--skip)", "cmds": []}
        elif core_fail and s not in ("S7", "PACK"):
            state[s] = {"status": "SKIPPED(전 단계 실패: %s)" % core_fail, "cmds": []}
        elif mobile_fail and s == "S8":
            state[s] = {"status": "SKIPPED(S1 실패)", "cmds": []}
        elif (s != "S0" and prev.startswith("OK") and "SKIPPED" not in prev and "FAIL" not in prev
              and s not in redo):
            state[s]["status"] = "OK(이전 실행 재사용)"
            state[s]["note"] = "이전 실행 재사용(이번 실행에서 실행 안 됨)"
        else:
            print("[D5 %s] %s 시작" % (datetime.datetime.now().strftime("%H:%M:%S"), s), flush=True)
            cmds = []
            for name, cmd, to, allow in steps[s]:
                if mobile_fail and name in MOBILE:
                    cmds.append({"name": name, "cmd": cmd, "rc": "SKIPPED(%s 실패)" % mobile_fail, "secs": 0, "log": "-", "tail": [], "d5json": [],
                                 "allow_fail": True})
                    continue
                left = args.budget_min * 60 - (time.time() - t_start)
                if left <= 0:
                    cmds.append({"name": name, "cmd": cmd, "rc": "budget", "secs": 0, "log": "-", "tail": [], "d5json": []})
                    break
                r = run(s, name, cmd, min(to, int(left)), env)
                r["allow_fail"] = allow
                cmds.append(r)
                print("[D5] %s %s rc=%s %ss" % (s, name, r["rc"], r["secs"]), flush=True)
                if r["rc"] != 0 and not allow and s == "S0":   # S0(복구)만 앞 명령 실패 시 중단, 나머지 단계는 항목별로 끝까지
                    break
            bad = [c["name"] for c in cmds if c["rc"] != 0 and not c.get("allow_fail")]
            if s == "S1" and any("_OOM" in l for c in cmds for l in c["d5json"]):
                bad.append("OOM — 측정값만 남기고 정지")
            sk = [c for c in cmds if str(c["rc"]).startswith("SKIPPED")]   # 원인 단계(S1 또는 D6 첫 Q 구성)를 표시, 전부/일부 구분
            part = (" / %s SKIPPED(%s 실패)" % ("전부" if len(sk) == len(cmds) else "일부", mobile_fail)) if sk else ""
            state[s] = {"status": ("FAIL(%s)" % ",".join(bad) if bad else "OK(rc=0)") + part, "cmds": cmds,
                        "note": "rc 기준. 동등성·Gate 판정은 수치를 보고 사람이 한다" if s in ("S5", "S6", "S8") else ""}
        if s in CORE and str(state[s]["status"]).startswith("FAIL"):
            core_fail = s
        if s == "S1" and str(state[s]["status"]).startswith("FAIL"):   # Mobile 필요 항목만 SKIPPED, 나머지는 계속
            mobile_fail = s
        if s == D6_Q[0] and str(state[s]["status"]).startswith("FAIL") and not os.path.exists(OUT + "/d9_%s/slot.json" % s):
            mobile_fail = s                                             # slot.json은 MobileSlot 생성 후에만 생김 (run_split.py 마지막의 SLOT[0].dump()) → 없으면 Mobile 로드 실패 계열
        json.dump(state, open(sp, "w"), ensure_ascii=False, indent=1)
        meta = {"time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "argv": " ".join(sys.argv[1:]),   # 단계(구성)마다 보고서 갱신
                "unknown": " ".join(unknown), "budget": args.budget_min, "elapsed": (time.time() - t_start) / 60}
        text = report(state, meta)
        open(OUT + "/d9_report.md", "w", encoding="utf-8").write(text + "\n")
    tl = text.splitlines()                                           # 전체 보고서(수백 KB)는 파일로만, 화면에는 단계 표까지만
    i0 = tl.index("| 단계 | 상태 | 비고 |")
    print("\\n".join(tl[i0:i0 + 2 + len(STEPS)]) + "\\n보고서 전체: %s/d9_report.md (%d바이트)" % (OUT, len(text)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

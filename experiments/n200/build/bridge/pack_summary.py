# bridge/pack_summary.py — 세션 결과를 사람이 올리기 쉬운 파일 2개로 묶는다. 계산·판정은 하지 않고 파일 내용을 그대로 모은다.
#   <ROOT>/d9_result_<TAG>.json : 분석에 필요한 숫자 전부(이미지별 PSNR·CLIP, 호출 수, 지연, 부분집합 행 번호, 단계 상태)
#   <ROOT>/d9_summary_<TAG>.zip : 위 JSON + 보고서 + 상태 + logs + 구성별 작은 파일 (이미지는 넣지 않음)
# 사용: python pack_summary.py <OUT 폴더> <TAG> <N> <구성들(쉼표)>      (ROOT = OUT의 상위 폴더)
# 일부 항목이 없어도 멈추지 않고 그 항목에 "error"를 적는다.
import glob, hashlib, json, os, re, statistics, subprocess, sys, time, zipfile
from collections import Counter
from importlib import metadata


def rd(path):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return {"error": "%s: %r" % (path, e)}


def sha_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def versions():
    out = {"python": sys.version.split()[0]}
    for n in ("torch", "torchvision", "diffusers", "transformers", "open_clip_torch", "numpy", "Pillow", "huggingface_hub", "peft"):
        try:
            out[n] = metadata.version(n)
        except Exception:  # noqa: BLE001
            out[n] = None
    return out


def gpu_info():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=60)
        return r.stdout.strip().splitlines()
    except Exception as e:  # noqa: BLE001
        return ["error: %r" % (e,)]


def routing_counts(log_path):
    try:
        text = open(log_path, encoding="utf-8", errors="replace").read()
    except OSError as e:
        return {"error": repr(e)}
    return {"unet_model_lines": dict(Counter("%s %s" % m for m in re.findall(r"Unet Model is:(\S+) (\d+)", text))),
            "len_timesteps_lines": dict(Counter(re.findall(r"len\(timesteps\)=\d+, num_inference_steps=\d+", text))),
            "invalid_value": text.count("invalid value")}


def slot_summary(path):
    c = rd(path)
    if "error" in c:
        return c
    ms = [x["ms"] for x in c["calls"]]
    return {"mobile_calls": len(ms), "ms_mean": statistics.mean(ms), "ms_pstdev": statistics.pstdev(ms), "ms_min": min(ms), "ms_max": max(ms),
            "k_match": sum(x["k_formula"] in x["k_loaded"] for x in c["calls"]),
            "k_loaded_len_not_1": sum(len(x["k_loaded"]) != 1 for x in c["calls"]),
            "n_act": c.get("n_act"), "max_memory_allocated": c.get("max_memory_allocated"), "ms_first_26": ms[:26]}


def clip_txt(path):
    try:
        t = open(path, encoding="utf-8").read().splitlines()
        return {"final": float(t[0].split()[-1]), "sum": float(t[1].split()[-1]), "len": int(t[2].split()[-1])}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def main(argv):
    out_dir, tag, n, cfgs = argv[0], argv[1], int(argv[2]), argv[3].split(",")
    root = os.path.dirname(os.path.normpath(out_dir))
    state = rd(os.path.join(out_dir, "d9_state.json"))
    cfg_res = {}
    for c in cfgs:
        d = os.path.join(out_dir, "d9_" + c)
        r = {"latency_s_per_img": rd(os.path.join(d, "model_info.json")).get("latency"),
             "n_im512": len(glob.glob(os.path.join(d, "im512", "*"))), "n_im256": len(glob.glob(os.path.join(d, "im256", "*"))),
             "routing": routing_counts(os.path.join(out_dir, "logs", "%s_gen_%s.log" % (c, c))),
             "clip_original": clip_txt(os.path.join(d, "clip.txt")), "clip_per_image": rd(os.path.join(d, "clip_per_image.json"))}
        if c != "B0":
            r["slot"] = slot_summary(os.path.join(d, "slot.json"))
            r["psnr_vs_B0"] = rd(os.path.join(d, "psnr_vs_B0.json"))
        else:
            r["im512_sha256"] = {os.path.basename(p): sha_file(p) for p in sorted(glob.glob(os.path.join(d, "im512", "*")))}
        cfg_res[c] = r
    base = {"psnr_unrelated": rd(os.path.join(out_dir, "d9_base", "psnr_unrelated.json")), "pairs": rd(os.path.join(out_dir, "d9_base", "pairs.json"))}
    st = {}
    for s, v in (state.items() if "error" not in state else []):
        st[s] = {"status": v.get("status"), "note": v.get("note"),
                 "cmds": [{"name": x["name"], "rc": x["rc"], "secs": x["secs"],        # check_img의 이미지별 줄(shape_a 포함)은 PSNR 파일에 있으므로 뺀다
                   "d5json": [l for l in x.get("d5json", []) if '"shape_a"' not in l]} for x in v.get("cmds", [])]}
    res = {"meta": {"tag": tag, "n": n, "cfgs": cfgs, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "versions": versions(), "gpu": gpu_info()},
           "subset": rd(os.path.join(out_dir, "meta_n200.csv.idx.json")), "configs": cfg_res, "base": base, "state": st, "state_error": state.get("error")}
    res_path = os.path.join(root, "d9_result_%s.json" % tag)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(res, f)
    zip_path = os.path.join(root, "d9_summary_%s.zip" % tag)
    keep = [res_path] + [os.path.join(out_dir, x) for x in ("d9_report.md", "d9_state.json", "meta_n200.csv", "meta_n200.csv.idx.json")]
    keep += sorted(glob.glob(os.path.join(out_dir, "logs", "*.log")))
    for sub in ["d9_" + c for c in cfgs] + ["d9_reg", "d9_base"]:
        for fn in ("model_info.json", "slot.json", "clip.txt", "clip_per_image.json", "psnr_vs_B0.json", "psnr_unrelated.json", "pairs.json"):
            keep.append(os.path.join(out_dir, sub, fn))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in keep:
            if os.path.isfile(p):
                z.write(p, os.path.relpath(p, root))
    print("D5JSON:" + json.dumps(dict(result=res_path, result_bytes=os.path.getsize(res_path), zip=zip_path, zip_bytes=os.path.getsize(zip_path),
                                      n_in_zip=len(zipfile.ZipFile(zip_path).namelist()), cfgs=cfgs)))


if __name__ == "__main__":
    main(sys.argv[1:])

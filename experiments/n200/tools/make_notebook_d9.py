#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tools/make_notebook_d9.py — build/ 의 파일들을 EMBED 한 줄로 묶어 Kaggle용 셀(.py)과 노트북(.ipynb)을 만든다.
# 사용: python3 tools/make_notebook_d9.py <n200 루트> <TAG> <구성(쉼표, B0가 맨 앞)> [N=200]
#   예:  python3 tools/make_notebook_d9.py . A B0,Q20,Q25          →  kaggle/d9_cell_A.py, kaggle/d9_notebook_A.ipynb
#        python3 tools/make_notebook_d9.py . B B0,Q5,Q10,Q15
import ast, base64, hashlib, json, os, sys

ROOT, TAG, CFGS = sys.argv[1], sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 200
BUILD, OUT = os.path.join(ROOT, "build"), os.path.join(ROOT, "kaggle")
ALL = ("B0", "Q5", "Q10", "Q15", "Q20", "Q25")
cfgs = CFGS.split(",")
assert cfgs[0] == "B0" and len(cfgs) > 1 and all(c in ALL for c in cfgs) and len(set(cfgs)) == len(cfgs), cfgs
files = ["d9_kaggle.py"] + ["bridge/" + f for f in sorted(os.listdir(os.path.join(BUILD, "bridge"))) if f.endswith(".py")]
assert len(files) == 8, files                                   # d9_kaggle.py + bridge 7개
EMBED = {}
for rel in files:
    raw = open(os.path.join(BUILD, rel), "rb").read()
    EMBED[rel] = (hashlib.sha256(raw).hexdigest(), base64.b64encode(raw).decode("ascii"))
wrapper = open(os.path.join(BUILD, "d9_wrapper.py"), encoding="utf-8").read()
head = ("# 이 셀은 tools/make_notebook_d9.py가 만들었다. 세션 %s: 구성 %s, N=%d. EMBED = {경로: (sha256, base64)}\n" % (TAG, CFGS, N)
        + "EMBED = %r\n" % (EMBED,) + "D9_TAG = %r\nD9_CFGS = %r\nD9_N = %d\n" % (TAG, CFGS, N))
cell = head + wrapper
ast.parse(cell)
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "d9_cell_%s.py" % TAG), "w", encoding="utf-8", newline="\n") as f:
    f.write(cell)
md = ("# D9 세션 %s — N=%d, 구성 %s\n\n"
      "1. 오른쪽 Settings에서 **Accelerator = GPU T4 x2**, **Internet = On**. 데이터셋 `dgq-sd14-w4a8g1-merged`를 Add Data로 붙인다.\n"
      "2. 아래 셀 하나를 실행한다(Run All을 두 번 누르지 않는다). 끝나면 `/kaggle/working`의 `d9_result_%s.json`, `d9_summary_%s.zip`, "
      "`d9_out/d9_report.md`를 내려받는다.\n"
      "3. 도중에 끊기면 **같은 세션에서** 이 셀을 다시 실행한다. 끝난 단계는 건너뛴다.\n") % (TAG, N, CFGS, TAG, TAG)
nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": md.splitlines(True)},
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell.splitlines(True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}
nbp = os.path.join(OUT, "d9_notebook_%s.ipynb" % TAG)
with open(nbp, "w", encoding="utf-8", newline="\n") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
for p in (os.path.join(OUT, "d9_cell_%s.py" % TAG), nbp):
    print("%s  %s  (%d bytes)" % (hashlib.sha256(open(p, "rb").read()).hexdigest(), os.path.relpath(p, ROOT), os.path.getsize(p)))

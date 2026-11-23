# 이미지 두 장(또는 두 폴더의 같은 이름 파일) 비교: 최대 픽셀 차, PSNR, 각 이미지 평균·표준편차. 판정은 하지 않고 수치만 출력.
# 사용: python check_img.py <A 파일|폴더> <B 파일|폴더>
import json, math, os, sys
import numpy as np
from PIL import Image
a, b = sys.argv[1], sys.argv[2]
if os.path.isdir(a):
    pairs = [(os.path.join(a, f), os.path.join(b, f)) for f in sorted(os.listdir(a)) if os.path.exists(os.path.join(b, f))]
else:
    pairs = [(a, b)]
for pa, pb in pairs:
    x = np.asarray(Image.open(pa).convert("RGB")).astype(np.float64)
    y = np.asarray(Image.open(pb).convert("RGB")).astype(np.float64)
    mse = float(((x - y) ** 2).mean()) if x.shape == y.shape else None
    psnr = None if mse is None else (float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse))
    print("D5JSON:" + json.dumps(dict(a=pa, b=pb, shape_a=x.shape, shape_b=y.shape, mse=mse, psnr=psnr,
          max_abs=None if mse is None else float(np.abs(x - y).max()),
          mean_a=float(x.mean()), std_a=float(x.std()), mean_b=float(y.mean()), std_b=float(y.std()))))
print("D5JSON:" + json.dumps(dict(pairs=len(pairs))))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 전송량 민감도. (A) 손익분기 활용률 u*, (B) M/D/1 교차 부하 lam*.
# 표준 라이브러리만 사용. 최적 N_M을 찾지 않는다.
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys

# ===== 상수 (출처) =====
E = 0.179          # T4에서 측정한 서버 U-Net 호출당 지연 상한 [s]. u*에서 E로 사용
TAU = 0.179        # E와 같은 값 [s]. lam*에서 tau로 사용
T4_TM = 1.128      # T4 fake-quant 모바일 호출 시간 [s]
F = 1357066539008  # U-Net 호출당 FLOPs, CFG 배치 2
N_TOTAL = 26       # PNDM 25스텝 = U-Net 26회
NM_LIST = [6, 11, 16, 21, 26]  # 분할 구성 N_M. B0(N_M=0)는 기준
DEV = [("RPi5", 13e12), ("Jetson", 100e12)]  # 기기 연산 성능 [OPS] (Splitwise Table 2의 프로필)
LINK = [("WiFi", 100e6, 0.010), ("5G", 25e6, 0.040), ("4G", 10e6, 0.080)]  # (B [bps], l_n [s]) (Splitwise Table 3의 프로필)
LAT = 4 * 64 * 64 * 2  # FP16 잠재 텐서 [B]
EMB = 77 * 768 * 2     # FP16 텍스트 임베딩 [B]
IMG = 512 * 512 * 3    # 8비트 RGB 이미지 [B]
PAYLOAD = [("PA", 0), ("PB", 3 * LAT), ("PC", EMB), ("PD", 3 * LAT + EMB)]
SCENARIOS = [("r2.0", 2.0 * TAU), ("r2.2", 2.2 * TAU), ("T4", T4_TM)]  # t_M 시나리오 [s] (r2.0·r2.2는 가정값, T4는 예비 실험 측정값)

DEVD = dict(DEV)
LINKD = {n: (b, l) for n, b, l in LINK}
EXTRA = dict(PAYLOAD)
PNAMES = [p for p, _ in PAYLOAD]
OUT_FILES = ["o3_comm.csv", "o3_ustar.csv", "o3_lamstar.csv", "o3_summary.json"]
HDR_COMM = ["link", "payload", "extra_bytes", "nonimg_bytes", "split_bytes", "comm_split_s", "comm_b0_s", "delta_vs_PA_s"]
HDR_USTAR = ["device", "link", "payload", "nm", "ts_s", "ustar_pct", "delta_pp_vs_PA", "feasible"]
HDR_LAM = ["link", "scenario", "payload", "nm", "tm_s", "lam_star", "lam_max_b0", "ratio_pct", "delta_pp_vs_PA"]

REF_C = """
WiFi PA: split=0.087619 b0=0.072915
WiFi PB: split=0.095484 b0=0.072915
WiFi PC: split=0.097081 b0=0.072915
WiFi PD: split=0.104945 b0=0.072915
5G PA: split=0.350477 b0=0.291658
5G PB: split=0.381934 b0=0.291658
5G PC: split=0.388324 b0=0.291658
5G PD: split=0.419781 b0=0.291658
4G PA: split=0.856192 b0=0.709146
4G PB: split=0.934835 b0=0.709146
4G PC: split=0.950810 b0=0.709146
4G PD: split=1.029453 b0=0.709146
"""
REF_U = """
RPi5 WiFi PA: 59.1278 58.7571 58.6193 58.5473 58.5031
RPi5 WiFi PB: 59.5701 58.9945 58.7815 58.6705 58.6025
RPi5 WiFi PC: 59.6607 59.0429 58.8146 58.6956 58.6227
RPi5 WiFi PD: 60.1110 59.2827 58.9779 58.8195 58.7224
RPi5 5G PA: 61.6972 60.1140 59.5411 59.2453 59.0648
RPi5 5G PB: 63.6701 61.1206 60.2164 59.7533 59.4719
RPi5 5G PC: 64.0864 61.3292 60.3554 59.8576 59.5553
RPi5 5G PD: 66.2177 62.3772 61.0494 60.3762 59.9692
RPi5 4G PA: 67.5696 63.0250 61.4746 60.6925 60.2210
RPi5 4G PB: 73.8336 65.8682 63.3070 62.0434 61.2905
RPi5 4G PC: 75.2507 66.4773 63.6927 62.3251 61.5124
RPi5 4G PD: 83.1026 69.6483 65.6618 63.7505 62.6286
Jetson WiFi PA: 7.6866 7.6384 7.6205 7.6112 7.6054
Jetson WiFi PB: 7.7441 7.6693 7.6416 7.6272 7.6183
Jetson WiFi PC: 7.7559 7.6756 7.6459 7.6304 7.6209
Jetson WiFi PD: 7.8144 7.7067 7.6671 7.6465 7.6339
Jetson 5G PA: 8.0206 7.8148 7.7403 7.7019 7.6784
Jetson 5G PB: 8.2771 7.9457 7.8281 7.7679 7.7313
Jetson 5G PC: 8.3312 7.9728 7.8462 7.7815 7.7422
Jetson 5G PD: 8.6083 8.1090 7.9364 7.8489 7.7960
Jetson 4G PA: 8.7840 8.1933 7.9917 7.8900 7.8287
Jetson 4G PB: 9.5984 8.5629 8.2299 8.0656 7.9678
Jetson 4G PC: 9.7826 8.6421 8.2800 8.1023 7.9966
Jetson 4G PD: 10.8033 9.0543 8.5360 8.2876 8.1417
"""
REF_L = """
WiFi r2.0 PA: 47.7755 52.4432 57.3741 62.2335 66.7367
WiFi r2.0 PB: 47.9294 52.5334 57.4383 62.2820 66.7741
WiFi r2.0 PC: 47.9605 52.5517 57.4513 62.2918 66.7817
WiFi r2.0 PD: 48.1131 52.6415 57.5152 62.3401 66.8189
WiFi r2.2 PA: 51.6125 56.5406 61.5769 66.3630 70.6428
WiFi r2.2 PB: 51.7399 56.6146 61.6287 66.4014 70.6719
WiFi r2.2 PC: 51.7657 56.6295 61.6392 66.4092 70.6778
WiFi r2.2 PD: 51.8921 56.7032 61.6907 66.4475 70.7068
WiFi T4 PA: 78.3567 83.4093 87.0551 89.5948 91.3865
WiFi T4 PB: 78.3759 83.4189 87.0608 89.5985 91.3890
WiFi T4 PC: 78.3798 83.4209 87.0620 89.5992 91.3895
WiFi T4 PD: 78.3990 83.4305 87.0677 89.6029 91.3920
5G r2.0 PA: 48.6246 52.9445 57.7314 62.5038 66.9452
5G r2.0 PB: 49.2095 53.2949 57.9824 62.6941 67.0922
5G r2.0 PC: 49.3263 53.3653 58.0330 62.7326 67.1219
5G r2.0 PD: 49.8917 53.7088 58.2803 62.9206 67.2674
5G r2.2 PA: 52.3167 56.9519 61.8654 66.5773 70.8053
5G r2.2 PB: 52.8037 57.2400 62.0684 66.7284 70.9200
5G r2.2 PC: 52.9011 57.2980 62.1094 66.7589 70.9432
5G r2.2 PD: 53.3736 57.5810 62.3098 66.9084 71.0569
5G T4 PA: 78.4639 83.4632 87.0870 89.6153 91.4005
5G T4 PB: 78.5395 83.5013 87.1096 89.6299 91.4105
5G T4 PC: 78.5548 83.5091 87.1142 89.6328 91.4125
5G T4 PD: 78.6298 83.5470 87.1367 89.6474 91.4225
4G r2.0 PA: 50.2243 53.9128 58.4277 63.0328 67.3543
4G r2.0 PB: 51.5503 54.7395 59.0286 63.4919 67.7106
4G r2.0 PC: 51.8091 54.9034 59.1484 63.5838 67.7820
4G r2.0 PD: 53.0351 55.6918 59.7281 64.0293 68.1290
4G r2.2 PA: 53.6523 57.7493 62.4293 66.9977 71.1249
4G r2.2 PB: 54.7680 58.4333 62.9176 67.3637 71.4040
4G r2.2 PC: 54.9867 58.5693 63.0152 67.4370 71.4600
4G r2.2 PD: 56.0266 59.2247 63.4881 67.7933 71.7326
4G T4 PA: 78.6746 83.5698 87.1503 89.6561 91.4285
4G T4 PB: 78.8586 83.6636 87.2061 89.6921 91.4532
4G T4 PC: 78.8955 83.6825 87.2174 89.6994 91.4583
4G T4 PD: 79.0752 83.7751 87.2727 89.7352 91.4828
"""

NOTES = """NOTES
- All delay and load values are model outputs, not measurements. t_M is not measured on a mobile device.
- Payload cases PB, PC, PD are sensitivity cases (PB: 3 FP16 solver-state tensors, PC: unconditional text embedding sent every time, PD: both). They are not measured payloads.
- The same B and l_n are used for both directions. No retransmission, congestion, or mobile queue.
- Server: single server, FIFO, Poisson arrivals, deterministic service, no batching. tau = 0.179 s is a measured upper bound on T4.
- u* is a break-even value without server queueing (lambda -> 0). lam* is the load above which the split has the shorter mean end-to-end delay. Neither is a recommendation."""


def nonimg_bytes(extra):
    return EMB + 2 * LAT + extra  # X(p)


def split_bytes(extra):
    return nonimg_bytes(extra) + IMG


def comm(nbytes, b, ln):
    return 8.0 * nbytes / b + ln  # l_n 포함


def tm_star(x, b, nm):
    return E - x * 8.0 / (b * nm)


def ustar_pct(c, ts):
    return 100.0 * F / (c * ts) if ts > 0 else None


def t_noqueue(nm, tm, comm_s):
    return nm * tm + (N_TOTAL - nm) * E + comm_s


def wq(lam, s):
    if s == 0:
        return 0.0
    rho = lam * s
    if rho >= 1.0:
        return math.inf
    return rho * s / (2.0 * (1.0 - rho))


def t_queue(lam, nm, tm, comm_s):
    s = (N_TOTAL - nm) * TAU
    return wq(lam, s) + s + nm * tm + comm_s


def lam_max_b0():
    return 1.0 / (N_TOTAL * TAU)


def solve_lam(nm, tm, cs, cb):
    def f(lam):
        return t_queue(lam, 0, 0.0, cb) - t_queue(lam, nm, tm, cs)
    lmax = lam_max_b0()
    lo, hi = 0.0, lmax * (1.0 - 1e-12)
    g = [f(lmax * 0.999999 * i / 400) for i in range(401)]
    r = {"f0_neg": f(lo) < 0, "fend_pos": f(hi) > 0, "lam": None, "resid": None,
         "mono": all(g[i + 1] >= g[i] - 1e-12 for i in range(400))}
    if r["f0_neg"] and r["fend_pos"]:
        for _ in range(200):
            mid = (lo + hi) / 2.0
            if f(mid) < 0:
                lo = mid
            else:
                hi = mid
        r["lam"] = (lo + hi) / 2.0
        r["resid"] = abs(f(r["lam"]))
    return r


def compute():
    cm, us, ls = {}, {}, {}
    for ln, b, l in LINK:
        for p, ex in PAYLOAD:
            cm[(ln, p)] = {"cs": comm(split_bytes(ex), b, l), "cb": comm(IMG, b, l)}
    for d, c in DEV:
        for ln, b, l in LINK:
            for p, ex in PAYLOAD:
                for nm in NM_LIST:
                    ts = tm_star(nonimg_bytes(ex), b, nm)
                    us[(d, ln, p, nm)] = {"ts": ts, "u": ustar_pct(c, ts)}
    for ln, b, l in LINK:
        for s, tm in SCENARIOS:
            for p, ex in PAYLOAD:
                for nm in NM_LIST:
                    r = solve_lam(nm, tm, cm[(ln, p)]["cs"], cm[(ln, p)]["cb"])
                    r["tm"] = tm
                    r["ratio"] = None if r["lam"] is None else 100.0 * r["lam"] / lam_max_b0()
                    ls[(ln, s, p, nm)] = r
    return cm, us, ls


def diff(a, b):
    return None if a is None or b is None else a - b


# ===== 출력 행 =====
def rows_comm(cm):
    return [[ln, p, EXTRA[p], nonimg_bytes(EXTRA[p]), split_bytes(EXTRA[p]), cm[(ln, p)]["cs"], cm[(ln, p)]["cb"],
             cm[(ln, p)]["cs"] - cm[(ln, "PA")]["cs"]] for ln, _, _ in LINK for p in PNAMES]


def rows_ustar(us):
    return [[d, ln, p, nm, us[(d, ln, p, nm)]["ts"], us[(d, ln, p, nm)]["u"],
             diff(us[(d, ln, p, nm)]["u"], us[(d, ln, "PA", nm)]["u"]), 0 if us[(d, ln, p, nm)]["u"] is None else 1]
            for d, _ in DEV for ln, _, _ in LINK for p in PNAMES for nm in NM_LIST]


def rows_lam(ls):
    return [[ln, s, p, nm, ls[(ln, s, p, nm)]["tm"], ls[(ln, s, p, nm)]["lam"], lam_max_b0(), ls[(ln, s, p, nm)]["ratio"],
             diff(ls[(ln, s, p, nm)]["ratio"], ls[(ln, s, "PA", nm)]["ratio"])]
            for ln, _, _ in LINK for s, _ in SCENARIOS for p in PNAMES for nm in NM_LIST]


def cell(v):
    if v is None:
        return ""
    return "%.6f" % v if isinstance(v, float) else str(v)


def write_csv(path, header, rows):
    with open(path, "x", newline="", encoding="utf-8") as fh:  # "x": 기존 파일 덮어쓰기 금지
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow([cell(v) for v in r])


def same_cell(text, v):
    if v is None:
        return text == ""
    if isinstance(v, float):
        try:
            return abs(float(text) - v) <= 1e-6
        except ValueError:
            return False
    return text == str(v)


# ===== 검증 =====
def chk(checks, name, ok, detail=""):
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def chk_list(checks, name, fails):
    chk(checks, name, not fails, "; ".join(fails))


def parse_ref(text):
    return {tuple(k.split()): v.split() for k, v in (ln.split(":") for ln in text.strip().splitlines())}


def close(v, ref, tol):
    return v is not None and abs(v - ref) <= tol


def checks_ab(checks, cm):
    chk(checks, "A.LAT=32768", LAT == 32768)
    chk(checks, "A.EMB=118272", EMB == 118272)
    chk(checks, "A.IMG=786432", IMG == 786432)
    chk(checks, "A.extra_PA..PD", [EXTRA[p] for p in PNAMES] == [0, 98304, 118272, 216576])
    chk(checks, "A.extra_PD=PB+PC", EXTRA["PD"] == EXTRA["PB"] + EXTRA["PC"])
    chk(checks, "A.X_PA..PD", [nonimg_bytes(EXTRA[p]) for p in PNAMES] == [183808, 282112, 302080, 400384])
    chk(checks, "A.split_bytes_PA..PD", [split_bytes(EXTRA[p]) for p in PNAMES] == [970240, 1068544, 1088512, 1186816])
    for d, fc, u0 in [("RPi5", 0.104390, 58.3183), ("Jetson", 0.013571, 7.5814)]:
        chk(checks, "A.F_over_Ce_" + d, close(F / DEVD[d], fc, 1e-6))
        chk(checks, "A.u0_" + d, close(100.0 * F / (DEVD[d] * E), u0, 1e-3))
    d7 = {"WiFi": (0.08762, 0.07291), "5G": (0.35048, 0.29166), "4G": (0.85619, 0.70915)}
    for ln, _, _ in LINK:
        chk(checks, "B.comm_PA_D7_" + ln, close(cm[(ln, "PA")]["cs"], d7[ln][0], 1e-5) and close(cm[(ln, "PA")]["cb"], d7[ln][1], 1e-5))
    chk_list(checks, "B.comm_delta_identity", ["%s %s" % (ln, p) for ln, b, _ in LINK for p in PNAMES
             if abs((cm[(ln, p)]["cs"] - cm[(ln, "PA")]["cs"]) - EXTRA[p] * 8.0 / b) > 1e-12])
    ref = parse_ref(REF_C)
    chk_list(checks, "B.table_C_12", ["%s %s" % k for k, v in ref.items() if not (
        close(cm[k]["cs"], float(v[0].split("=")[1]), 1e-6) and close(cm[k]["cb"], float(v[1].split("=")[1]), 1e-6))])


def checks_c(checks, cm, us):
    ref = parse_ref(REF_U)
    chk_list(checks, "C.table_U_120", ["%s %s %s N_M=%d" % (k + (nm,)) for k, v in ref.items()
             for nm, x in zip(NM_LIST, v) if not close(us[k + (nm,)]["u"], float(x), 1e-3)])
    chk_list(checks, "C.tmstar_positive", ["%s %s %s N_M=%d" % k for k, r in us.items() if not r["ts"] > 0])
    chk_list(checks, "C.breakeven_identity", ["%s %s %s N_M=%d" % k for k, r in us.items() if not abs(
        t_noqueue(k[3], r["ts"], cm[(k[1], k[2])]["cs"]) - t_noqueue(0, 0.0, cm[(k[1], k[2])]["cb"])) < 1e-9])
    chk_list(checks, "C.ustar_ge_u0", ["%s %s %s N_M=%d" % k for k, r in us.items()
             if r["u"] is None or r["u"] < 100.0 * F / (DEVD[k[0]] * E)])
    seq = {}
    for k, r in us.items():
        seq.setdefault(k[:3], []).append(r["u"])
    chk_list(checks, "C.ustar_nonincreasing_in_nm", ["%s %s %s" % k for k, v in seq.items()
             if None in v or any(v[i + 1] > v[i] for i in range(len(v) - 1))])
    chk_list(checks, "C.ustar_strict_PA<PB<PC<PD", ["%s %s N_M=%d" % (d, ln, nm) for d, _ in DEV for ln, _, _ in LINK
             for nm in NM_LIST if not strictly_up([us[(d, ln, p, nm)]["u"] for p in PNAMES])])
    for d, ln, p, v in [("RPi5", "4G", "PA", 67.6), ("RPi5", "4G", "PD", 83.1), ("Jetson", "4G", "PA", 8.8),
                        ("Jetson", "4G", "PD", 10.8), ("RPi5", "5G", "PA", 61.7), ("RPi5", "5G", "PD", 66.2)]:
        u = us[(d, ln, p, 6)]["u"]
        chk(checks, "C.ref_%s_%s_N_M=6_%s=%.1f" % (d, ln, p, v), u is not None and abs(round(u, 1) - v) < 1e-9)


def strictly_up(v):
    return None not in v and all(v[i] < v[i + 1] for i in range(len(v) - 1))


def checks_d(checks, ls):
    ref = parse_ref(REF_L)
    chk_list(checks, "D.table_L_180", ["%s %s %s N_M=%d" % (k + (nm,)) for k, v in ref.items()
             for nm, x in zip(NM_LIST, v) if not close(ls[k + (nm,)]["ratio"], float(x), 1e-3)])
    for key, name in [("f0_neg", "D.f0_negative"), ("fend_pos", "D.f_end_positive"), ("mono", "D.grid_nondecreasing")]:
        chk_list(checks, name, ["%s %s %s N_M=%d" % k for k, r in ls.items() if not r[key]])
    chk_list(checks, "D.cross_identity", ["%s %s %s N_M=%d" % k for k, r in ls.items()
             if r["resid"] is None or not r["resid"] < 1e-9])
    chk_list(checks, "D.ratio_strict_PA<PB<PC<PD", ["%s %s N_M=%d" % (ln, s, nm) for ln, _, _ in LINK
             for s, _ in SCENARIOS for nm in NM_LIST if not strictly_up([ls[(ln, s, p, nm)]["ratio"] for p in PNAMES])])
    r = ls[("5G", "r2.2", "PA", 6)]
    chk(checks, "D.O2_ref_5G_r2.2_N_M=6_PA", r["lam"] is not None and abs(round(r["ratio"], 1) - 52.3) < 1e-9
        and abs(round(r["lam"], 4) - 0.1124) < 1e-9)


def hand_u(dev, ln, p, nm):
    c, (b, _) = DEVD[dev], LINKD[ln]
    x = nonimg_bytes(EXTRA[p])
    x8b = x * 8.0 / b
    per = x8b / nm
    ts = E - per
    fc = F / c
    u = 100.0 * fc / ts
    lines = ["(%s %s N_M=%d %s)" % (dev, ln, nm, p),
             "  X = EMB + 2*LAT + extra = %d + 2*%d + %d = %d B" % (EMB, LAT, EXTRA[p], x),
             "  X*8/B = %d*8/%.0f = %.7f s" % (x, b, x8b),
             "  /N_M = %.7f/%d = %.7f s" % (x8b, nm, per),
             "  t_M* = E - X*8/(B*N_M) = %.3f - %.7f = %.7f s" % (E, per, ts),
             "  F/C_e = %d/%.0f = %.6f s" % (F, c, fc),
             "  u* = 100*(F/C_e)/t_M* = 100*%.6f/%.7f = %.4f %%" % (fc, ts, u)]
    return lines, {"X": x, "X*8/B": x8b, "/N_M": per, "t_M*": ts, "u*": u}


def hand_lam(ls, ln, s, p, nm):
    b, l = LINKD[ln]
    tm, lam = dict(SCENARIOS)[s], ls[(ln, s, p, nm)]["lam"]
    head = "(%s %s N_M=%d %s, s0=0, tau=%.3f)" % (ln, s, nm, p, TAU)
    if lam is None:
        return [head, "  lam* not available"], {}
    cb, cs = comm(IMG, b, l), comm(split_bytes(EXTRA[p]), b, l)
    sb, ss = N_TOTAL * TAU, (N_TOTAL - nm) * TAU
    tb = wq(lam, sb) + sb + cb
    tsp = wq(lam, ss) + ss + nm * tm + cs
    ratio = 100.0 * lam / lam_max_b0()
    lines = [head,
             "  lam* = %.6f img/s, lam_max(B0) = 1/(26*%.3f) = %.6f, ratio = %.4f %%" % (lam, TAU, lam_max_b0(), ratio),
             "  B0: S = 26*%.3f = %.6f, rho = %.6f, W_q = rho*S/(2*(1-rho)) = %.6f, comm = 8*%d/%.0f + %.3f = %.6f"
             % (TAU, sb, lam * sb, wq(lam, sb), IMG, b, l, cb),
             "  T_B0 = W_q + S + comm = %.6f s" % tb,
             "  split: S = %d*%.3f = %.6f, rho = %.6f, W_q = %.6f, N_M*t_M = %d*%.4f = %.6f, comm = 8*%d/%.0f + %.3f = %.6f"
             % (N_TOTAL - nm, TAU, ss, lam * ss, wq(lam, ss), nm, tm, nm * tm, split_bytes(EXTRA[p]), b, l, cs),
             "  T_split = W_q + S + N_M*t_M + comm = %.6f s" % tsp,
             "  T_B0 - T_split = %.3e s" % (tb - tsp)]
    return lines, {"lam*": lam, "ratio": ratio, "T_B0": tb, "T_split": tsp}


def hand_block(checks, ls):
    out = []
    refs = [(hand_u("RPi5", "4G", "PA", 6), "E.i", {"X": 183808, "X*8/B": 0.1470464, "/N_M": 0.0245077, "t_M*": 0.1544923, "u*": 67.5696}),
            (hand_u("RPi5", "4G", "PD", 6), "E.ii", {"X": 400384, "X*8/B": 0.3203072, "/N_M": 0.0533845, "t_M*": 0.1256155, "u*": 83.1026}),
            (hand_lam(ls, "5G", "r2.2", "PD", 6), "E.iii", {"lam*": 0.114683, "ratio": 53.3736, "T_B0": 7.609394, "T_split": 7.609394})]
    for (lines, vals), tag, ref in refs:
        out.extend(lines)
        for k, v in ref.items():
            chk(checks, "%s.%s" % (tag, k), close(vals.get(k), v, 1e-4))
    return out


def checks_f(checks, outdir, tables):
    for fname, header, rows, nrow in tables:
        with open(os.path.join(outdir, fname), newline="", encoding="utf-8") as fh:
            got = list(csv.reader(fh))
        body = got[1:]
        chk(checks, "F.%s.rows=%d" % (fname, nrow), len(body) == nrow == len(rows))
        chk(checks, "F.%s.header" % fname, got[:1] == [header])
        bad = []
        for i, g in enumerate(body):
            for j, t in enumerate(g):
                blank_ok = fname == "o3_ustar.csv" and j in (5, 6) and g[7] == "0"
                if t.strip().lower() in ("nan", "inf", "+inf", "-inf") or (t == "" and not blank_ok):
                    bad.append("row %d col %s" % (i + 1, header[j]))
        chk_list(checks, "F.%s.no_nan_inf" % fname, bad)
        chk_list(checks, "F.%s.reread_values" % fname, ["row %d" % (i + 1) for i, (g, r) in enumerate(zip(body, rows))
                 if len(g) != len(r) or not all(same_cell(t, v) for t, v in zip(g, r))])


# ===== 요약 =====
def max_delta(items):
    # items: (value, base, label). PA 대비 최대 증가
    best = None
    for v, b, lab in items:
        if v is not None and b is not None and (best is None or v - b > best[0]):
            best = (v - b, lab)
    return {"pp": best[0], "at": best[1]} if best else {"pp": None, "at": ""}


def delta_max(us, ls):
    out = {}
    for p in PNAMES[1:]:
        out[p] = {"ustar": {d: max_delta([(us[(d, ln, p, nm)]["u"], us[(d, ln, "PA", nm)]["u"],
                                           "%s %s N_M=%d %s vs PA" % (d, ln, nm, p)) for ln, _, _ in LINK for nm in NM_LIST])
                            for d, _ in DEV},
                  "ratio": {s: max_delta([(ls[(ln, s, p, nm)]["ratio"], ls[(ln, s, "PA", nm)]["ratio"],
                                           "%s %s N_M=%d %s vs PA" % (ln, s, nm, p)) for ln, _, _ in LINK for nm in NM_LIST])
                            for s, _ in SCENARIOS}}
    return out


def vrange(vals):
    v = [x for x in vals if x is not None]
    return {"min": min(v), "max": max(v)} if v else {"min": None, "max": None}


def r6(o):
    if isinstance(o, float):
        return round(o, 6)
    if isinstance(o, dict):
        return {k: r6(v) for k, v in o.items()}
    if isinstance(o, list):
        return [r6(v) for v in o]
    return o


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def f1(v):
    return "n/a" if v is None else "%.1f" % v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./o3_out")
    outdir = ap.parse_args().out
    os.makedirs(outdir, exist_ok=True)
    exist = [f for f in OUT_FILES if os.path.exists(os.path.join(outdir, f))]
    if exist:
        sys.stderr.write("output files already exist, not overwriting: %s\n" % ", ".join(exist))
        return 2
    script_sha = sha256(os.path.abspath(__file__))

    cm, us, ls = compute()
    checks = []
    checks_ab(checks, cm)
    checks_c(checks, cm, us)
    checks_d(checks, ls)
    hand = hand_block(checks, ls)

    tables = [("o3_comm.csv", HDR_COMM, rows_comm(cm), 12), ("o3_ustar.csv", HDR_USTAR, rows_ustar(us), 120),
              ("o3_lamstar.csv", HDR_LAM, rows_lam(ls), 180)]
    for fname, header, rows, _ in tables:
        write_csv(os.path.join(outdir, fname), header, rows)
    checks_f(checks, outdir, tables)

    npass = sum(c["ok"] for c in checks)
    failed = [c["name"] for c in checks if not c["ok"]]
    u0 = {d: 100.0 * F / (c * E) for d, c in DEV}
    dm = delta_max(us, ls)
    summary = {
        "python_version": platform.python_version(),
        "script_sha256": script_sha,
        "constants": {"E": E, "TAU": TAU, "T4_TM": T4_TM, "F": F, "N_TOTAL": N_TOTAL, "NM_LIST": NM_LIST,
                      "DEV_OPS": DEVD, "LINK": {n: {"B_bps": b, "l_n_s": l} for n, b, l in LINK},
                      "LAT": LAT, "EMB": EMB, "IMG": IMG, "PAYLOAD_EXTRA": EXTRA, "SCENARIOS_TM_S": dict(SCENARIOS)},
        "payloads": {p: {"extra_bytes": ex, "nonimg_bytes": nonimg_bytes(ex), "split_bytes": split_bytes(ex)} for p, ex in PAYLOAD},
        "u0_pct": u0,
        "delta_max": dm,
        "ustar_range_pct": {d: {p: vrange([us[(d, ln, p, nm)]["u"] for ln, _, _ in LINK for nm in NM_LIST]) for p in PNAMES}
                            for d, _ in DEV},
        "ratio_range_pct": {s: {p: vrange([ls[(ln, s, p, nm)]["ratio"] for ln, _, _ in LINK for nm in NM_LIST]) for p in PNAMES}
                            for s, _ in SCENARIOS},
        "checks": {"total": len(checks), "pass": npass, "fail": len(failed), "failed_names": failed},
    }
    with open(os.path.join(outdir, "o3_summary.json"), "x", encoding="utf-8", newline="\n") as fh:
        json.dump(r6(summary), fh, indent=2, ensure_ascii=True)
        fh.write("\n")

    # ----- stdout -----
    pr = print
    pr("O3 payload sensitivity (d8_payload_sens.py)")
    pr("python %s" % platform.python_version())
    pr("constants: E = tau = %.3f s, T4_TM = %.3f s, F = %d FLOPs, N_TOTAL = %d, NM_LIST = %s" % (E, T4_TM, F, N_TOTAL, NM_LIST))
    pr("  devices: " + ", ".join("%s %g OPS" % (d, c) for d, c in DEV))
    pr("  links: " + ", ".join("%s B=%g bps l_n=%.3f s" % (n, b, l) for n, b, l in LINK))
    pr("  LAT = %d B, EMB = %d B, IMG = %d B" % (LAT, EMB, IMG))
    pr("  scenarios: " + ", ".join("%s t_M=%.4f s" % (s, t) for s, t in SCENARIOS))
    pr("payloads (X = EMB + 2*LAT + extra; split bytes = X + IMG; B0 bytes = IMG):")
    pr("  %-4s %8s %8s %10s" % ("name", "extra", "X", "split"))
    for p, ex in PAYLOAD:
        pr("  %-4s %8d %8d %10d" % (p, ex, nonimg_bytes(ex), split_bytes(ex)))
    pr("")
    pr("[P] comm time [s] = 8*bytes/B + l_n")
    for ln, _, _ in LINK:
        pr("  %s: B0 comm = %.4f" % (ln, cm[(ln, "PA")]["cb"]))
        for p in PNAMES:
            pr("    %s split comm = %.4f  delta vs PA = %.4f" % (p, cm[(ln, p)]["cs"], cm[(ln, p)]["cs"] - cm[(ln, "PA")]["cs"]))
    pr("")
    pr("[U] u* [%] (no server queueing)")
    for d, _ in DEV:
        for ln, _, _ in LINK:
            pr("  %s %s (u0 = %.1f %%)   N_M: %s" % (d, ln, u0[d], " ".join("%7d" % nm for nm in NM_LIST)))
            for p in PNAMES:
                cells = []
                for nm in NM_LIST:
                    u = us[(d, ln, p, nm)]["u"]
                    cells.append("%7s" % f1(u) + (" u*>100" if u is not None and u > 100 else ""))
                pr("    %-4s %s" % (p, " ".join(cells)))
    pr("")
    pr("[L] lam* ratio [%] = 100*lam*/lam_max(B0) (s0 = 0, tau = 0.179)")
    for ln, _, _ in LINK:
        pr("  %s (lam_max(B0) = %.6f img/s)   N_M: %s" % (ln, lam_max_b0(), " ".join("%7d" % nm for nm in NM_LIST)))
        for s, _ in SCENARIOS:
            for p in PNAMES:
                pr("    %-4s %-4s %s" % (s, p, " ".join("%7s" % f1(ls[(ln, s, p, nm)]["ratio"]) for nm in NM_LIST)))
    pr("")
    pr("[S] max increase vs PA [pp]")
    for p in PNAMES[1:]:
        for d, _ in DEV:
            m = dm[p]["ustar"][d]
            pr("  %s u* %s: %s pp at %s" % (p, d, f1(m["pp"]), m["at"]))
        for s, _ in SCENARIOS:
            m = dm[p]["ratio"][s]
            pr("  %s ratio %s: %s pp at %s" % (p, s, f1(m["pp"]), m["at"]))
    pr("")
    pr("[E] hand-check")
    for line in hand:
        pr("  " + line)
    pr("")
    pr("[V] checks")
    for c in checks:
        pr("  %s %s" % ("PASS" if c["ok"] else "FAIL", c["name"]))
        if not c["ok"] and c["detail"]:
            pr("       failing: " + c["detail"])
    pr("checks: %d total, %d PASS, %d FAIL" % (len(checks), npass, len(failed)))
    pr("")
    pr(NOTES)
    pr("")
    pr("files (sha256):")
    for f in OUT_FILES:
        pr("  %s  %s" % (sha256(os.path.join(outdir, f)), f))
    pr("  %s  %s (script)" % (script_sha, os.path.basename(__file__)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

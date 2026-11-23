#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""d8_load_model.py - 엣지 서버 부하 M/D/1 해석 모델

SD v1.4 PNDM 25스텝(U-Net 26회)을 엣지 서버(앞 N_E회)와 모바일(뒤 N_M회)이 나눠
수행할 때, 서버를 M/D/1 대기열로 보고 서버 부하와 사용자 평균 종단 지연을 계산한다.
  S = N_E*tau + s0,  rho = lam*S,  W_q = rho*S / (2*(1-rho))  (rho < 1)
  T(lam) = W_q + S + N_M*t_M + comm(N_M, link)
표준 라이브러리만 사용한다. 실행: python d8_load_model.py --out /kaggle/working/o2

가정·한계:
- 포아송 도착을 가정했고 실제 요청 도착 패턴은 검증하지 않았다.
- 단일 서버, FIFO이고 배치 처리는 없다.
- 서비스 시간을 결정적으로 가정했다. U-Net 호출 시간의 변동은 측정하지 않았고, 확산 U-Net에 대한 M/D/1 모델 검증도 하지 않았다.
- τ = 0.179 s는 T4에서 측정한 상한이며 실제 서버는 다를 수 있다.
- t_M은 모바일 기기 실측이 아니다. r2.0과 r2.2는 비율 가정이고 T4 1.128 s는 GPU fake-quant 시뮬레이션 값이다.
- 모바일 대기열, 전송 혼잡, 재전송은 모형에 없다.
- 서버 고정 시간 s0는 민감도(0, 0.2 s)로만 다뤘다.
- 정상상태(ρ < 1) 평균 지연만 계산했고 분위수나 꼬리 지연은 계산하지 않았다.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys

TAU = 0.179            # 서버 U-Net 호출당 지연 [s]. T4 실측에서 나온 상한
TAU_SENS = 0.173       # tau 민감도 값 [s]
N_TOTAL = 26           # PNDM 25스텝 = U-Net 26회 호출
NM_LIST = [0, 6, 11, 16, 21, 26]   # 모바일 호출 수 N_M. 0 = B0(서버가 26회 전부)
NM_SPLIT = [6, 11, 16, 21, 26]     # B0를 뺀 분할 구성
LAT = 4 * 64 * 64 * 2      # FP16 잠재 텐서 4x64x64 [B] = 32768
EMB = 77 * 768 * 2         # FP16 텍스트 임베딩 77x768 [B] = 118272
IMG = 512 * 512 * 3        # 8비트 RGB 512x512 이미지 [B] = 786432
LINKS = {"WiFi": (100e6, 0.010), "5G": (25e6, 0.040), "4G": (10e6, 0.080)}  # (B [bps], l_n [s]) 가정값
LINK_ORDER = ["WiFi", "5G", "4G"]
REF_LINK = "5G"            # 기준 링크
S0_LIST = [0.0, 0.2]       # 서버 고정 시간 [s]. 민감도용, 기준 0.0
LAM_EX = 0.20              # 예시 부하점 [img/s] (B0 한계 0.2149 근처)
T4_TM = 1.128              # T4 GPU fake-quant 모바일 U-Net 1회 [s] (모바일 실측 아님)
SCENARIOS = [("r2.0", 2.0), ("r2.2", 2.2), ("T4", None)]  # t_M = r*tau, None이면 T4_TM 고정
# B0 대비 PSNR [dB] (평균, SD), 이미지 N=50. 계산에 쓰지 않음 (measured, copied)
PSNR = {6: (22.40, 2.41), 11: (21.01, 2.30), 16: (18.91, 2.14), 21: (15.84, 2.02), 26: (13.52, 1.97)}
# CLIP 점수 평균. 계산에 쓰지 않음 (measured, copied)
CLIP = {0: 0.3021, 6: 0.2981, 11: 0.2974, 16: 0.2970, 21: 0.2905, 26: 0.2958}
N_GRID = 120               # T(lam) 격자 점 수
RSWEEP_NM = [6, 16, 26]    # r 스윕 대상 N_M (r = i/10, i = 10..120)
SIM_SEED, SIM_N, SIM_WARM, SIM_S = 20261011, 2000000, 20000, 1.0
SIM_RHOS = [0.5, 0.7]
LIMITS = [  # 가정·한계 고정 문구 (docstring 문구의 ASCII 번역, stdout용)
    "- Poisson arrivals are assumed; the real request arrival pattern was not validated.",
    "- Single server, FIFO, no batching.",
    "- Service time is assumed deterministic. U-Net call time variation was not measured, "
    "and the M/D/1 model was not validated for diffusion U-Net serving.",
    "- tau = 0.179 s is an upper bound measured on T4; a real server may differ.",
    "- t_M is not measured on a mobile device. r2.0 and r2.2 are ratio assumptions; "
    "T4 1.128 s is a GPU fake-quant simulation value.",
    "- Mobile queueing, transmission congestion and retransmission are not modeled.",
    "- Server fixed time s0 is treated only as a sensitivity (0, 0.2 s).",
    "- Only steady-state (rho < 1) mean latency is computed; quantiles and tail latency are not.",
]


# ---------------- 모형 식 ----------------
def comm_time(nm, link):
    """전송 시간 + 링크 지연 l_n [s]. N_M=0이면 이미지만, N_M>0이면 EMB+2*LAT+IMG."""
    bw, ln = LINKS[link]
    nbytes = IMG if nm == 0 else EMB + 2 * LAT + IMG
    return 8.0 * nbytes / bw + ln


def service_time(nm, tau, s0):
    """서버 작업 시간 S = N_E*tau + s0."""
    return (N_TOTAL - nm) * tau + s0


def wait_time(lam, S):
    """M/D/1 평균 대기 W_q. S=0이면 0, rho>=1이면 inf."""
    if S == 0:
        return 0.0
    rho = lam * S
    if rho >= 1.0:
        return math.inf
    return rho * S / (2.0 * (1.0 - rho))


def lam_max(S):
    """한계 도착률 1/S (S=0이면 inf)."""
    return math.inf if S == 0 else 1.0 / S


def mobile_tm(scen, tau):
    """시나리오별 모바일 U-Net 1회 시간 t_M."""
    r = dict(SCENARIOS)[scen]
    return T4_TM if r is None else r * tau


def total_time(lam, nm, tm, tau, s0, link):
    """평균 종단 지연 T = W_q + S + N_M*t_M + comm."""
    S = service_time(nm, tau, s0)
    return wait_time(lam, S) + S + nm * tm + comm_time(nm, link)


def solve_lam_star(nm, tm, tau, s0, link):
    """교차 부하 lam*: f(lam) = T_B0 - T_split = 0. 끝점 부호와 격자 비감소도 확인."""
    lm0 = lam_max(service_time(0, tau, s0))

    def f(lam):
        return total_time(lam, 0, 0.0, tau, s0, link) - total_time(lam, nm, tm, tau, s0, link)

    res = {"lam_star": 0.0, "lam_max_B0": lm0, "ratio_pct": 0.0,
           "all_split": False, "ok": True, "note": "", "resid": 0.0}
    if f(0.0) >= 0.0:            # 전 구간에서 분할이 유리
        res["all_split"] = True
        return res
    lo, hi = 0.0, lm0 * (1.0 - 1e-12)
    end_ok = f(hi) > 0.0
    grid = [f(lm0 * 0.999999 * i / 400) for i in range(401)]
    mono_ok = all(grid[i + 1] >= grid[i] for i in range(400))
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)
    res.update(lam_star=lam, ratio_pct=100.0 * lam / lm0, ok=end_ok and mono_ok,
               note="end_ok=%s mono_ok=%s" % (end_ok, mono_ok), resid=f(lam))
    return res


# ---------------- 표 만들기 ----------------
def table1_rows():
    """표 1 (5G, s0=0, tau=0.179): 서버 축 + 모바일/품질 축."""
    lm0 = lam_max(service_time(0, TAU, 0.0))
    rows = []
    for nm in NM_LIST:
        S = service_time(nm, TAU, 0.0)
        row = {"nm": nm, "ne": N_TOTAL - nm, "S_s": S, "lam_max": lam_max(S),
               "cap_x": lam_max(S) / lm0, "rho_ex": LAM_EX * S, "Wq_ex_s": wait_time(LAM_EX, S)}
        for scen in ("r2.2", "T4"):
            tm = mobile_tm(scen, TAU)
            row["NMtM_%s_s" % scen] = nm * tm
            row["T_%s_lam0_s" % scen] = total_time(0.0, nm, tm, TAU, 0.0, REF_LINK)
            row["T_%s_ex_s" % scen] = total_time(LAM_EX, nm, tm, TAU, 0.0, REF_LINK)
        psnr = PSNR.get(nm)
        row["psnr_mean"] = psnr[0] if psnr else None
        row["psnr_sd"] = psnr[1] if psnr else None
        row["clip_mean"] = CLIP[nm]
        rows.append(row)
    return rows


def lamstar_row(tau, link, s0, scen, nm):
    tm = mobile_tm(scen, tau)
    row = {"tau": tau, "link": link, "s0": s0, "scenario": scen, "tM_s": tm, "nm": nm}
    row.update(solve_lam_star(nm, tm, tau, s0, link))
    return row


def lamstar_rows():
    """표 2: tau=0.179 90행 + tau=0.173 민감도 10행."""
    rows = [lamstar_row(TAU, link, s0, scen, nm) for link in LINK_ORDER for s0 in S0_LIST
            for scen, _ in SCENARIOS for nm in NM_SPLIT]
    rows += [lamstar_row(TAU_SENS, REF_LINK, 0.0, scen, nm) for scen in ("r2.2", "T4") for nm in NM_SPLIT]
    return rows


def t_vs_lambda_rows():
    """T(lam) 격자 120점 x config 4개 x 시나리오 2개 (5G, s0=0, tau=0.179)."""
    lm0 = lam_max(service_time(0, TAU, 0.0))
    lams = [0.005 + (0.99 * lm0 - 0.005) * i / (N_GRID - 1) for i in range(N_GRID)]
    rows = []
    for scen in ("r2.2", "T4"):
        tm = mobile_tm(scen, TAU)
        for cfg, nm in (("B0", 0), ("NM6", 6), ("NM16", 16), ("NM26", 26)):
            S = service_time(nm, TAU, 0.0)
            for lam in lams:
                rows.append({"scenario": scen, "config": cfg, "nm": nm, "lam": lam, "rho": lam * S,
                             "Wq_s": wait_time(lam, S), "S_s": S,
                             "T_s": total_time(lam, nm, tm, TAU, 0.0, REF_LINK)})
    return rows


def r_sweep_rows():
    """lam*의 r 의존성: r = i/10 (i=10..120), t_M = r*tau, 5G, s0=0."""
    rows = []
    for i in range(10, 121):
        r = i / 10.0
        for nm in RSWEEP_NM:
            row = {"r": r, "nm": nm}
            row.update(solve_lam_star(nm, r * TAU, TAU, 0.0, REF_LINK))
            rows.append(row)
    return rows


def sensitivity(ls_rows):
    """링크 효과와 s0 효과: |ratio 차이|의 최대와 그 조합 (tau=0.179 90행만)."""
    rat = {(x["link"], x["s0"], x["scenario"], x["nm"]): x["ratio_pct"] for x in ls_rows if x["tau"] == TAU}
    link_best, s0_best = (-1.0, ""), (-1.0, "")
    for (link, s0, scen, nm), v in sorted(rat.items()):
        if link != REF_LINK:
            d = abs(v - rat[(REF_LINK, s0, scen, nm)])
            if d > link_best[0]:
                link_best = (d, "%s, nm=%d, s0=%.1f, %s vs 5G" % (scen, nm, s0, link))
        if s0 != S0_LIST[0]:
            d = abs(v - rat[(link, S0_LIST[0], scen, nm)])
            if d > s0_best[0]:
                s0_best = (d, "%s, nm=%d, %s, s0=0.2 vs 0.0" % (scen, nm, link))
    return {"link_effect_pp": link_best[0], "link_effect_at": link_best[1],
            "s0_effect_pp": s0_best[0], "s0_effect_at": s0_best[1]}


def hand_calc(ls_rows):
    """손계산 검산: 식에 숫자를 대입한 중간값 (출력 줄, 값 dict)."""
    v, lines = {}, []
    bw, ln = LINKS[REF_LINK]
    S = service_time(0, TAU, 0.0)
    rho, wq, c = LAM_EX * S, wait_time(LAM_EX, S), comm_time(0, REF_LINK)
    v.update(b0_S=S, b0_rho=rho, b0_Wq=wq, b0_comm=c, b0_T=wq + S + c)
    lines += ["(i) B0, lam=0.20, 5G, s0=0",
              "  S    = 26*%.3f + 0.0 = %.6f" % (TAU, S),
              "  rho  = 0.20*%.6f = %.6f" % (S, rho),
              "  W_q  = %.6f*%.6f / (2*(1-%.6f)) = %.6f" % (rho, S, rho, wq),
              "  comm = 8*%d/%.0f + %.3f = %.6f" % (IMG, bw, ln, c),
              "  T    = %.6f + %.6f + 0 + %.6f = %.6f" % (wq, S, c, v["b0_T"])]
    tm = mobile_tm("r2.2", TAU)
    S = service_time(6, TAU, 0.0)
    rho, wq, c = LAM_EX * S, wait_time(LAM_EX, S), comm_time(6, REF_LINK)
    v.update(n6_S=S, n6_rho=rho, n6_Wq=wq, n6_NMtM=6 * tm, n6_comm=c, n6_T=wq + S + 6 * tm + c)
    lines += ["(ii) N_M=6, r2.2 (t_M=2.2*%.3f=%.4f), lam=0.20, 5G, s0=0" % (TAU, tm),
              "  S     = 20*%.3f + 0.0 = %.6f" % (TAU, S),
              "  rho   = 0.20*%.6f = %.6f" % (S, rho),
              "  W_q   = %.6f*%.6f / (2*(1-%.6f)) = %.6f" % (rho, S, rho, wq),
              "  N_M*t_M = 6*%.4f = %.6f" % (tm, 6 * tm),
              "  comm  = 8*(%d+2*%d+%d)/%.0f + %.3f = %.6f" % (EMB, LAT, IMG, bw, ln, c),
              "  T     = %.6f + %.6f + %.6f + %.6f = %.6f" % (wq, S, 6 * tm, c, v["n6_T"])]
    row = [x for x in ls_rows if (x["tau"], x["link"], x["s0"], x["scenario"], x["nm"])
           == (TAU, REF_LINK, 0.0, "r2.2", 26)][0]
    lam = row["lam_star"]
    tb0 = total_time(lam, 0, 0.0, TAU, 0.0, REF_LINK)
    tsp = total_time(lam, 26, tm, TAU, 0.0, REF_LINK)
    v.update(n26_lam=lam, n26_TB0=tb0, n26_Tsplit=tsp)
    lines += ["(iii) N_M=26, r2.2, 5G, s0=0 at lam*=%.8f" % lam,
              "  T_B0    = %.9f" % tb0, "  T_split = %.9f" % tsp, "  diff    = %.3e" % (tb0 - tsp)]
    return lines, v


def simulate_md1(rho):
    """Lindley 점화식 W_{n+1} = max(0, W_n + S - A_{n+1})로 M/D/1 평균 대기 모의 (S=1)."""
    rng = random.Random(SIM_SEED)
    w, total, cnt = 0.0, 0.0, 0
    for n in range(SIM_N):
        if n >= SIM_WARM:
            total += w
            cnt += 1
        w = max(0.0, w + SIM_S - rng.expovariate(rho))
    return total / cnt


def sim_sanity():
    out = []
    for rho in SIM_RHOS:
        sim, ref = simulate_md1(rho), wait_time(rho / SIM_S, SIM_S)
        out.append({"rho": rho, "S": SIM_S, "seed": SIM_SEED, "n_customers": SIM_N, "warmup": SIM_WARM,
                    "Wq_sim": sim, "Wq_formula": ref, "rel_err": abs(sim - ref) / ref})
    return out


# ---------------- 검증 ----------------
def fmt(x, d):
    if x is None:
        return ""
    if isinstance(x, float):
        return "inf" if math.isinf(x) else "%.*f" % (d, x)
    return str(x)


def check(results, name, passed, detail=""):
    """모든 검증이 거치는 도우미. 실패해도 멈추지 않고 기록만 한다."""
    results.append({"name": name, "passed": bool(passed), "detail": detail})


def check_close(results, name, actual, expected, tol):
    if math.isinf(expected):
        ok = math.isinf(actual)
    else:
        ok = (not math.isinf(actual)) and abs(actual - expected) <= tol
    check(results, name, ok, "actual=%s expected=%s tol=%g" % (fmt(actual, 8), fmt(expected, 8), tol))


def run_checks(t1, ls, grid, rs, sens, hv, sim):
    res = []
    # A. 입력 상수
    check(res, "A_LAT", LAT == 32768, "LAT=%d" % LAT)
    check(res, "A_EMB", EMB == 118272, "EMB=%d" % EMB)
    check(res, "A_EMB+LAT", EMB + LAT == 151040, "%d" % (EMB + LAT))
    check(res, "A_IMG", IMG == 786432, "IMG=%d" % IMG)
    check(res, "A_EMB+2LAT+IMG", EMB + 2 * LAT + IMG == 970240, "%d" % (EMB + 2 * LAT + IMG))
    for link, c0, c1 in (("WiFi", 0.07291, 0.08762), ("5G", 0.29166, 0.35048), ("4G", 0.70915, 0.85619)):
        check_close(res, "A_comm_%s_nm0" % link, comm_time(0, link), c0, 1e-5)
        check_close(res, "A_comm_%s_nm>0" % link, comm_time(6, link), c1, 1e-5)
    check(res, "A_ne_plus_nm_26", all(r["ne"] + r["nm"] == N_TOTAL for r in t1), "all NM_LIST")
    # B. 식 단위 검사
    check_close(res, "B_Wq(0.20,4.654)", wait_time(0.20, 4.654), 31.30017, 1e-4)
    check_close(res, "B_Wq(0,4.654)", wait_time(0.0, 4.654), 0.0, 0.0)
    check_close(res, "B_Wq(5.0,0)", wait_time(5.0, 0.0), 0.0, 0.0)
    check_close(res, "B_Wq(0.25,4.654)_inf", wait_time(0.25, 4.654), math.inf, 0.0)
    check_close(res, "B_Wq(1.0,1.0)_inf", wait_time(1.0, 1.0), math.inf, 0.0)
    check(res, "B_nm26_S0", service_time(26, TAU, 0.0) == 0.0, "S=%r" % service_time(26, TAU, 0.0))
    check(res, "B_nm26_Wq0", wait_time(LAM_EX, service_time(26, TAU, 0.0)) == 0.0, "")
    for scen, _ in SCENARIOS:
        tm = mobile_tm(scen, TAU)
        check_close(res, "B_nm26_T_%s" % scen, total_time(LAM_EX, 26, tm, TAU, 0.0, REF_LINK),
                    26 * tm + comm_time(26, REF_LINK), 1e-12)
    # C(a). 서버 축
    ref_a = {"ne": ([26, 20, 15, 10, 5, 0], 0), "S_s": ([4.654, 3.580, 2.685, 1.790, 0.895, 0.0], 1e-9),
             "lam_max": ([0.214869, 0.279330, 0.372439, 0.558659, 1.117318, math.inf], 1e-6),
             "cap_x": ([1.0, 1.3, 1.7333, 2.6, 5.2, math.inf], 1e-4),
             "rho_ex": ([0.9308, 0.7160, 0.5370, 0.3580, 0.1790, 0.0], 1e-9)}
    for key, (vals, tol) in ref_a.items():
        for row, e in zip(t1, vals):
            check_close(res, "Ca_%s_nm%d" % (key, row["nm"]), float(row[key]), float(e), tol)
    check_close(res, "Ca_Wq_B0_0.20", t1[0]["Wq_ex_s"], 31.30017, 1e-3)
    check_close(res, "Ca_T_B0_0.20", t1[0]["T_r2.2_ex_s"], 36.2458, 1e-3)
    check_close(res, "Ca_T_B0_0", t1[0]["T_r2.2_lam0_s"], 4.9457, 1e-3)
    # C(b). 모바일 축
    ref_b = {"T_r2.2_lam0_s": [6.2933, 7.3673, 8.4413, 9.5153, 10.5893],
             "T_T4_lam0_s": [10.6985, 15.4435, 20.1885, 24.9335, 29.6785],
             "T_r2.2_ex_s": [10.8061, 8.9243, 8.9404, 9.6128, 10.5893],
             "T_T4_ex_s": [15.2113, 17.0005, 20.6876, 25.0310, 29.6785],
             "NMtM_r2.2_s": [2.3628, 4.3318, 6.3008, 8.2698, 10.2388],
             "NMtM_T4_s": [6.7680, 12.4080, 18.0480, 23.6880, 29.3280]}
    for key, vals in ref_b.items():
        for row, e in zip(t1[1:], vals):
            check_close(res, "Cb_%s_nm%d" % (key, row["nm"]), row[key], e, 1e-3)
    # C(c)(d)(e). 교차 부하
    lk = {(x["tau"], x["link"], x["s0"], x["scenario"], x["nm"]): x for x in ls}

    def chk_ls(tag, key, lam_e, ratio_e):
        check_close(res, "%s_lamstar_%s" % (tag, key), lk[key]["lam_star"], lam_e, 2e-5)
        check_close(res, "%s_ratio_%s" % (tag, key), lk[key]["ratio_pct"], ratio_e, 2e-3)

    ref_c = {"r2.0": ([0.10448, 0.11376, 0.12405, 0.13430, 0.14384], [48.625, 52.944, 57.731, 62.504, 66.945]),
             "r2.2": ([0.11241, 0.12237, 0.13293, 0.14305, 0.15214], [52.317, 56.952, 61.865, 66.577, 70.805]),
             "T4": ([0.16859, 0.17934, 0.18712, 0.19256, 0.19639], [78.464, 83.463, 87.087, 89.615, 91.401])}
    for scen, (lams, ratios) in ref_c.items():
        for nm, le, re_ in zip(NM_SPLIT, lams, ratios):
            chk_ls("Cc", (TAU, "5G", 0.0, scen, nm), le, re_)
    check_close(res, "Cc_lam_max_B0", lk[(TAU, "5G", 0.0, "r2.0", 6)]["lam_max_B0"], 0.214869, 1e-6)
    for link, s0, scen, nm, le, re_ in (("WiFi", 0.2, "T4", 6, 0.16083, 78.069), ("WiFi", 0.2, "T4", 26, 0.18758, 91.051),
                                        ("4G", 0.0, "r2.0", 6, 0.10792, 50.224), ("4G", 0.0, "r2.0", 26, 0.14472, 67.354),
                                        ("5G", 0.2, "r2.2", 11, 0.11637, 56.487), ("5G", 0.2, "r2.2", 26, 0.14409, 69.939)):
        chk_ls("Cd", (TAU, link, s0, scen, nm), le, re_)
    check_close(res, "Cd_lam_max_B0_s0=0.2", lk[(TAU, "5G", 0.2, "r2.2", 6)]["lam_max_B0"], 0.206016, 1e-6)
    ref_e = {"r2.2": ([0.11638, 0.12666, 0.13757, 0.14804, 0.15743], [52.349, 56.971, 61.879, 66.587, 70.813]),
             "T4": ([0.17568, 0.18669, 0.19458, 0.20005, 0.20390], [79.019, 83.972, 87.524, 89.984, 91.713])}
    for scen, (lams, ratios) in ref_e.items():
        for nm, le, re_ in zip(NM_SPLIT, lams, ratios):
            chk_ls("Ce", (TAU_SENS, "5G", 0.0, scen, nm), le, re_)
    check_close(res, "Ce_lam_max_B0_tau0.173", lk[(TAU_SENS, "5G", 0.0, "T4", 6)]["lam_max_B0"], 0.222321, 1e-6)
    # C(f). r 스윕 표본
    rk = {(int(round(x["r"] * 10)), x["nm"]): x for x in rs}
    for i10, vals in ((10, (5.722, 2.872, 2.465)), (40, (69.684, 79.538, 85.766)),
                      (100, (84.912, 91.878, 94.744)), (120, (86.937, 93.234, 95.657))):
        for nm, e in zip(RSWEEP_NM, vals):
            check_close(res, "Cf_ratio_r%.1f_nm%d" % (i10 / 10.0, nm), rk[(i10, nm)]["ratio_pct"], e, 2e-3)
    half = solve_lam_star(26, 0.5 * TAU, TAU, 0.0, REF_LINK)
    check(res, "Cf_r0.5_nm26_all_split", half["all_split"] and half["lam_star"] == 0.0,
          "all_split=%s lam_star=%r" % (half["all_split"], half["lam_star"]))
    # C(g). 민감도 상한
    check(res, "Cg_link_effect<=2.0pp", sens["link_effect_pp"] <= 2.0,
          "%.4f pp at %s (ref 1.5997 at r2.0, nm=6, s0=0, 4G)" % (sens["link_effect_pp"], sens["link_effect_at"]))
    check(res, "Cg_s0_effect<=1.0pp", sens["s0_effect_pp"] <= 1.0,
          "%.4f pp at %s (ref 0.9271 at r2.0, nm=26, WiFi)" % (sens["s0_effect_pp"], sens["s0_effect_at"]))
    # C(h). 손계산 값
    for key, e, tol in (("b0_S", 4.654, 1e-9), ("b0_rho", 0.9308, 1e-9), ("b0_Wq", 31.3002, 1e-3),
                        ("b0_comm", 0.29166, 1e-5), ("b0_T", 36.2458, 1e-3), ("n6_S", 3.580, 1e-9),
                        ("n6_rho", 0.7160, 1e-9), ("n6_Wq", 4.5128, 1e-3), ("n6_NMtM", 2.3628, 1e-3),
                        ("n6_comm", 0.35048, 1e-5), ("n6_T", 10.8061, 1e-3)):
        check_close(res, "Ch_%s" % key, hv[key], e, tol)
    # C(i). T(lam) 격자 표본
    def g(scen, cfg, i):
        return [x for x in grid if x["scenario"] == scen and x["config"] == cfg][i]
    for i, lam_e, tb0, t6, t26 in ((0, 0.005, 5.00110, 6.32590, 29.67848),
                                   (N_GRID - 1, 0.21272024, 235.31866, 12.00973, 29.67848)):
        check_close(res, "Ci_lam_i%d" % i, g("r2.2", "B0", i)["lam"], lam_e, 1e-6)
        check_close(res, "Ci_T_B0_i%d" % i, g("r2.2", "B0", i)["T_s"], tb0, 1e-3)
        check_close(res, "Ci_T_NM6_r2.2_i%d" % i, g("r2.2", "NM6", i)["T_s"], t6, 1e-3)
        check_close(res, "Ci_T_NM26_T4_i%d" % i, g("T4", "NM26", i)["T_s"], t26, 1e-3)
    check_close(res, "Ci_rho_B0_i119", g("r2.2", "B0", N_GRID - 1)["rho"], 0.99, 1e-9)
    check_close(res, "Ci_Wq_B0_i119", g("r2.2", "B0", N_GRID - 1)["Wq_s"], 230.37300, 1e-3)
    # D. 성질
    allrows = [("tau=%g %s s0=%g %s nm=%d" % (x["tau"], x["link"], x["s0"], x["scenario"], x["nm"]), x) for x in ls]
    allrows += [("rsweep r=%.1f nm=%d" % (x["r"], x["nm"]), x) for x in rs]
    bad = [n for n, x in allrows if not x["ok"]]
    check(res, "D_lamstar_bracket_and_monotone", not bad, "n=%d failed=%s" % (len(allrows), bad))
    pos = [(abs(x["resid"]), n) for n, x in allrows if x["lam_star"] > 0]
    worst = max(pos) if pos else (0.0, "")
    check(res, "D_T_equal_at_lamstar<1e-9", worst[0] < 1e-9, "max|diff|=%.3e at %s" % worst)
    errs = [(abs(lam_max(service_time(nm, tau, s0)) * service_time(nm, tau, s0) - 1.0), tau, s0, nm)
            for tau in (TAU, TAU_SENS) for s0 in S0_LIST for nm in NM_LIST if service_time(nm, tau, s0) > 0]
    check(res, "D_lam_max*S=1(<1e-12)", max(errs)[0] < 1e-12, "max err=%.3e" % max(errs)[0])
    for tau in (TAU, TAU_SENS):
        for s0 in S0_LIST:
            lm0 = lam_max(service_time(0, tau, s0))
            caps = [lam_max(service_time(nm, tau, s0)) / lm0 for nm in NM_SPLIT]
            check(res, "D_cap_increasing_tau%g_s0%g" % (tau, s0), all(b > a for a, b in zip(caps, caps[1:])),
                  "caps=%s" % [fmt(c, 4) for c in caps])
    for nm in RSWEEP_NM:
        seq = [x["ratio_pct"] for x in rs if x["nm"] == nm]
        check(res, "D_rsweep_ratio_nondecreasing_nm%d" % nm, all(b >= a for a, b in zip(seq, seq[1:])), "")
    for name, n, e in (("table1", len(t1), 6), ("lamstar_table", len(ls), 100),
                       ("T_vs_lambda", len(grid), 960), ("lamstar_vs_r", len(rs), 333)):
        check(res, "D_rows_%s" % name, n == e, "rows=%d expected=%d" % (n, e))
    # E. W_q 식 구현의 모의 확인
    for s in sim:
        check(res, "E_sim_rho%.1f_relerr<=2%%" % s["rho"], s["rel_err"] <= 0.02,
              "sim=%.5f formula=%.5f rel_err=%.4f%%" % (s["Wq_sim"], s["Wq_formula"], 100 * s["rel_err"]))
    return res


# ---------------- 출력 ----------------
def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow([fmt(r[k], 6) for k in header])


def jsonable(x):
    if isinstance(x, float):
        return "inf" if math.isinf(x) else round(x, 6)
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def write_outputs(out, t1, ls, grid, rs, sens, checks, sim):
    paths = [os.path.join(out, n) for n in ("o2_table1.csv", "o2_lamstar_table.csv", "o2_T_vs_lambda.csv",
                                            "o2_lamstar_vs_r.csv", "o2_summary.json")]
    write_csv(paths[0], ["nm", "ne", "S_s", "lam_max", "cap_x", "rho_ex", "Wq_ex_s", "NMtM_r2.2_s", "NMtM_T4_s",
                         "T_r2.2_lam0_s", "T_T4_lam0_s", "T_r2.2_ex_s", "T_T4_ex_s", "psnr_mean", "psnr_sd",
                         "clip_mean"], t1)
    write_csv(paths[1], ["tau", "link", "s0", "scenario", "tM_s", "nm", "lam_star", "lam_max_B0", "ratio_pct"], ls)
    write_csv(paths[2], ["scenario", "config", "nm", "lam", "rho", "Wq_s", "S_s", "T_s"], grid)
    write_csv(paths[3], ["r", "nm", "lam_star", "lam_max_B0", "ratio_pct"], rs)
    ref = {s: {str(x["nm"]): {"lam_star": x["lam_star"], "ratio_pct": x["ratio_pct"]} for x in ls
               if (x["tau"], x["link"], x["s0"], x["scenario"]) == (TAU, REF_LINK, 0.0, s)} for s, _ in SCENARIOS}
    consts = {"TAU": TAU, "TAU_SENS": TAU_SENS, "N_TOTAL": N_TOTAL, "NM_LIST": NM_LIST, "LAT": LAT, "EMB": EMB,
              "IMG": IMG, "LINKS": {k: {"B_bps": b, "l_n_s": l} for k, (b, l) in LINKS.items()},
              "S0_LIST": S0_LIST, "LAM_EX": LAM_EX, "T4_TM": T4_TM, "SCENARIOS": dict(SCENARIOS),
              "PSNR_measured_copied": PSNR, "CLIP_measured_copied": CLIP}
    summary = {"constants": consts, "table1": t1, "lamstar_5G_s0_0": ref, "sensitivity": sens,
               "checks": checks, "sim_sanity": sim}
    with open(paths[4], "w", encoding="utf-8") as fh:
        json.dump(jsonable(summary), fh, indent=2)
    return paths


def print_report(t1, ls, sens, hand_lines, checks, paths):
    print("== O2 M/D/1 server load model ==")
    print("python %s (%s)" % (platform.python_version(), sys.executable))
    print("TAU=%.3f TAU_SENS=%.3f N_TOTAL=%d LAT=%d EMB=%d IMG=%d" % (TAU, TAU_SENS, N_TOTAL, LAT, EMB, IMG))
    print("LINKS: " + ", ".join("%s(B=%g bps, l_n=%.3f s)" % (k, LINKS[k][0], LINKS[k][1]) for k in LINK_ORDER))
    print("S0_LIST=%s LAM_EX=%.2f T4_TM=%.3f" % (S0_LIST, LAM_EX, T4_TM))
    for tau in (TAU, TAU_SENS):
        print("t_M at tau=%.3f: " % tau + ", ".join("%s=%.4f" % (s, mobile_tm(s, tau)) for s, _ in SCENARIOS))
    print("\n[Table 1a] server axis (5G, s0=0, tau=%.3f, lam_ex=%.2f)" % (TAU, LAM_EX))
    print("%4s %4s %8s %9s %7s %7s %9s" % ("nm", "ne", "S", "lam_max", "cap_x", "rho", "W_q"))
    for r in t1:
        print("%4d %4d %8s %9s %7s %7s %9s" % (r["nm"], r["ne"], fmt(r["S_s"], 3), fmt(r["lam_max"], 4),
                                             fmt(r["cap_x"], 2), fmt(r["rho_ex"], 3), fmt(r["Wq_ex_s"], 2)))
    print("\n[Table 1b] mobile / quality axis (5G, s0=0). PSNR, CLIP: measured, copied")
    print("%4s %10s %9s %10s %8s %10s %8s %12s %7s" % ("nm", "NMtM_r2.2", "NMtM_T4", "T0_r2.2", "T0_T4",
                                                      "Tex_r2.2", "Tex_T4", "PSNR", "CLIP"))
    for r in t1:
        ps = "-" if r["psnr_mean"] is None else "%.2f+-%.2f" % (r["psnr_mean"], r["psnr_sd"])
        print("%4d %10s %9s %10s %8s %10s %8s %12s %7.4f" % (
            r["nm"], fmt(r["NMtM_r2.2_s"], 2), fmt(r["NMtM_T4_s"], 2), fmt(r["T_r2.2_lam0_s"], 2),
            fmt(r["T_T4_lam0_s"], 2), fmt(r["T_r2.2_ex_s"], 2), fmt(r["T_T4_ex_s"], 2), ps, r["clip_mean"]))
    print("\n[Table 2] crossover load lam* (ratio % of lam_max(B0)). A = split better on whole range, ! = check failed")
    blocks = [(TAU, l, s0) for l in LINK_ORDER for s0 in S0_LIST] + [(TAU_SENS, REF_LINK, 0.0)]
    for tau, link, s0 in blocks:
        sub = [x for x in ls if (x["tau"], x["link"], x["s0"]) == (tau, link, s0)]
        print("-- tau=%.3f link=%s s0=%.1f lam_max(B0)=%.4f" % (tau, link, s0, sub[0]["lam_max_B0"]))
        print("%6s " % "scen" + " ".join("%17s" % ("nm=%d" % nm) for nm in NM_SPLIT))
        for scen in [s for s, _ in SCENARIOS if any(x["scenario"] == s for x in sub)]:
            cells = ["%.4f (%.1f)%s%s" % (x["lam_star"], x["ratio_pct"], "A" if x["all_split"] else "",
                                          "" if x["ok"] else "!") for x in sub if x["scenario"] == scen]
            print("%6s " % scen + " ".join("%17s" % c for c in cells))
    print("\n[Sensitivity] (tau=%.3f, 90 rows)" % TAU)
    print("link effect max |ratio(link,s0) - ratio(5G,s0)| = %.4f pp at %s" % (sens["link_effect_pp"], sens["link_effect_at"]))
    print("s0 effect   max |ratio(link,0.2) - ratio(link,0)| = %.4f pp at %s" % (sens["s0_effect_pp"], sens["s0_effect_at"]))
    print("\n[Hand check]")
    for line in hand_lines:
        print(line)
    print("\n[Checks]")
    for c in checks:
        print("%s %s %s" % ("PASS" if c["passed"] else "FAIL", c["name"], c["detail"]))
    n_fail = sum(1 for c in checks if not c["passed"])
    print("checks: %d total, %d PASS, %d FAIL" % (len(checks), len(checks) - n_fail, n_fail))
    print("\n[Assumptions and limits]")
    for line in LIMITS:
        print(line)
    print("\n[Files]")
    for p in paths:
        print("%s  sha256=%s" % (p, sha256(p)))
    print("script %s  sha256=%s" % (os.path.abspath(__file__), sha256(os.path.abspath(__file__))))
    return n_fail


def main():
    ap = argparse.ArgumentParser(description="O2 M/D/1 server load model")
    ap.add_argument("--out", default="./o2_out", help="output folder (created if missing)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t1, ls, grid, rs = table1_rows(), lamstar_rows(), t_vs_lambda_rows(), r_sweep_rows()
    sens = sensitivity(ls)
    hand_lines, hv = hand_calc(ls)
    sim = sim_sanity()
    checks = run_checks(t1, ls, grid, rs, sens, hv, sim)
    paths = write_outputs(args.out, t1, ls, grid, rs, sens, checks, sim)
    n_fail = print_report(t1, ls, sens, hand_lines, checks, paths)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()

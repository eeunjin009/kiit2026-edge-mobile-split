# 엣지–모바일 분할 추론 기반 확산 모델 서비스의 서버 부하·지연·충실도 분석

2026 한국정보기술학회 추계 종합학술대회 논문의 실험 코드와 결과입니다.
민은진(국민대학교), 이창우(국민대학교, 교신저자)

Stable Diffusion v1.4의 U-Net 호출 26회(PNDM 25스텝)를 FP16 엣지 서버가 앞쪽 N_E회, DGQ W4A8 양자화 U-Net을 쓰는 모바일이 뒤쪽 N_M회 나눠 처리할 때(N_E + N_M = 26), 서버 부하(M/D/1 모델), 평균 종단 지연, 충실도(CLIP, PSNR)를 분석합니다.

> **주의:** 모바일 쪽은 실제 모바일 기기가 아니라 T4 GPU에서 양자화를 모사(fake-quantization)한 결과입니다. 모바일 호출 시간은 실측하지 못해 비율 r = t_M / τ_E (2.0, 2.2, 6.3)를 가정으로 두었습니다. 자세한 한계는 논문 4장을 참고하세요.

## 폴더 구조

```
bridge/                  분할 추론 핵심 코드 (Hybrid-SD 위에 모바일 U-Net만 교체)
  run_split.py             Hybrid-SD 생성 루프를 그대로 실행하고 두 메서드만 감쌈
  mobile_slot.py           모바일 U-Net(DGQ W4A8) 불러오기와 호출 슬롯
  check_*.py               이미지·메모리·동치성 점검 스크립트
analysis/
  d8_load_model.py         M/D/1 서버 부하·지연 모델 (논문 2.2절, 표 1, 그림 2)
  d8_payload_sens.py       전송량 민감도와 모바일 부담 (논문 3.2~3.3절)
experiments/
  n200/                    최종 실험: MS-COCO val2014 캡션 200개 (논문 3.3절)
    build/                   실험에 쓴 코드
      d9_kaggle.py             단계별 실행(환경 준비 → 생성 → PSNR·CLIP → 결과 JSON)
      d9_wrapper.py            노트북 셀 본문(내장 코드 풀기, 체크포인트 찾기, d9_kaggle.py 실행)
      bridge/                  run_split.py, mobile_slot.py와 부분집합·기준선·CLIP·요약 스크립트
    tools/
      make_notebook_d9.py      build/로 Kaggle 노트북을 만드는 생성기
results/
  n200/                    d9_result_A.json, d9_result_B.json — 충실도 결과 원본
  o2_model/                M/D/1 모델 출력 (표 1, 그림 2, 교차 부하)
  o3_payload/              전송량 민감도와 모바일 부담 출력
requirements.txt           실험 환경의 패키지 버전
LICENSE                    MIT 라이선스
```

`bridge/`의 `run_split.py`, `mobile_slot.py`, `check_img.py`는 `experiments/n200/build/bridge/`에도 같은 코드로 들어 있습니다. 최상위 `bridge/`는 핵심 코드를 찾기 쉽게 따로 둔 것이고, 노트북에는 `build/bridge/`가 들어갑니다. `check_*.py`는 예비 단계에서 모바일 U-Net 동작을 확인할 때 쓴 스크립트로, 최종 실험에서는 실행하지 않습니다.

## 논문 항목과 파일의 대응

| 논문 항목 | 값 | 파일 |
|---|---|---|
| 표 1 한계 도착률(배), 이용률, 대기 시간(31.30 → 4.51 s) | cap_x, rho_ex, Wq_ex_s 열 | `results/o2_model/o2_table1.csv` |
| 표 1 교차 부하(r=2.0, 2.2, 6.3) | ratio_pct 열 (5G, s0=0 행) | `results/o2_model/o2_lamstar_table.csv` |
| 그림 2 도착률에 따른 평균 종단 지연(r=2.2) | T_s 열 | `results/o2_model/o2_T_vs_lambda.csv` |
| 3.2절 링크·s0 민감도(1.6%p, 0.9%p) | link_effect_pp, s0_effect_pp | `results/o2_model/o2_summary.json` |
| 3.2절 추가 전송량 민감도(2.8%p) | delta_pp_vs_PA 열 | `results/o3_payload/o3_lamstar.csv` |
| 3.3절 모바일 실효 처리율(RPi 5, Orin NX) | ustar_pct 열 | `results/o3_payload/o3_ustar.csv` |
| 그림 3, 3.3절 PSNR·CLIP | `psnr_vs_B0`, `clip_original` | `results/n200/d9_result_A.json`, `d9_result_B.json` |
| 3.3절 서로 다른 캡션 쌍 기준선(평균 8.51, 최댓값 12.57 dB) | `base.psnr_unrelated` (100쌍이 두 번 기록되어 항목은 200개) | `results/n200/d9_result_*.json` |
| 3.3절 T4 모사 모바일 호출 시간(1.10~1.13 s) | `slot.ms_mean` | `results/n200/d9_result_*.json` |

### 구성 이름과 논문 표기

결과 파일의 구성 이름은 `--steps "n_edge,n_mobile"` 인자에서 온 것이고, 모바일 호출은 n_mobile + 1회입니다. 논문의 N_M과 이렇게 대응합니다.

| 파일의 구성 이름 | 논문의 N_M | 세션 |
|---|---|---|
| B0 | 0 (엣지 서버 단독) | A, B 둘 다 |
| Q5 | 6 | B |
| Q10 | 11 | B |
| Q15 | 16 | B |
| Q20 | 21 | A |
| Q25 | 26 | A |

세션 A는 `B0, Q20, Q25`, 세션 B는 `B0, Q5, Q10, Q15`입니다. 두 세션의 B0 CLIP은 0.2999로 같습니다.

### 알려진 차이

- `o2_table1.csv`의 `psnr_mean`, `clip_mean` 열은 모델 계산 때 복사한 이전 실행 값이라 **논문의 값(N=200)과 다릅니다.** 논문의 PSNR·CLIP은 반드시 `results/n200/`에서 가져오세요. (예: N_M=6 PSNR 22.80, N_M=26 PSNR 13.92, CLIP 0.2999 → 0.2848)
- 논문 3.2절의 처리 시간 변동계수 C²=1 민감도(교차 부하 34~84%)를 계산한 스크립트는 이 저장소에 없습니다.

## 실행 방법

### 1. 서버 부하·지연 모델 (CPU, 표준 라이브러리만 사용)

```bash
python analysis/d8_load_model.py --out out/o2
python analysis/d8_payload_sens.py --out out/o3
```

두 스크립트는 `results/o2_model/`, `results/o3_payload/`와 같은 CSV, JSON을 만듭니다. 다시 실행해 비교하면 `o3_summary.json`의 파이썬 버전과 스크립트 sha256 줄만 다르고 나머지는 모두 같습니다. `d8_payload_sens.py`는 출력 파일이 이미 있으면 덮어쓰지 않고 종료합니다.

### 2. 분할 추론 실험 (GPU 필요)

실험은 Kaggle(T4 GPU)에서 세션 두 개(A, B)로 나눠 실행했습니다. 노트북은 저장소에 넣지 않았으니 먼저 만듭니다.

```bash
cd experiments/n200
python3 tools/make_notebook_d9.py . A B0,Q20,Q25        # kaggle/d9_notebook_A.ipynb
python3 tools/make_notebook_d9.py . B B0,Q5,Q10,Q15     # kaggle/d9_notebook_B.ipynb
```

생성기는 `build/`의 코드를 base64로 노트북 셀 하나에 넣고, 셀이 실행될 때 sha256으로 내용을 확인합니다. 그다음 Kaggle에서 아래 순서로 실행합니다.

1. 만든 노트북을 Kaggle에 올리고 Settings에서 GPU T4와 Internet을 켭니다.
2. DGQ W4A8 체크포인트 `cali_ckpt_activation_w4a8g1.pth_merged`(6,883,907,740바이트)를 Kaggle 데이터셋으로 붙입니다. 이 파일은 저장소에 없습니다. 노트북은 `/kaggle/input` 아래에서 파일 이름으로 찾고, 크기가 다르면 멈춥니다.
3. HuggingFace 토큰을 Kaggle Secrets에 `HF_TOKEN`으로 등록합니다.
4. 셀을 실행합니다. DGQ와 Hybrid-SD를 아래 커밋으로 받고 `requirements.txt`의 버전으로 패키지를 설치한 뒤 생성과 평가를 진행합니다. 끝나면 `/kaggle/working/d9_result_A.json`(또는 B)이 생깁니다. 도중에 끊기면 같은 세션에서 셀을 다시 실행하면 끝난 단계는 건너뜁니다.

평가에는 MS-COCO val2014 캡션 중 200개를 씁니다. Nota(BK-SDM)가 공개한 MS-COCO val2014 30k `metadata.csv`에서 예비 실험에 쓴 앞 50행을 빼고, 나머지에서 sha256("2026:<파일 이름>")이 작은 200행을 고릅니다(`experiments/n200/build/bridge/make_subset.py`).

참고:
- `build/d9_wrapper.py`는 생성기가 맨 앞에 `EMBED`, `D9_TAG` 등의 변수를 붙인 뒤에 실행되는 파일이라 단독으로는 실행되지 않습니다.
- 일부 함수 이름(`d5_*`)과 환경변수 이름(`D5_CKPT` 등)은 예비 실험(D5) 코드에서 그대로 이어받은 것입니다.

## 외부 코드와 모델

아래 외부 저장소의 코드는 이 저장소에 넣지 않았고, 실험 때 정해진 커밋으로 직접 받아 썼습니다.

| 대상 | 주소 | 버전 |
|---|---|---|
| DGQ | https://github.com/ugonfor/DGQ | `main` 브랜치, 커밋 `f449f6ad8052105c72e0bdf5a1cd05bbb98ae037` |
| Hybrid-SD | https://github.com/bytedance/Hybrid-SD | 커밋 `dcac636c82658ee055c71b8377ee06508fb9937c` |
| Stable Diffusion v1.4 | HuggingFace `CompVis/stable-diffusion-v1-4` | 모델 가중치는 별도 다운로드 |

실험 환경: Python 3.9.25, torch 2.2.2+cu121, diffusers 0.27.0(DGQ 쪽은 DGQ 저장소에 포함된 diffusers를 사용), open_clip_torch 3.3.0, NVIDIA Tesla T4. 전체 버전은 `requirements.txt`에 있습니다.

## 출처

- **DGQ** (Ryu et al., ICLR 2025, https://github.com/ugonfor/DGQ, 커밋 `f449f6a`): 모바일 쪽 W4A8 양자화 U-Net을 만들고 불러오는 코드(`get_qmodel` 등)와 CLIP 점수 계산 스크립트를 그대로 사용했습니다. 이 저장소의 `bridge/mobile_slot.py`, `bridge/check_*.py`는 DGQ 함수를 불러 쓰기만 하고 DGQ 코드를 담고 있지 않습니다. CLIP 점수는 DGQ의 `src/eval_clip_score.py`로 계산했고, `experiments/n200/build/bridge/clip_per_image.py`는 이 스크립트와 같은 계산으로 이미지별 점수를 저장하도록 새로 작성했습니다.
- **Hybrid-SD** (https://github.com/bytedance/Hybrid-SD, 커밋 `dcac636`, Apache-2.0): 엣지 서버와 모바일 U-Net을 단계별로 나눠 부르는 생성 루프(`examples/hybrid_sd/hybrid_batch.py`, `compression/hybrid_sd/inference_pipeline.py`)를 그대로 실행했습니다. `bridge/run_split.py`는 이 루프의 메서드 두 개만 감쌉니다.

## 라이선스

이 저장소의 코드는 MIT 라이선스입니다(`LICENSE`). DGQ와 Hybrid-SD 코드는 각 저장소의 조건을 따릅니다.

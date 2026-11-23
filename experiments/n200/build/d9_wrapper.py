# d9_wrapper.py — d5_wrapper.py(sha256 앞 12자 b069f9c4576d)의 N=200 변형. tools/make_notebook_d9.py가 맨 위에 EMBED·D9_TAG·D9_CFGS·D9_N 줄을 붙여 셀(.py)과 .ipynb를 만든다.
# 순서: 1 bridge 쓰기·sha 확인 → 2 d9_kaggle.py 쓰기·sha 확인 → 3 ckpt 찾기·크기 확인 → 4 GPU·인터넷·디스크 → 5 실행(스트리밍) → 6 보고서 경로
import base64, glob, hashlib, os, shutil, subprocess   # 표준 라이브러리만 (Kaggle 기본 이미지)
CKPT_NAME = "cali_ckpt_activation_w4a8g1.pth_merged"   # d9_kaggle.py CKPT_DEFAULT의 파일 이름
CKPT_SIZE = 6883907740                                 # d9_kaggle.py EXPECTED_SIZE (--ckpt 경로면 d5는 경고만 → 여기서 확인)
DGQ_URL = "https://github.com/ugonfor/DGQ"             # d9_kaggle.py DGQ_URL (S0 clone 대상)
MIN_FREE = 12 * 2**30

def d5_write(work):
    """1·2단계: EMBED의 bridge 7개와 d9_kaggle.py를 work 아래에 덮어쓰고 sha256 확인. 실패 사유 또는 None."""
    for rel, (want, b64) in sorted(EMBED.items()):                       # rel 예: bridge/run_split.py, d9_kaggle.py
        path = os.path.join(work, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)                # bridge/ 폴더가 없으면 만듦
        with open(path, "wb") as f:                                      # 덮어쓰기 → Run All 재실행에도 같은 결과
            f.write(base64.b64decode(b64))
        got = hashlib.sha256(open(path, "rb").read()).hexdigest()        # 디스크에 쓴 결과를 다시 읽어 확인
        print("  %s  %s" % (got[:12], rel), flush=True)
        if got != want:                                                  # 빌드 시 기대값과 다르면 중단
            return "sha 불일치 %s: %s != %s (노트북을 make_notebook.py로 다시 만드세요)" % (rel, got, want)
    return None

def d5_find_ckpt(inp):
    """3단계: inp 아래에서 ckpt를 찾고 크기 확인. (경로, None) 또는 (None, 사유)."""
    hits = sorted(glob.glob(os.path.join(inp, "**", CKPT_NAME), recursive=True))   # 데이터셋 마운트 경로가 달라도 찾음
    if not hits:
        return None, "ckpt(%s) 없음 → 데이터셋 dgq-sd14-w4a8g1-merged를 Add Data로 붙이세요" % CKPT_NAME
    print("  ckpt 후보 %d개 (정렬 순 첫 번째 사용): %s" % (len(hits), hits), flush=True)   # 어느 파일을 썼는지 기록
    size = os.path.getsize(hits[0])
    if size != CKPT_SIZE:                                                # 손상·다른 파일이면 GPU 쓰기 전에 중단
        return None, "ckpt 크기 %d != %d (%s) → 데이터셋 버전을 확인하세요" % (size, CKPT_SIZE, hits[0])
    return hits[0], None

def d5_rc0(cmd):   # 명령이 60초 안에 rc=0으로 끝나는가 (명령 없음·시간 초과는 False)
    try:
        return subprocess.run(cmd, capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False

def d5_env(work):
    """4단계: GPU·인터넷·디스크. 실패 사유 리스트(비면 통과)."""
    why = []
    if not d5_rc0(["nvidia-smi", "-L"]):                                 # nvidia-smi 없음/실패 → GPU 꺼짐
        why.append("GPU 없음 → Settings > Accelerator를 GPU T4 x2로 바꾸세요")
    else:                                                                # 예비 실험과 같은 이미지가 나오려면 같은 종류의 GPU(T4)여야 함
        try:
            gname = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=60).stdout
        except (OSError, subprocess.TimeoutExpired):
            gname = ""
        print("  GPU:", gname.strip().replace("\n", " | "), flush=True)
        if "T4" not in gname:
            why.append("GPU가 T4가 아님(%r) → Settings > Accelerator를 GPU T4 x2로 바꾸세요 (P100 등에서는 D6과 같은 이미지를 보장하지 않음)" % gname.strip())
    if not d5_rc0(["git", "ls-remote", DGQ_URL, "HEAD"]):                # S0의 git clone과 같은 대상에 접근 확인
        why.append("인터넷 접근 실패(%s) → Settings > Internet을 On으로 바꾸세요" % DGQ_URL)
    free = shutil.disk_usage(work).free                                  # /kaggle/working이 있는 파일시스템 기준
    if free < MIN_FREE:
        why.append("여유 디스크 %.1fGB < 12GB → /kaggle/working의 불필요한 파일을 지우세요" % (free / 2**30))
    return why

def d5_main(work="/kaggle/working", inp="/kaggle/input"):   # 실패는 print 후 return (sys.exit 안 함 → 노트북이 깨지지 않음)
    print("[1·2] EMBED %d개 쓰기 (%s)" % (len(EMBED), work), flush=True)
    err = d5_write(work)
    if err:
        return print("중단: " + err)
    print("[3] ckpt 찾기 (%s)" % inp, flush=True)
    ckpt, err = d5_find_ckpt(inp)
    if err:
        return print("중단: " + err)
    print("  ckpt %s (%d바이트)" % (ckpt, CKPT_SIZE), flush=True)
    print("[4] GPU·인터넷·디스크 확인", flush=True)
    why = d5_env(work)
    if why:
        return print("중단:\n  " + "\n  ".join(why))
    cmd = ["python3", "-u", os.path.join(work, "d9_kaggle.py"), "--ckpt", ckpt, "--budget-min", "360"]   # 시간 예산 360분
    env = dict(os.environ, D9_TAG=D9_TAG, D9_CFGS=D9_CFGS, D9_N=str(D9_N))   # 세션 설정은 노트북 빌더가 셀 맨 위에 쓴 변수
    print("[5] 실행: %s (cwd=%s)" % (" ".join(cmd), work), flush=True)
    p = subprocess.Popen(cmd, cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    try:
        for line in p.stdout:                                            # 출력 그대로 스트리밍
            print(line, end="", flush=True)
        rc = p.wait()
    except KeyboardInterrupt:                                            # 셀 중단 시 자식 프로세스도 정리
        p.terminate()
        rc = "중단됨(KeyboardInterrupt)"
    print("\n[6] d9_kaggle.py 종료 rc=%s\n보고서: %s\n내려받을 파일(/kaggle/working): d9_result_%s.json, d9_summary_%s.zip, d9_out/d9_report.md"
          % (rc, os.path.join(work, "d9_out", "d9_report.md"), D9_TAG, D9_TAG), flush=True)

if __name__ == "__main__":                                               # import만 하면 실행하지 않음
    try:                                                                 # 예기치 않은 예외도 print로 끝냄
        d5_main()
    except Exception as e:  # noqa: BLE001
        print("래퍼 예외: %r" % (e,))

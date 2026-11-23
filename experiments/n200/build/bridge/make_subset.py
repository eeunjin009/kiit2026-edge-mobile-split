# bridge/make_subset.py — metadata.csv에서 부분집합 CSV를 만든다 (N=200). 난수 라이브러리 없이 해시로 고른다.
# 규칙: 헤더를 뺀 행에서 앞 SKIP행(N=50 예비 실험 범위)을 제외하고, sha256("<seed>:<file_name>")이 작은 순서로 N행을 고른 뒤
#       원래 행 순서로 저장한다. 같은 입력이면 어느 파이썬 버전에서도 같은 결과다.
# 사용: python make_subset.py <원본 csv> <출력 csv> <N> <seed> <skip>
# 출력: <출력 csv>, <출력 csv>.idx.json (원본 행 번호·file_name), 마지막에 D5JSON 한 줄. 판정은 하지 않는다.
import csv, hashlib, io, json, sys


def pick(rows, n, seed, skip):
    """rows: 헤더를 뺀 행 목록. 반환: 선택된 행 번호(0부터, 오름차순)."""
    key = lambda i: (hashlib.sha256(("%d:%s" % (seed, rows[i][0])).encode("utf-8")).hexdigest(), i)
    return sorted(sorted(range(skip, len(rows)), key=key)[:n])


def main(argv):
    src, dst, n, seed, skip = argv[0], argv[1], int(argv[2]), int(argv[3]), int(argv[4])
    raw = open(src, "rb").read()
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8", errors="surrogateescape"), newline="")))   # utf-8이 아닌 바이트도 그대로 보존
    head, body = rows[0], rows[1:]
    assert len(body) > skip + n, (len(body), skip, n)
    assert all(len(r) == len(head) for r in body), "열 수가 헤더와 다른 행이 있음"
    idx = pick(body, n, seed, skip)
    names = [body[i][0] for i in idx]
    assert len(idx) == n and len(set(names)) == n, "file_name이 겹치거나 개수가 N이 아님"
    with open(dst, "w", newline="", encoding="utf-8", errors="surrogateescape") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(head)
        w.writerows(body[i] for i in idx)
    with open(dst + ".idx.json", "w", encoding="utf-8") as f:
        json.dump({"src_sha256": hashlib.sha256(raw).hexdigest(), "seed": seed, "skip": skip, "n": n,
                   "idx": idx, "file_names": names}, f)
    print("D5JSON:" + json.dumps(dict(src_sha256_12=hashlib.sha256(raw).hexdigest()[:12], src_rows=len(body), header=head, n=n,
                                      seed=seed, skip=skip, dst_sha256_12=hashlib.sha256(open(dst, "rb").read()).hexdigest()[:12],
                                      idx_first=idx[:5], idx_min=min(idx), idx_max=max(idx), names_first=names[:2])))


if __name__ == "__main__":
    main(sys.argv[1:])

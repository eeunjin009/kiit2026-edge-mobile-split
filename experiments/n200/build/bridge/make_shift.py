# bridge/make_shift.py — PSNR 기준선용 "다른 캡션 쌍" 폴더를 만든다. B0 이미지를 파일 이름 순으로 k칸 순환 이동해
#   <dst>/<i번째 이름> = <src>/<(i+k) mod N번째 이름> 으로 복사한다. 그러면 check_img.py <src> <dst> 가 B0[i]와 B0[(i+k) mod N]을 짝짓는다.
# 사용: python make_shift.py <src 폴더(B0 im512)> <dst 폴더> <k>   (k는 N의 배수가 아니어야 함)
# 출력: <dst>, <dst의 상위 폴더>/pairs.json (이름 → 원본 이름), 마지막에 D5JSON 한 줄. 판정은 하지 않는다.
import json, os, shutil, sys


def main(argv):
    src, dst, k = argv[0], argv[1], int(argv[2])
    names = sorted(os.listdir(src))
    n = len(names)
    assert n > 1 and k % n != 0, (n, k)
    if os.path.isdir(dst):
        shutil.rmtree(dst)                               # 이 스크립트가 만든 폴더만 지운다 (재실행용)
    os.makedirs(dst)
    pairs = {}
    for i, nm in enumerate(names):
        pairs[nm] = names[(i + k) % n]
        shutil.copyfile(os.path.join(src, pairs[nm]), os.path.join(dst, nm))
    with open(os.path.join(os.path.dirname(os.path.normpath(dst)), "pairs.json"), "w", encoding="utf-8") as f:
        json.dump(pairs, f)
    print("D5JSON:" + json.dumps(dict(n=n, k=k, n_self_pairs=sum(a == b for a, b in pairs.items()), first=[[a, pairs[a]] for a in names[:2]])))


if __name__ == "__main__":
    main(sys.argv[1:])

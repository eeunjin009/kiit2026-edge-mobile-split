# bridge/clip_per_image.py — DGQ src/eval_clip_score.py(원본)와 같은 계산을 하되 이미지별 점수를 저장한다. 원본은 평균만 쓴다.
# 같은 모델(ViT-g-14, laion2b_s34b_b88k)·전처리·정규화·내적·평균 식(sum/len)이다. 원본 파일은 수정하지 않는다.
# 사용: python clip_per_image.py --dgq_src <DGQ>/src --data_list <csv> --img_dir <im256> --num_imgs N --out <json>
#   --dgq_src: 원본 스크립트가 있는 폴더. 원본과 같이 그 폴더가 sys.path 맨 앞에 와서 `utils`가 DGQ/src/utils.py를 가리킨다.
# 출력: <json>(files, scores, sum_score, mean_score), stdout에 원본과 같은 형식의 "FINAL clip score ..." 줄. 판정은 하지 않는다.
import argparse, json, os, sys


def score_images(model, preprocess, tokenizer, torch, Image, file_list, img_dir, device):
    """원본 eval_clip_score.py의 반복문과 같은 순서·같은 연산. 반환: (파일 이름 목록, 점수 목록(numpy float32))."""
    names, score_arr = [], []
    for file_info in file_list:
        img_path = os.path.join(img_dir, file_info[0])
        text = tokenizer([file_info[1]]).to(device)
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        probs = (text_features.cpu().numpy() @ image_features.cpu().numpy().T)
        names.append(file_info[0])
        score_arr.append(probs[0][0])
    return names, score_arr


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dgq_src", required=True)
    ap.add_argument("--data_list", required=True)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--num_imgs", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)
    sys.path.insert(0, args.dgq_src)
    import torch, open_clip
    from PIL import Image
    from utils import get_file_list_from_csv
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s34b_b88k', device=args.device)
    tokenizer = open_clip.get_tokenizer('ViT-g-14')
    file_list = get_file_list_from_csv(args.data_list)
    if args.num_imgs is not None:
        file_list = file_list[:args.num_imgs]
    names, score_arr = score_images(model, preprocess, tokenizer, torch, Image, file_list, args.img_dir, args.device)
    final_score = sum(score_arr) / len(score_arr)               # 원본과 같은 식 (numpy 버전에 따른 누적 정밀도도 원본과 같게)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"model": "ViT-g-14/laion2b_s34b_b88k", "n": len(score_arr), "files": names,
                   "scores": [float(s) for s in score_arr],       # float32 → float64 변환은 값이 그대로 보존됨
                   "sum_score": float(sum(score_arr)), "mean_score": float(final_score)}, f)
    print(f"FINAL clip score {final_score}\n")                   # 원본과 같은 f-string 형식 (numpy 스칼라의 출력 자릿수까지 같게)
    print(f"-- sum score {sum(score_arr)}\n")
    print(f"-- len {len(score_arr)}\n")
    print("D5JSON:" + json.dumps(dict(n=len(score_arr), mean_score=float(final_score), out=args.out)))


if __name__ == "__main__":
    main()

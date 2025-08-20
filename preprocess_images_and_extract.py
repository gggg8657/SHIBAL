#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch

"""
이미지 전처리(ROI 크롭) -> 디스크 저장 -> 잘린 이미지로 특징 추출(.npy)까지 수행.
- 입력: image_segments.json
- ROI: 픽셀 좌표 x1,y1,x2,y2 (원본 이미지 해상도 기준)
- 출력 이미지: --out_images_dir
- 출력 특징: --out_features_dir (잘린 이미지 경로를 기준으로 npy 저장)
- 재추출 방지: 기존 파일 존재시 스킵
"""

# 간단 전처리 (X3D simple pipeline과 일치)
def preprocess_tensor(frames_np: np.ndarray, out_size=(320, 320)) -> torch.Tensor:
    # frames_np: (T, H, W, C) uint8
    T, H, W, C = frames_np.shape
    x = torch.from_numpy(frames_np).float() / 255.0  # (T,H,W,C)
    x = x.permute(0, 3, 1, 2)  # (T,C,H,W)
    # resize to out_size
    x = torch.nn.functional.interpolate(x, size=out_size, mode='bilinear', align_corners=False)  # (T,C,h,w)
    # normalize
    mean = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1)
    std = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1)
    x = (x - mean) / std
    x = x.unsqueeze(0)  # (1,T,C,h,w)
    # (B,T,C,H,W)->(B,C,T,H,W)
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    return x


def load_x3d_model(device: str = 'cuda', model_name='x3d_l'):
    model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
    model = model.eval().to(device)
    # 마지막 분류 레이어 제거
    del model.blocks[-1]
    return model


def extract_feature_from_image_path(model, device, img_path: str, out_npy: str, frames: int = 16):
    # 한 장 이미지를 T프레임으로 복제
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"이미지 열기 실패: {img_path}: {e}")
        return False
    img_np = np.array(img)
    frames_np = np.stack([img_np] * frames)  # (T,H,W,C)
    x = preprocess_tensor(frames_np, out_size=(320, 320))
    x = x.to(device)
    with torch.no_grad():
        feat = model(x)
    os.makedirs(os.path.dirname(out_npy), exist_ok=True)
    np.save(out_npy, feat.cpu().numpy())
    return True


def main():
    ap = argparse.ArgumentParser(description='이미지 ROI 크롭 및 특징 추출')
    ap.add_argument('--json_path', default='image_segments.json')
    ap.add_argument('--windows_prefix', default='D:/output_2025/', help='원본 윈도우 경로 접두사')
    ap.add_argument('--in_base', default='.', help='원본 이미지 루트 (윈도우 경로를 매핑할 로컬 루트)')
    ap.add_argument('--out_images_dir', default='cropped_images')
    ap.add_argument('--out_features_dir', default='features_cropped')
    ap.add_argument('--roi_pixels', default='0,0,480,270', help='픽셀 ROI x1,y1,x2,y2')
    ap.add_argument('--frames', type=int, default=16)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    x1, y1, x2, y2 = map(int, args.roi_pixels.split(','))

    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    model = load_x3d_model(device=device)

    with open(args.json_path, 'r', encoding='utf-8') as f:
        segments = json.load(f)

    total = 0
    done = 0
    for seg in segments:
        for img_win in seg.get('images', []):
            # 윈도우 경로를 상대 경로로 변환
            rel = img_win.replace('\\', '/').replace(args.windows_prefix, '')
            in_img = os.path.join(args.in_base, rel).replace('\\', '/')
            # 출력 경로
            out_img = os.path.join(args.out_images_dir, rel).replace('\\', '/')
            out_npy = os.path.join(args.out_features_dir, rel).replace('\\', '/')
            out_img = Path(out_img).with_suffix('.jpg').as_posix()
            out_npy = Path(out_npy).with_suffix('.npy').as_posix()

            total += 1
            # 스킵 조건
            if os.path.exists(out_npy):
                continue

            # 원본 로드 및 크롭
            try:
                os.makedirs(os.path.dirname(out_img), exist_ok=True)
                with Image.open(in_img).convert('RGB') as im:
                    W, H = im.size
                    cx1 = max(0, min(x1, W-1)); cx2 = max(1, min(x2, W))
                    cy1 = max(0, min(y1, H-1)); cy2 = max(1, min(y2, H))
                    if cx2 <= cx1 or cy2 <= cy1:
                        print(f"ROI 불가 스킵: {in_img} (W,H=({W},{H}))")
                        continue
                    crop = im.crop((cx1, cy1, cx2, cy2))
                    crop.save(out_img, quality=95)
            except Exception as e:
                print(f"크롭 실패: {in_img}: {e}")
                continue

            # 특징 추출
            ok = extract_feature_from_image_path(model, device, out_img, out_npy, frames=args.frames)
            if ok:
                done += 1
                if done % 100 == 0:
                    print(f"진행: {done}/{total} 저장")

    print(f"총 이미지: {total}, 저장된 특징: {done}")

if __name__ == '__main__':
    main()

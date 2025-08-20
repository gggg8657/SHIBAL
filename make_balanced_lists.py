#!/usr/bin/env python3
import argparse
import json
import os
import random
from collections import defaultdict

"""
image_segments.json을 읽어 normal=0, 그 외=1로 라벨링하고,
클래스 균형을 맞춰 train/test 리스트를 생성합니다.

리스트 포맷: feature_path|category|label|description
- feature_path는 --features_dir 아래에 상대 경로로 저장될 .npy 경로를 가리킵니다.
- 상대 경로 계산은 윈도우 경로에서 --windows_prefix (기본: D:\\output_2025\\) 를 제거하여 얻습니다.

주의: 이 스크립트는 feature 파일을 생성하지 않습니다. (preprocess_images_and_extract.py로 생성)
"""

CATEGORY_TO_LABEL = lambda c: 0 if str(c).strip().lower() == 'normal' else 1


def normalize_windows_path(p: str) -> str:
    return p.replace('\\', '/').replace('D:/', 'D:/').replace('d:/', 'D:/')


def to_relative(p: str, windows_prefix: str) -> str:
    p = normalize_windows_path(p)
    wp = normalize_windows_path(windows_prefix)
    if p.lower().startswith(wp.lower()):
        return p[len(wp):]
    # 접두사가 다르면 파일명만 사용
    return os.path.basename(p)


def build_items(segments, windows_prefix: str):
    items = []
    for seg in segments:
        category = seg.get('category', 'unknown')
        desc = seg.get('description', '')
        label = CATEGORY_TO_LABEL(category)
        for img in seg.get('images', []):
            rel = to_relative(img, windows_prefix)
            items.append({
                'rel_path': rel,
                'category': category,
                'label': label,
                'description': desc,
            })
    return items


def split_balanced(items, test_ratio: float):
    by_label = defaultdict(list)
    for it in items:
        by_label[it['label']].append(it)
    # 균형 수
    min_count = min(len(by_label[0]), len(by_label[1]))
    # 셔플
    for k in by_label:
        random.shuffle(by_label[k])
    # 동일 수로 자르기
    sel0 = by_label[0][:min_count]
    sel1 = by_label[1][:min_count]
    balanced = sel0 + sel1
    random.shuffle(balanced)
    # train/test 분할
    n_test_per_class = int(min_count * test_ratio)
    test = sel0[:n_test_per_class] + sel1[:n_test_per_class]
    train = sel0[n_test_per_class:] + sel1[n_test_per_class:]
    random.shuffle(train); random.shuffle(test)
    return train, test


def write_list(items, features_dir: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for it in items:
            rel_img = it['rel_path']
            rel_npy = rel_img.rsplit('.', 1)[0] + '.npy'
            feat_path = os.path.join(features_dir, rel_npy).replace('\\', '/').replace('//', '/')
            f.write(f"{feat_path}|{it['category']}|{it['label']}|{it['description']}\n")


def main():
    ap = argparse.ArgumentParser(description='균형 데이터 리스트 생성 (normal vs abnormal, 50:50)')
    ap.add_argument('--json_path', default='image_segments.json')
    ap.add_argument('--windows_prefix', default='D:/output_2025/', help='원본 윈도우 경로 접두사')
    ap.add_argument('--features_dir', default='features_cropped', help='리스트가 가리킬 특징 디렉토리')
    ap.add_argument('--out_dir', default='custom_data', help='리스트 출력 디렉토리')
    ap.add_argument('--test_ratio', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=2025)
    args = ap.parse_args()

    random.seed(args.seed)

    with open(args.json_path, 'r', encoding='utf-8') as f:
        segments = json.load(f)

    items = build_items(segments, args.windows_prefix)
    train, test = split_balanced(items, args.test_ratio)

    train_path = os.path.join(args.out_dir, 'balanced_train.txt')
    test_path = os.path.join(args.out_dir, 'balanced_test.txt')
    write_list(train, args.features_dir, train_path)
    write_list(test, args.features_dir, test_path)

    print(f"normal vs abnormal 50:50로 분할 완료")
    print(f"train: {len(train)}개, test: {len(test)}개")
    print(f"저장: {train_path}, {test_path}")

if __name__ == '__main__':
    main()

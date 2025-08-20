#!/usr/bin/env python3
import argparse
import os

"""
기존 feature 리스트(예: custom_data/custom_test.txt)를 읽어,
category가 'normal'인 경우만 0, 그 외 전부 1로 라벨을 재설정하여
새로운 리스트 파일로 저장합니다. (feature 재추출 없이 사용)

입력 포맷: feature_path|category|label|description
또는 경로만 포함된 라인도 허용: feature_path
"""

def remap_labels(input_list: str, output_list: str):
    if not os.path.exists(input_list):
        raise FileNotFoundError(f"입력 리스트가 없습니다: {input_list}")

    os.makedirs(os.path.dirname(output_list), exist_ok=True)

    num_total = 0
    num_normal = 0
    num_anomaly = 0

    with open(input_list, 'r', encoding='utf-8') as fin, open(output_list, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            num_total += 1
            parts = line.split('|')
            if len(parts) >= 4:
                feature_path, category, _label, description = parts[:4]
                is_normal = str(category).strip().lower() == 'normal'
                new_label = '0' if is_normal else '1'
                if is_normal:
                    num_normal += 1
                else:
                    num_anomaly += 1
                fout.write(f"{feature_path}|{category}|{new_label}|{description}\n")
            else:
                # 경로만 있는 경우, 경로명에 Normal 포함 여부로 판정
                feature_path = parts[0]
                is_normal = ('normal' in feature_path.lower())
                new_label = '0' if is_normal else '1'
                if is_normal:
                    num_normal += 1
                else:
                    num_anomaly += 1
                fout.write(f"{feature_path}|unknown|{new_label}|auto\n")

    print(f"총 {num_total}개 라인 처리 완료")
    print(f"normal(0): {num_normal}개, abnormal(1): {num_anomaly}개")
    print(f"저장: {output_list}")


def main():
    parser = argparse.ArgumentParser(description='리스트 라벨 재설정: normal만 0, 나머지 전부 1')
    parser.add_argument('--input_list', default='custom_data/custom_test.txt', help='입력 리스트 경로')
    parser.add_argument('--output_list', default='custom_data/custom_test_non_normal_is_anomaly.txt', help='출력 리스트 경로')
    args = parser.parse_args()

    remap_labels(args.input_list, args.output_list)

if __name__ == '__main__':
    main()

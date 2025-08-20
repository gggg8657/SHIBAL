#!/usr/bin/env python3
"""
세그먼트 기반 이미지 프리프로세싱 스크립트
STEAD 모델 훈련을 위한 특징 추출 및 데이터 준비
"""

import argparse
import os
import sys
from segment_parser import SegmentParser
from segment_feat_extractor import SegmentFeatureExtractor

def main():
    parser = argparse.ArgumentParser(description='세그먼트 기반 이미지 프리프로세싱')
    parser.add_argument('--json_path', default='image_segments.json', help='세그먼트 JSON 파일 경로')
    parser.add_argument('--output_dir', default='./custom_data', help='출력 디렉토리')
    parser.add_argument('--features_dir', default='./features', help='특징 저장 디렉토리')
    parser.add_argument('--model_name', default='x3d_l', help='X3D 모델명 (x3d_l, x3d_m, x3d_s)')
    parser.add_argument('--skip_features', action='store_true', help='특징 추출 건너뛰기 (리스트만 생성)')
    
    args = parser.parse_args()
    
    print("=== 세그먼트 기반 프리프로세싱 시작 ===")
    print(f"JSON 파일: {args.json_path}")
    print(f"출력 디렉토리: {args.output_dir}")
    print(f"특징 디렉토리: {args.features_dir}")
    print(f"모델명: {args.model_name}")
    
    # 1. JSON 파일 존재 확인
    if not os.path.exists(args.json_path):
        print(f"❌ JSON 파일을 찾을 수 없습니다: {args.json_path}")
        print("image_segments.json 파일이 현재 디렉토리에 있는지 확인하세요.")
        return
    
    # 2. 세그먼트 파서 생성
    print("\n1. 세그먼트 정보 파싱...")
    try:
        segment_parser = SegmentParser(args.json_path)
        print(f"✅ 총 {len(segment_parser.segments)}개 세그먼트 로드 완료")
    except Exception as e:
        print(f"❌ 세그먼트 파싱 실패: {e}")
        return
    
    # 3. 훈련/테스트 리스트 생성
    print("\n2. 데이터 리스트 생성...")
    try:
        train_list, test_list = segment_parser.create_training_lists(args.output_dir)
        print(f"✅ 훈련 데이터: {len(train_list)}개")
        print(f"✅ 테스트 데이터: {len(test_list)}개")
    except Exception as e:
        print(f"❌ 데이터 리스트 생성 실패: {e}")
        return
    
    # 4. 특징 추출 (선택사항)
    if not args.skip_features:
        print("\n3. 특징 추출...")
        try:
            extractor = SegmentFeatureExtractor(args.json_path, args.model_name)
            if extractor.model is None:
                print("❌ 모델 로드 실패. PyTorchVideo 설치가 필요합니다.")
                print("pip install pytorchvideo torchvision")
                return
            
            features_info = extractor.extract_features_from_segments(args.features_dir)
            print(f"✅ 총 특징 추출: {len(features_info)}개")
        except Exception as e:
            print(f"❌ 특징 추출 실패: {e}")
            print("특징 추출을 건너뛰고 리스트만 생성합니다.")
    else:
        print("\n3. 특징 추출 건너뛰기 (--skip_features 옵션)")
    
    # 5. 카테고리별 통계 출력
    print("\n=== 카테고리별 분포 ===")
    categories = {}
    for segment in segment_parser.segments:
        cat = segment['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    for cat, count in sorted(categories.items()):
        print(f"{cat}: {count}개 세그먼트")
    
    # 6. 다음 단계 안내
    print("\n=== 프리프로세싱 완료 ===")
    print(f"📁 데이터 리스트: {args.output_dir}/")
    if not args.skip_features:
        print(f"📁 특징 파일: {args.features_dir}/")
    
    print("\n=== 다음 단계 ===")
    print("1. 특징 추출이 완료되었다면:")
    print(f"   python train_custom.py --rgb_list {args.output_dir}/custom_train.txt --test_rgb_list {args.output_dir}/custom_test.txt")
    print("\n2. 특징 추출이 필요하다면:")
    print("   pip install pytorchvideo torchvision")
    print("   python preprocess_segments.py (--skip_features 옵션 제거)")
    
    print("\n3. 테스트:")
    print("   python test_custom.py --model_path saved_models/888tiny.pkl")

if __name__ == "__main__":
    main()

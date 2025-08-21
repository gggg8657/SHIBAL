#!/usr/bin/env python3
"""
데이터 리스트 파일 검증 스크립트
실제로 읽어올 수 있는 데이터 개수를 확인합니다.
"""

import os
import json
from collections import defaultdict

def load_config(config_path='config.json'):
    """config.json 파일을 로드합니다."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✅ 설정 파일 로드: {config_path}")
        return config
    except FileNotFoundError:
        print(f"❌ 설정 파일을 찾을 수 없습니다: {config_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ JSON 파싱 오류: {e}")
        return None

def verify_data_list(file_path, name="데이터"):
    """데이터 리스트 파일을 검증합니다."""
    print(f"\n=== {name} 검증 ===")
    print(f"파일 경로: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ 파일이 존재하지 않습니다: {file_path}")
        return None
    
    # 파일 통계
    total_lines = 0
    valid_lines = 0
    missing_files = 0
    category_counts = defaultdict(int)
    label_counts = defaultdict(int)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # 빈 줄 건너뛰기
                    continue
                
                total_lines += 1
                
                # 파이프(|)로 분리된 형식 파싱
                parts = line.split('|')
                if len(parts) >= 3:
                    feature_path = parts[0].strip()
                    category = parts[1].strip()
                    label = parts[2].strip()
                    
                    # 카테고리와 라벨 카운트
                    category_counts[category] += 1
                    label_counts[label] += 1
                    
                    # 파일 존재 여부 확인
                    if os.path.exists(feature_path):
                        valid_lines += 1
                    else:
                        missing_files += 1
                        if missing_files <= 5:  # 처음 5개만 출력
                            print(f"  ❌ 파일 없음: {feature_path}")
                else:
                    print(f"  ⚠️ 잘못된 형식 (라인 {line_num}): {line}")
        
        # 결과 출력
        print(f"📊 총 라인 수: {total_lines:,}")
        print(f"✅ 유효한 파일: {valid_lines:,}")
        print(f"❌ 누락된 파일: {missing_files:,}")
        print(f"📈 유효 비율: {valid_lines/total_lines*100:.1f}%")
        
        # 카테고리별 분포
        print(f"\n📋 카테고리별 분포:")
        for category, count in sorted(category_counts.items()):
            print(f"  {category}: {count:,}개")
        
        # 라벨별 분포
        print(f"\n🏷️ 라벨별 분포:")
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count:,}개")
        
        return {
            'total_lines': total_lines,
            'valid_files': valid_lines,
            'missing_files': missing_files,
            'valid_ratio': valid_lines/total_lines*100,
            'categories': dict(category_counts),
            'labels': dict(label_counts)
        }
        
    except Exception as e:
        print(f"❌ 파일 읽기 오류: {e}")
        return None

def main():
    """메인 함수"""
    print("🔍 데이터 리스트 파일 검증 시작")
    
    # 설정 파일 로드
    config = load_config()
    if config is None:
        print("❌ config.json 파일을 찾을 수 없습니다.")
        print("config.json 파일을 생성하고 데이터 경로를 설정하세요.")
        return
    
    train_file = config['data']['train_list']
    test_file = config['data']['test_list']
    
    print(f"📁 설정된 데이터 경로:")
    print(f"  훈련: {train_file}")
    print(f"  테스트: {test_file}")
    
    # 훈련 데이터 검증
    train_stats = verify_data_list(train_file, "훈련 데이터")
    
    # 테스트 데이터 검증
    test_stats = verify_data_list(test_file, "테스트 데이터")
    
    # 전체 요약
    if train_stats and test_stats:
        print(f"\n📊 전체 요약")
        print(f"훈련 데이터: {train_stats['valid_files']:,}개 (유효 비율: {train_stats['valid_ratio']:.1f}%)")
        print(f"테스트 데이터: {test_stats['valid_files']:,}개 (유효 비율: {test_stats['valid_ratio']:.1f}%)")
        print(f"총 데이터: {train_stats['valid_files'] + test_stats['valid_files']:,}개")
        
        if train_stats['valid_files'] > 0 and test_stats['valid_files'] > 0:
            ratio = test_stats['valid_files'] / train_stats['valid_files']
            print(f"테스트/훈련 비율: {ratio:.2f}:1")
            
            if ratio > 5:
                print("⚠️ 경고: 테스트 데이터가 훈련 데이터보다 5배 이상 많습니다!")
                print("   데이터 분할을 재검토하는 것을 권장합니다.")
        elif train_stats['valid_files'] == 0 and test_stats['valid_files'] == 0:
            print("❌ 모든 데이터 파일이 누락되었습니다!")
            print("   특징 파일을 생성하거나 경로를 확인하세요.")
    
    print(f"\n✅ 검증 완료!")

if __name__ == "__main__":
    main()

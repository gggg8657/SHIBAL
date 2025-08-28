import os
import json
import argparse
import random
from pathlib import Path

def load_image_segments(json_path):
    """image_segments.json 파일 로드"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"[INFO] {len(data)}개 세그먼트 로드됨")
        return data
    except Exception as e:
        print(f"❌ JSON 파일 로드 실패: {e}")
        return None

def create_image_lists(segments_data, output_dir, train_ratio=0.7, test_ratio=0.2, valid_ratio=0.1):
    """이미지 리스트를 train/test/valid로 나누어 생성"""
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 정상/비정상 데이터 분리
    normal_images = []
    abnormal_images = []
    
    for segment in segments_data:
        try:
            # 이미지 경로 추출
            image_path = segment.get('image_path', '')
            if not image_path:
                continue
            
            # 라벨 추출
            label = segment.get('label', 'normal')
            category = segment.get('category', 'unknown')
            
            # Windows 경로를 Unix 경로로 변환
            image_path = image_path.replace('\\', '/')
            
            # 절대 경로인 경우 상대 경로로 변환
            if image_path.startswith('C:/') or image_path.startswith('D:/'):
                # 상위 디렉토리에서 이미지 찾기
                relative_path = None
                for root, dirs, files in os.walk('..'):
                    for file in files:
                        if file in image_path or os.path.basename(image_path) == file:
                            relative_path = os.path.join(root, file)
                            break
                    if relative_path:
                        break
                
                if relative_path:
                    image_path = relative_path
                else:
                    print(f"⚠️ 이미지 파일을 찾을 수 없음: {image_path}")
                    continue
            
            # 이미지 파일 존재 확인
            if not os.path.exists(image_path):
                print(f"⚠️ 이미지 파일이 존재하지 않음: {image_path}")
                continue
            
            # 이미지 확장자 확인
            if not image_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                print(f"⚠️ 지원하지 않는 이미지 형식: {image_path}")
                continue
            
            # 라벨을 숫자로 변환
            if label.lower() in ['normal', '정상']:
                label_num = 0
                normal_images.append((image_path, label_num, category))
            elif label.lower() in ['abnormal', '비정상', 'anomaly', 'violence', 'abnormal movement', 'baggage movement', 'collapse', 'suspicious behavior']:
                label_num = 1
                abnormal_images.append((image_path, label_num, category))
            else:
                print(f"⚠️ 알 수 없는 라벨: {label}")
                continue
                
        except Exception as e:
            print(f"❌ 세그먼트 처리 실패: {e}")
            continue
    
    print(f"\n=== 데이터 분류 완료 ===")
    print(f"정상 이미지: {len(normal_images)}개")
    print(f"비정상 이미지: {len(abnormal_images)}개")
    print(f"총 이미지: {len(normal_images) + len(abnormal_images)}개")
    
    # 데이터 분할
    random.seed(2025)  # 재현성을 위한 시드 설정
    
    # 정상 데이터 분할
    random.shuffle(normal_images)
    normal_train_end = int(len(normal_images) * train_ratio)
    normal_valid_end = normal_train_end + int(len(normal_images) * valid_ratio)
    
    normal_train = normal_images[:normal_train_end]
    normal_valid = normal_images[normal_train_end:normal_valid_end]
    normal_test = normal_images[normal_valid_end:]
    
    # 비정상 데이터 분할
    random.shuffle(abnormal_images)
    abnormal_train_end = int(len(abnormal_images) * train_ratio)
    abnormal_valid_end = abnormal_train_end + int(len(abnormal_images) * valid_ratio)
    
    abnormal_train = abnormal_images[:abnormal_train_end]
    abnormal_valid = abnormal_images[abnormal_train_end:abnormal_valid_end]
    abnormal_test = abnormal_images[abnormal_valid_end:]
    
    print(f"\n=== 데이터 분할 결과 ===")
    print(f"훈련: 정상 {len(normal_train)}개, 비정상 {len(abnormal_train)}개 (총 {len(normal_train) + len(abnormal_train)}개)")
    print(f"검증: 정상 {len(normal_valid)}개, 비정상 {len(abnormal_valid)}개 (총 {len(normal_valid) + len(abnormal_valid)}개)")
    print(f"테스트: 정상 {len(normal_test)}개, 비정상 {len(abnormal_test)}개 (총 {len(normal_test) + len(abnormal_test)}개)")
    
    # 리스트 파일 생성
    def write_list_file(filename, data_list):
        with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
            for image_path, label, category in data_list:
                f.write(f"{image_path}|{label}|{category}\n")
    
    # 훈련 데이터 (정상 + 비정상)
    train_data = normal_train + abnormal_train
    random.shuffle(train_data)  # 훈련 데이터 섞기
    write_list_file('original_images_train.txt', train_data)
    
    # 검증 데이터 (정상 + 비정상)
    valid_data = normal_valid + abnormal_valid
    random.shuffle(valid_data)  # 검증 데이터 섞기
    write_list_file('original_images_valid.txt', valid_data)
    
    # 테스트 데이터 (정상 + 비정상)
    test_data = normal_test + abnormal_test
    random.shuffle(test_data)  # 테스트 데이터 섞기
    write_list_file('original_images_test.txt', test_data)
    
    print(f"\n=== 리스트 파일 생성 완료 ===")
    print(f"훈련: {os.path.join(output_dir, 'original_images_train.txt')}")
    print(f"검증: {os.path.join(output_dir, 'original_images_valid.txt')}")
    print(f"테스트: {os.path.join(output_dir, 'original_images_test.txt')}")

def main():
    parser = argparse.ArgumentParser(description='원본 이미지 리스트를 train/test/valid로 나누기')
    parser.add_argument('--json', type=str, default='../image_segments.json', help='image_segments.json 파일 경로')
    parser.add_argument('--output_dir', type=str, default='custom_data', help='출력 디렉토리')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='훈련 데이터 비율 (기본값: 0.7)')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='테스트 데이터 비율 (기본값: 0.2)')
    parser.add_argument('--valid_ratio', type=float, default=0.1, help='검증 데이터 비율 (기본값: 0.1)')
    
    args = parser.parse_args()
    
    # 비율 합계 확인
    if abs(args.train_ratio + args.test_ratio + args.valid_ratio - 1.0) > 0.01:
        print("❌ 비율의 합이 1.0이 되어야 합니다!")
        print(f"현재: {args.train_ratio} + {args.test_ratio} + {args.valid_ratio} = {args.train_ratio + args.test_ratio + args.valid_ratio}")
        return
    
    # JSON 파일 존재 확인
    if not os.path.exists(args.json):
        print(f"❌ JSON 파일을 찾을 수 없습니다: {args.json}")
        return
    
    # 세그먼트 데이터 로드
    segments_data = load_image_segments(args.json)
    if segments_data is None:
        return
    
    # 이미지 리스트 생성
    create_image_lists(segments_data, args.output_dir, args.train_ratio, args.test_ratio, args.valid_ratio)
    
    print(f"\n✅ 완료! 이제 End-to-End 학습을 사용할 수 있습니다.")
    print(f"사용법:")
    print(f"  python train_custom_e2e.py --config config_e2e_tiny.json")

if __name__ == "__main__":
    main()

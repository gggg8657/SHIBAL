import os
import json
import shutil
import argparse
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

def create_image_list(segments_data, output_dir, output_list_path):
    """이미지 리스트 파일 생성 및 이미지 복사"""
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 이미지 리스트 파일 생성
    with open(output_list_path, 'w', encoding='utf-8') as f:
        normal_count = 0
        abnormal_count = 0
        
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
                
                # 출력 이미지 경로
                filename = os.path.basename(image_path)
                output_image_path = os.path.join(output_dir, filename)
                
                # 이미지 복사
                try:
                    shutil.copy2(image_path, output_image_path)
                    
                    # 라벨을 숫자로 변환
                    if label.lower() in ['normal', '정상']:
                        label_num = 0
                        normal_count += 1
                    elif label.lower() in ['abnormal', '비정상', 'anomaly']:
                        label_num = 1
                        abnormal_count += 1
                    else:
                        print(f"⚠️ 알 수 없는 라벨: {label}")
                        continue
                    
                    # 리스트 파일에 추가
                    f.write(f"{output_image_path}|{label_num}|{category}\n")
                    
                except Exception as e:
                    print(f"❌ 이미지 복사 실패: {image_path} -> {output_image_path}, 에러: {e}")
                    continue
                
            except Exception as e:
                print(f"❌ 세그먼트 처리 실패: {e}")
                continue
    
    print(f"\n=== 이미지 생성 완료 ===")
    print(f"정상 이미지: {normal_count}개")
    print(f"비정상 이미지: {abnormal_count}개")
    print(f"총 이미지: {normal_count + abnormal_count}개")
    print(f"출력 디렉토리: {output_dir}")
    print(f"리스트 파일: {output_list_path}")

def main():
    parser = argparse.ArgumentParser(description='원본 이미지 파일 생성')
    parser.add_argument('--json', type=str, default='../image_segments.json', help='image_segments.json 파일 경로')
    parser.add_argument('--output_dir', type=str, default='original_images', help='출력 이미지 디렉토리')
    parser.add_argument('--output_list', type=str, default='custom_data/original_images.txt', help='출력 리스트 파일 경로')
    
    args = parser.parse_args()
    
    # JSON 파일 존재 확인
    if not os.path.exists(args.json):
        print(f"❌ JSON 파일을 찾을 수 없습니다: {args.json}")
        return
    
    # 세그먼트 데이터 로드
    segments_data = load_image_segments(args.json)
    if segments_data is None:
        return
    
    # 이미지 리스트 생성
    create_image_list(segments_data, args.output_dir, args.output_list)
    
    print(f"\n✅ 완료! 이제 End-to-End 학습을 사용할 수 있습니다.")
    print(f"사용법:")
    print(f"  python train_custom_e2e.py --config config_e2e_tiny.json")

if __name__ == "__main__":
    main()

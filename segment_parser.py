import json
import os
import numpy as np
from pathlib import Path

class SegmentParser:
    def __init__(self, json_path):
        self.json_path = json_path
        self.segments = self.load_segments()
        self.category_mapping = self.create_category_mapping()
    
    def load_segments(self):
        """JSON 파일에서 세그먼트 정보 로드"""
        with open(self.json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def create_category_mapping(self):
        """카테고리를 라벨로 매핑"""
        return {
            'normal': 0,
            'rest': 0,  # 정상 상태로 분류
            'violence': 1,
            'abnormal movement': 1,
            'baggage movement': 1,
            'collapse': 1,
            'suspicious behavior': 1,
            'unknown': 1  # 알 수 없는 것은 비정상으로 분류
        }
    
    def get_image_info(self, image_path):
        """이미지 경로에 해당하는 세그먼트 정보 반환"""
        for segment in self.segments:
            if image_path in segment['images']:
                return {
                    'category': segment['category'],
                    'label': self.category_mapping.get(segment['category'], 1),
                    'description': segment['description'],
                    'description_en': segment['description_en'],
                    'start_time': segment['start_time'],
                    'start_frame': segment['start_frame']
                }
        return None
    
    def create_training_lists(self, output_dir="./custom_data"):
        """STEAD 훈련용 데이터 리스트 생성"""
        os.makedirs(output_dir, exist_ok=True)
        
        train_list = []
        test_list = []
        
        for segment in self.segments:
            category = segment['category']
            label = self.category_mapping.get(category, 1)
            
            for img_path in segment['images']:
                # Windows 경로를 Unix 경로로 변환
                unix_path = img_path.replace('\\', '/').replace('D:/output_2025/', '')
                
                # 특징 파일 경로 (나중에 생성될)
                feature_path = f"features/{unix_path.replace('.jpg', '.npy')}"
                
                info = f"{feature_path}|{category}|{label}|{segment['description']}"
                
                # 80% 훈련, 20% 테스트로 분할
                if np.random.random() < 0.8:
                    train_list.append(info)
                else:
                    test_list.append(info)
        
        # 리스트 파일 저장
        with open(f"{output_dir}/custom_train.txt", 'w', encoding='utf-8') as f:
            for item in train_list:
                f.write(f"{item}\n")
        
        with open(f"{output_dir}/custom_test.txt", 'w', encoding='utf-8') as f:
            for item in test_list:
                f.write(f"{item}\n")
        
        print(f"훈련 데이터: {len(train_list)}개")
        print(f"테스트 데이터: {len(test_list)}개")
        
        return train_list, test_list

if __name__ == "__main__":
    # 테스트 실행
    parser = SegmentParser("image_segments.json")
    train_list, test_list = parser.create_training_lists()

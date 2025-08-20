import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import json
import os
from pathlib import Path
from segment_parser import SegmentParser

class SegmentFeatureExtractor:
    def __init__(self, json_path, model_name='x3d_l'):
        self.segment_parser = SegmentParser(json_path)
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_model()
        self.transform = self.create_transform()
        
    def load_model(self):
        """X3D 모델 로드"""
        try:
            model = torch.hub.load('facebookresearch/pytorchvideo', self.model_name, pretrained=True)
            model = model.eval().to(self.device)
            
            # 마지막 분류 레이어 제거
            del model.blocks[-1]
            return model
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            print("PyTorchVideo가 설치되지 않았습니다. pip install pytorchvideo로 설치하세요.")
            return None
    
    def create_transform(self):
        """전처리 변환 생성"""
        try:
            from torchvision.transforms.v2 import CenterCrop, Normalize
            from torchvision.transforms import Compose, Lambda
            from pytorchvideo.transforms import (
                ApplyTransformToKey,
                ShortSideScale,
                UniformTemporalSubsample
            )
        except ImportError:
            print("필요한 라이브러리가 설치되지 않았습니다.")
            return None
        
        # X3D 모델별 파라미터
        transform_params = {
            "x3d_l": {
                "side_size": 320,
                "crop_size": 320,
                "num_frames": 16,
                "sampling_rate": 5,
            },
            "x3d_m": {
                "side_size": 256,
                "crop_size": 256,
                "num_frames": 16,
                "sampling_rate": 5,
            },
            "x3d_s": {
                "side_size": 182,
                "crop_size": 182,
                "num_frames": 13,
                "sampling_rate": 6,
            }
        }
        
        params = transform_params.get(self.model_name, transform_params["x3d_l"])
        
        class Permute(nn.Module):
            def __init__(self, dims):
                super().__init__()
                self.dims = dims
            def forward(self, x):
                return torch.permute(x, self.dims)
        
        return ApplyTransformToKey(
            key="video",
            transform=Compose([
                UniformTemporalSubsample(params["num_frames"]),
                Lambda(lambda x: x/255.0),
                Permute((1, 0, 2, 3)),
                Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
                ShortSideScale(size=params["side_size"]),
                CenterCrop((params["crop_size"], params["crop_size"])),
                Permute((1, 0, 2, 3))
            ]),
        )
    
    def extract_features_from_segments(self, output_dir="./features"):
        """세그먼트별로 특징 추출"""
        if self.model is None or self.transform is None:
            print("모델 또는 변환이 로드되지 않았습니다.")
            return []
        
        os.makedirs(output_dir, exist_ok=True)
        
        features_info = []
        
        for segment in tqdm(self.segment_parser.segments, desc="세그먼트 처리"):
            category = segment['category']
            label = self.segment_parser.category_mapping.get(category, 1)
            
            segment_features = []
            
            for img_path in segment['images']:
                try:
                    # 이미지 경로를 Unix 경로로 변환
                    unix_path = img_path.replace('\\', '/').replace('D:/output_2025/', '')
                    feature_path = f"{output_dir}/{unix_path.replace('.jpg', '.npy')}"
                    
                    # 이미 처리된 경우 스킵
                    if os.path.exists(feature_path):
                        continue
                    
                    # 이미지 로드 및 특징 추출
                    feature = self.extract_single_image_feature(img_path)
                    
                    if feature is not None:
                        # 특징 저장
                        os.makedirs(os.path.dirname(feature_path), exist_ok=True)
                        np.save(feature_path, feature)
                        
                        segment_features.append({
                            'path': unix_path,
                            'feature_path': feature_path,
                            'category': category,
                            'label': label,
                            'description': segment['description']
                        })
                
                except Exception as e:
                    print(f"이미지 처리 실패: {img_path}, 에러: {e}")
                    continue
            
            if segment_features:
                features_info.extend(segment_features)
        
        return features_info
    
    def extract_single_image_feature(self, img_path):
        """단일 이미지에서 특징 추출"""
        try:
            from PIL import Image
            import torchvision.transforms as transforms
            
            # 이미지 로드
            image = Image.open(img_path).convert('RGB')
            
            # 이미지를 비디오 형태로 변환 (단일 프레임을 여러 번 복제)
            frames = [np.array(image)] * 16  # 16프레임으로 확장
            video_tensor = torch.from_numpy(np.stack(frames)).float()
            video_tensor = video_tensor.permute(0, 3, 1, 2).unsqueeze(0)  # (1, 16, 3, H, W)
            
            # 전처리 적용
            video_data = {'video': video_tensor}
            processed_video = self.transform(video_data)['video'].to(self.device)
            
            # 특징 추출
            with torch.no_grad():
                features = self.model(processed_video)
                return features.cpu().numpy()
                
        except Exception as e:
            print(f"특징 추출 실패: {img_path}, 에러: {e}")
            return None

if __name__ == "__main__":
    # 테스트 실행
    extractor = SegmentFeatureExtractor("image_segments.json")
    features_info = extractor.extract_features_from_segments()
    print(f"총 특징 추출: {len(features_info)}개")

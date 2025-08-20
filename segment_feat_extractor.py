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
        
        # 수정된 Permute 클래스 - 차원 검증 및 자동 수정
        class Permute(nn.Module):
            def __init__(self, dims):
                super().__init__()
                self.dims = dims

            def forward(self, x):
                # 차원 수 검증 및 자동 수정
                if x.dim() != len(self.dims):
                    print(f"경고: 입력 텐서 차원 {x.dim()}과 permute 차원 {len(self.dims)}이 일치하지 않습니다.")
                    print(f"입력 텐서 형태: {x.shape}")
                    
                    # 차원 수에 맞게 조정
                    if x.dim() == 5:  # (B, T, C, H, W) 또는 (B, C, T, H, W)
                        if len(self.dims) == 4:
                            # 5차원을 4차원으로 변환
                            B, dim1, dim2, H, W = x.shape
                            # 채널 수가 3인지 확인하여 올바른 순서 결정
                            if dim1 == 3:  # (B, C, T, H, W)
                                x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
                                x = x.view(B * dim2, dim1, H, W)  # (B*T, C, H, W)
                            elif dim2 == 3:  # (B, T, C, H, W)
                                x = x.view(B * dim1, dim2, H, W)  # (B*T, C, H, W)
                            else:
                                # 채널 수가 3이 아닌 경우, 첫 번째 차원을 시간으로 간주
                                x = x.view(B * dim1, dim2, H, W)  # (B*T, C, H, W)
                            return x.permute(self.dims)
                        else:
                            return x.permute(self.dims)
                    elif x.dim() == 4:  # (B, C, H, W)
                        if len(self.dims) == 5:
                            # 4차원을 5차원으로 변환
                            B, C, H, W = x.shape
                            x = x.unsqueeze(1)  # (B, 1, C, H, W)
                            return x.permute(self.dims)
                        else:
                            return x.permute(self.dims)
                    else:
                        return x.permute(self.dims)
                return x.permute(self.dims)
        
        # 안전한 transform 생성
        try:
            transform = ApplyTransformToKey(
                key="video",
                transform=Compose([
                    UniformTemporalSubsample(params["num_frames"]),
                    Lambda(lambda x: x/255.0),
                    Permute((1, 0, 2, 3)),  # (T, C, H, W) -> (C, T, H, W)
                    Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
                    ShortSideScale(size=params["side_size"]),
                    CenterCrop((params["crop_size"], params["crop_size"])),
                    Permute((1, 0, 2, 3))   # (C, T, H, W) -> (T, C, H, W)
                ]),
            )
            return transform
        except Exception as e:
            print(f"Transform 생성 실패: {e}")
            # 대체 transform 생성
            return self.create_simple_transform(params)
    
    def create_simple_transform(self, params):
        """간단한 transform 생성 (차원 문제 해결)"""
        from torchvision.transforms import Compose, Lambda, CenterCrop, Normalize
        from torchvision.transforms.functional import resize
        
        class SimpleTransform:
            def __init__(self, params):
                self.params = params
            
            def __call__(self, x):
                # x는 (B, T, C, H, W) 또는 (B, C, T, H, W) 형태
                B, dim1, dim2, H, W = x.shape
                
                # 채널 수가 3인지 확인하여 올바른 순서 결정
                if dim1 == 3:  # (B, C, T, H, W)
                    x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
                    T, C = dim2, dim1
                elif dim2 == 3:  # (B, T, C, H, W)
                    T, C = dim1, dim2
                else:
                    # 채널 수가 3이 아닌 경우, 첫 번째 차원을 시간으로 간주
                    T, C = dim1, dim2
                
                # 정규화
                x = x / 255.0
                
                # 크기 조정
                x = x.view(B * T, C, H, W)
                x = resize(x, [self.params["side_size"], self.params["side_size"]])
                
                # 크롭
                x = CenterCrop(self.params["crop_size"])(x)
                
                # 정규화
                x = Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])(x)
                
                # (B, T, C, H, W) 형태로 복원
                x = x.view(B, T, C, self.params["crop_size"], self.params["crop_size"])
                
                return x
        
        return SimpleTransform(params)
    
    def extract_features_from_segments(self, output_dir="./features"):
        """세그먼트별로 특징 추출"""
        if self.model is None:
            print("모델이 로드되지 않았습니다.")
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
            
            # 올바른 형태로 변환: (16, 3, H, W) -> (1, 16, 3, H, W)
            video_tensor = video_tensor.permute(0, 3, 1, 2).unsqueeze(0)  # (1, 16, 3, H, W)
            
            print(f"입력 텐서 형태: {video_tensor.shape}")
            
            # 전처리 적용
            if self.transform is not None:
                try:
                    video_data = {'video': video_tensor}
                    processed_video = self.transform(video_data)['video'].to(self.device)
                    print(f"Transform 후 형태: {processed_video.shape}")
                except Exception as e:
                    print(f"Transform 적용 실패: {e}")
                    # 간단한 전처리로 대체
                    processed_video = self.simple_preprocess(video_tensor).to(self.device)
                    print(f"Simple preprocess 후 형태: {processed_video.shape}")
            else:
                processed_video = self.simple_preprocess(video_tensor).to(self.device)
                print(f"Simple preprocess 후 형태: {processed_video.shape}")
            
            # X3D 모델에 맞는 형태로 변환: (B, T, C, H, W) -> (B, C, T, H, W)
            # 이 부분이 핵심! X3D 모델은 (B, C, T, H, W) 형태를 기대합니다.
            if processed_video.shape[2] == 3:  # 채널이 3인지 확인
                processed_video = processed_video.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W) -> (B, C, T, H, W)
                print(f"모델 입력용 변환 후: {processed_video.shape}")
            
            # 특징 추출
            with torch.no_grad():
                features = self.model(processed_video)
                return features.cpu().numpy()
                
        except Exception as e:
            print(f"특징 추출 실패: {img_path}, 에러: {e}")
            return None
    
    def simple_preprocess(self, video_tensor):
        """간단한 전처리 (차원 문제 해결용)"""
        # video_tensor: (B, T, C, H, W) = (1, 16, 3, H, W)
        B, T, C, H, W = video_tensor.shape
        
        print(f"Simple preprocess 입력: {video_tensor.shape}")
        
        # 정규화
        video_tensor = video_tensor / 255.0
        
        # 크기 조정 (320x320)
        video_tensor = torch.nn.functional.interpolate(
            video_tensor.view(B * T, C, H, W), 
            size=(320, 320), 
            mode='bilinear', 
            align_corners=False
        )
        
        # 정규화
        mean = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1)
        std = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1)
        video_tensor = (video_tensor - mean) / std
        
        # (B, T, C, H, W) 형태로 복원
        video_tensor = video_tensor.view(B, T, C, 320, 320)
        
        print(f"Simple preprocess 출력: {video_tensor.shape}")
        
        return video_tensor

if __name__ == "__main__":
    # 테스트 실행
    extractor = SegmentFeatureExtractor("image_segments.json")
    features_info = extractor.extract_features_from_segments()
    print(f"총 특징 추출: {len(features_info)}개")

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
        # transform을 완전히 제거
        self.transform = None
        
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
            
            # 이미지 로드
            image = Image.open(img_path).convert('RGB')
            
            # 이미지를 비디오 형태로 변환 (단일 프레임을 여러 번 복제)
            frames = [np.array(image)] * 16  # 16프레임으로 확장
            video_tensor = torch.from_numpy(np.stack(frames)).float()
            
            # 올바른 형태로 변환: (16, 3, H, W) -> (1, 16, 3, H, W)
            video_tensor = video_tensor.permute(0, 3, 1, 2).unsqueeze(0)  # (1, 16, 3, H, W)
            
            print(f"입력 텐서 형태: {video_tensor.shape}")
            
            # advanced_preprocess 사용 (더 나은 품질)
            processed_video = self.advanced_preprocess(video_tensor).to(self.device)
            print(f"Advanced preprocess 후 형태: {processed_video.shape}")
            
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
        
        # 개선된 크기 조정 (짧은 변 기준)
        if H < W:
            new_size = (320, int(320 * W / H))
        else:
            new_size = (int(320 * H / W), 320)
        
        video_tensor = torch.nn.functional.interpolate(
            video_tensor.view(B * T, C, H, W), 
            size=new_size, 
            mode='bilinear', 
            align_corners=False
        )
        
        # 중앙 크롭 (320x320)
        _, _, _, new_H, new_W = video_tensor.view(B, T, C, new_size[0], new_size[1]).shape
        
        if new_H > 320:
            start_h = (new_H - 320) // 2
            video_tensor = video_tensor[:, :, :, start_h:start_h+320, :]
        if new_W > 320:
            start_w = (new_W - 320) // 2
            video_tensor = video_tensor[:, :, :, :, start_w:start_w+320]
        
        # 정규화
        mean = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1)
        std = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1)
        video_tensor = (video_tensor - mean) / std
        
        # (B, T, C, H, W) 형태로 복원
        video_tensor = video_tensor.view(B, T, C, 320, 320)
        
        print(f"Simple preprocess 출력: {video_tensor.shape}")
        
        return video_tensor
    
    def advanced_preprocess(self, video_tensor):
        """고급 전처리 (PyTorchVideo 스타일)"""
        try:
            from torchvision.transforms import CenterCrop, Normalize
            from torchvision.transforms.functional import resize
            
            # video_tensor: (B, T, C, H, W) = (1, 16, 3, H, W)
            B, T, C, H, W = video_tensor.shape
            
            print(f"Advanced preprocess 입력: {video_tensor.shape}")
            
            # 1. 정규화 (0-255 -> 0-1)
            video_tensor = video_tensor / 255.0
            
            # 2. 짧은 변 기준 크기 조정 (PyTorchVideo ShortSideScale 스타일)
            if H < W:
                new_size = (320, int(320 * W / H))
            else:
                new_size = (int(320 * H / W), 320)
            
            video_tensor = torch.nn.functional.interpolate(
                video_tensor.view(B * T, C, H, W), 
                size=new_size, 
                mode='bilinear', 
                align_corners=False
            )
            
            # 3. 중앙 크롭 (PyTorchVideo CenterCrop 스타일)
            temp_tensor = video_tensor.view(B, T, C, new_size[0], new_size[1])
            _, _, _, new_H, new_W = temp_tensor.shape
            
            if new_H > 320:
                start_h = (new_H - 320) // 2
                temp_tensor = temp_tensor[:, :, :, start_h:start_h+320, :]
            if new_W > 320:
                start_w = (new_W - 320) // 2
                temp_tensor = temp_tensor[:, :, :, :, start_w:start_w+320]
            
            # 4. 정규화 (ImageNet 평균/표준편차)
            mean = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1)
            std = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1)
            temp_tensor = (temp_tensor - mean) / std
            
            # 5. (B, T, C, H, W) 형태로 복원
            video_tensor = temp_tensor.view(B, T, C, 320, 320)
            
            print(f"Advanced preprocess 출력: {video_tensor.shape}")
            
            return video_tensor
            
        except Exception as e:
            print(f"Advanced preprocess 실패: {e}")
            print("기본 simple_preprocess로 대체")
            return self.simple_preprocess(video_tensor)

if __name__ == "__main__":
    # 테스트 실행
    extractor = SegmentFeatureExtractor("image_segments.json")
    features_info = extractor.extract_features_from_segments()
    print(f"총 특징 추출: {len(features_info)}개")

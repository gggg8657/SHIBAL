import os
import sys
import json
import random
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
import torchvision.transforms as transforms
from PIL import Image
import tqdm

# 상위 디렉토리 import를 위한 경로 추가
sys.path.append('..')
from model import Model
from utils import Loss

class EndToEndDataset(torch.utils.data.Dataset):
    """End-to-End 학습을 위한 데이터셋 (이미지 → 라벨)"""
    
    def __init__(self, data_list_path, transform=None, test_mode=False):
        self.data_list = []
        self.labels = []
        self.categories = []
        self.transform = transform
        self.test_mode = test_mode
        
        # 데이터 리스트 로드
        with open(data_list_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 3:
                        image_path = parts[0]
                        label = int(parts[1])
                        category = parts[2] if len(parts) > 2 else "unknown"
                        
                        if os.path.exists(image_path):
                            self.data_list.append(image_path)
                            self.labels.append(label)
                            self.categories.append(category)
        
        print(f"[E2E] {len(self.data_list)}개 이미지 로드됨")
        
        # 정상/비정상 데이터 분리
        normal_indices = [i for i, label in enumerate(self.labels) if label == 0]
        abnormal_indices = [i for i, label in enumerate(self.labels) if label == 1]
        
        self.n_len = len(normal_indices)
        self.a_len = len(abnormal_indices)
        
        print(f"[E2E] 정상: {self.n_len}개, 비정상: {self.a_len}개")
        
        # 인덱스 초기화
        self.n_ind = list(normal_indices)
        self.a_ind = list(abnormal_indices)
        random.shuffle(self.n_ind)
        random.shuffle(self.a_ind)
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        # 정상/비정상 데이터 균형 맞추기
        if idx % 2 == 0:  # 짝수 인덱스: 정상 데이터
            if not self.n_ind:
                self.n_ind = list(range(self.n_len))
                random.shuffle(self.n_ind)
            data_idx = self.n_ind.pop()
        else:  # 홀수 인덱스: 비정상 데이터
            if not self.a_ind:
                self.a_ind = list(range(self.n_len, len(self.data_list)))
                random.shuffle(self.a_ind)
            data_idx = self.a_ind.pop()
        
        # 이미지 로드 및 전처리
        image_path = self.data_list[data_idx]
        label = self.labels[data_idx]
        category = self.categories[data_idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            return image, label, category
            
        except Exception as e:
            print(f"이미지 로드 실패: {image_path}, 에러: {e}")
            # 에러 발생 시 더미 데이터 반환
            dummy_image = torch.zeros(3, 224, 224)
            return dummy_image, label, category

class EndToEndModel(nn.Module):
    """End-to-End 모델: X3D + STEAD"""
    
    def __init__(self, model_arch='tiny', dropout=0.4, attn_dropout=0.1, 
                 ff_mult=1, dims=[32, 32], depths=[1, 1]):
        super().__init__()
        
        # X3D Feature Extractor (경량화)
        try:
            import torchvision.models as models
            # ResNet18 기반으로 경량화 (X3D 대신)
            self.feature_extractor = models.resnet18(pretrained=True)
            # 마지막 FC 레이어 제거
            self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
            self.feature_dim = 512
        except:
            # torchvision이 없는 경우 간단한 CNN
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            self.feature_dim = 256
        
        # STEAD 모델
        if model_arch == 'base':
            self.stead_model = Model(dropout=dropout, attn_dropout=attn_dropout)
        else:
            self.stead_model = Model(dropout=dropout, attn_dropout=attn_dropout,
                                   ff_mult=ff_mult, dims=tuple(dims), depths=tuple(depths))
        
        # Feature 차원 맞추기
        self.feature_adapter = nn.Linear(self.feature_dim, 400)  # STEAD 입력 차원
    
    def forward(self, x):
        # 이미지 → 특징 추출
        batch_size = x.size(0)
        
        # ResNet의 경우
        if hasattr(self.feature_extractor, 'avgpool'):
            features = self.feature_extractor(x)  # [B, 512, 1, 1]
            features = features.view(batch_size, -1)  # [B, 512]
        else:
            features = self.feature_extractor(x)  # [B, 256]
        
        # 차원 맞추기
        features = self.feature_adapter(features)  # [B, 400]
        
        # STEAD 모델 입력 형태로 변환
        features = features.unsqueeze(1)  # [B, 1, 400]
        
        # STEAD 모델 통과
        output = self.stead_model(features)
        
        return output

def load_config(config_path):
    """설정 파일 로드"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"❌ 설정 파일 로드 실패: {e}")
        return None

def print_config(config):
    """설정 출력"""
    print("\n=== 설정 정보 ===")
    print(f"데이터:")
    print(f"  - 훈련: {config['data']['train_list']}")
    print(f"  - 테스트: {config['data']['test_list']}")
    print(f"  - 모델: {config['data']['model_path']}")
    
    print(f"훈련:")
    print(f"  - 모델: {config['training']['model_arch']}")
    print(f"  - 배치 크기: {config['training']['batch_size']}")
    print(f"  - 학습률: {config['training']['lr']}")
    print(f"  - 최대 에포크: {config['training']['max_epoch']}")
    print(f"  - 드롭아웃: {config['training']['dropout_rate']}")
    
    print(f"GPU:")
    print(f"  - 멀티 GPU: {config['gpu']['use_multi_gpu']}")
    print(f"  - GPU ID: {config['gpu']['gpu_ids']}")

def setup_gpu(config, gpu_ids_arg=None):
    """GPU 설정"""
    if not torch.cuda.is_available():
        print("❌ CUDA를 사용할 수 없습니다")
        return 'cpu', False, 1, []
    
    gpu_config = config.get('gpu', {})
    use_multi_gpu = gpu_config.get('use_multi_gpu', False)
    auto_detect = gpu_config.get('auto_detect', True)
    
    if gpu_ids_arg:
        gpu_ids = gpu_ids_arg
    elif auto_detect:
        gpu_ids = list(range(torch.cuda.device_count()))
    else:
        gpu_ids = gpu_config.get('gpu_ids', [0])
    
    if not gpu_ids:
        gpu_ids = [0]
    
    gpu_count = len(gpu_ids)
    device = torch.device(f'cuda:{gpu_ids[0]}')
    
    print(f"[GPU] {gpu_count}개 GPU 사용: {gpu_ids}")
    for i, gpu_id in enumerate(gpu_ids):
        gpu_name = torch.cuda.get_device_name(gpu_id)
        gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
        print(f"  GPU {gpu_id}: {gpu_name} ({gpu_memory:.1f}GB)")
    
    return device, use_multi_gpu, gpu_count, gpu_ids

def parse_args():
    """커맨드라인 아규먼트 파싱"""
    parser = argparse.ArgumentParser(description='STEAD End-to-End 훈련')
    parser.add_argument('--config', type=str, required=True, help='설정 파일 경로')
    parser.add_argument('--batch_size', type=int, help='배치 크기 오버라이드')
    parser.add_argument('--lr', type=float, help='학습률 오버라이드')
    parser.add_argument('--max_epoch', type=int, help='최대 에포크 오버라이드')
    parser.add_argument('--gpu_ids', nargs='+', type=int, help='사용할 GPU ID 목록')
    
    return parser.parse_args()

def train(loader, model, optimizer, scheduler, device, epoch, use_multi_gpu):
    """훈련 함수"""
    model.train()
    total_loss = 0
    
    for step, (input_data, labels, categories) in enumerate(tqdm.tqdm(loader, desc=f"Epoch {epoch}")):
        input_data = input_data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(input_data)
        
        # Loss 계산
        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(0)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        
        # BCE Loss
        bce_loss = nn.BCEWithLogitsLoss()(outputs.squeeze(), labels.float())
        
        # Triplet Loss (간단한 버전)
        if outputs.size(0) > 1:
            # 정상/비정상 샘플 분리
            normal_mask = labels == 0
            abnormal_mask = labels == 1
            
            if normal_mask.sum() > 0 and abnormal_mask.sum() > 0:
                normal_features = outputs[normal_mask]
                abnormal_features = outputs[abnormal_mask]
                
                # 간단한 triplet loss
                anchor = normal_features[0:1]
                positive = normal_features[1:2] if normal_features.size(0) > 1 else normal_features[0:1]
                negative = abnormal_features[0:1]
                
                triplet_loss = nn.TripletMarginLoss()(anchor, positive, negative)
                total_loss_step = bce_loss + 0.1 * triplet_loss
            else:
                total_loss_step = bce_loss
        else:
            total_loss_step = bce_loss
        
        # Backward pass
        total_loss_step.backward()
        optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        total_loss += total_loss_step.item()
    
    return total_loss / len(loader)

def test(loader, model, device):
    """테스트 함수"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for input_data, labels, categories in tqdm.tqdm(loader, desc="테스트"):
            input_data = input_data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(input_data)
            pred = torch.sigmoid(outputs.squeeze())
            
            all_preds.extend(pred.cpu().detach().tolist())
            all_labels.extend(labels.cpu().detach().tolist())
    
    # 메트릭 계산
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    try:
        roc_auc = roc_auc_score(all_labels, all_preds)
        pr_auc = average_precision_score(all_labels, all_preds)
        
        print(f'ROC AUC: {roc_auc:.4f}')
        print(f'PR AUC: {pr_auc:.4f}')
        
        return roc_auc, pr_auc
    except Exception as e:
        print(f"메트릭 계산 실패: {e}")
        return 0.0, 0.0

def main():
    # 커맨드라인 아규먼트 파싱
    args = parse_args()
    
    # 설정 파일 로드
    config = load_config(args.config)
    if config is None:
        return
    
    # 커맨드라인 아규먼트로 설정 오버라이드
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
        print(f"[CFG] 배치 크기 오버라이드: {args.batch_size}")
    
    if args.lr is not None:
        config['training']['lr'] = args.lr
        print(f"[CFG] 학습률 오버라이드: {args.lr}")
    
    if args.max_epoch is not None:
        config['training']['max_epoch'] = args.max_epoch
        print(f"[CFG] 최대 에포크 오버라이드: {args.max_epoch}")
    
    print_config(config)
    
    # GPU 설정
    device, use_multi_gpu, gpu_count, gpu_ids = setup_gpu(config, args.gpu_ids)
    
    # 데이터 리스트 파일 존재 확인
    train_list = config['data']['train_list']
    test_list = config['data']['test_list']
    
    if not os.path.exists(train_list):
        print(f"❌ 훈련 데이터 리스트를 찾을 수 없습니다: {train_list}")
        return
    
    if not os.path.exists(test_list):
        print(f"❌ 테스트 데이터 리스트를 찾을 수 없습니다: {test_list}")
        return
    
    # 시드 설정
    random.seed(2025)
    np.random.seed(2025)
    torch.cuda.manual_seed(2025)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # 이미지 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 데이터 로더 생성
    try:
        train_loader = DataLoader(
            EndToEndDataset(train_list, transform=transform, test_mode=False),
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=0,  # Windows 호환성
            pin_memory=False
        )
        
        test_loader = DataLoader(
            EndToEndDataset(test_list, transform=transform, test_mode=True),
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=0,  # Windows 호환성
            pin_memory=False
        )
        
        print(f"훈련 데이터: {len(train_loader.dataset)}개")
        print(f"테스트 데이터: {len(test_loader.dataset)}개")
        
    except Exception as e:
        print(f"❌ 데이터 로더 생성 실패: {e}")
        return
    
    # End-to-End 모델 생성
    model_config = config['model']
    model = EndToEndModel(
        model_arch=config['training']['model_arch'],
        dropout=config['training']['dropout_rate'],
        attn_dropout=config['training']['attn_dropout_rate'],
        ff_mult=model_config['ff_mult'],
        dims=model_config['dims'],
        depths=model_config['depths']
    )
    
    # 프리트레인드 모델 로드 (STEAD 부분만)
    model_path = config['data']['model_path']
    if model_path and model_path != "null" and os.path.exists(model_path):
        try:
            model_ckpt = torch.load(model_path, map_location=device)
            # STEAD 모델 부분만 로드
            stead_state_dict = {}
            for key, value in model_ckpt.items():
                if key.startswith('stead_model.'):
                    stead_state_dict[key] = value
                elif not key.startswith('feature_extractor.') and not key.startswith('feature_adapter.'):
                    stead_state_dict[f'stead_model.{key}'] = value
            
            if stead_state_dict:
                model.stead_model.load_state_dict(stead_state_dict, strict=False)
                print(f"[OK] STEAD 모델 부분 로드: {model_path}")
        except Exception as e:
            print(f"❌ STEAD 모델 로드 실패: {e}")
            print("새로운 모델로 시작합니다.")
    else:
        print(f"[INFO] From Scratch 학습 모드")
    
    # 멀티 GPU 설정
    if use_multi_gpu:
        model = DataParallel(model, device_ids=gpu_ids)
        print(f"[GPU] DataParallel 활성화 (GPU {gpu_ids} 병렬)")
    
    model = model.to(device)
    
    # 체크포인트 저장 디렉토리 생성
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    savepath = os.path.join(config['training']['save_dir'], 
                           f"e2e_{config_name}_{config['training']['lr']}_{config['training']['batch_size']}_{config['training']['comment']}")
    os.makedirs(savepath, exist_ok=True)
    
    # 옵티마이저 및 스케줄러 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=0.2)
    
    num_steps = len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['max_epoch'])
    
    # 훈련 루프
    test_info = {"epoch": [], "test_AUC": [], "test_PR": []}
    
    print("\n=== End-to-End 훈련 시작 ===")
    for epoch in tqdm.tqdm(range(config['training']['max_epoch']), desc="전체 진행률"):
        # 훈련
        train_loss = train(train_loader, model, optimizer, scheduler, device, epoch, use_multi_gpu)
        
        # 테스트 (매 10 에포크마다)
        if (epoch + 1) % 10 == 0:
            roc_auc, pr_auc = test(test_loader, model, device)
            test_info["epoch"].append(epoch + 1)
            test_info["test_AUC"].append(roc_auc)
            test_info["test_PR"].append(pr_auc)
        
        # 모델 저장
        if (epoch + 1) % 20 == 0:
            if use_multi_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            torch.save(state_dict, os.path.join(savepath, f'model_epoch_{epoch+1}.pth'))
            print(f"[SAVE] Epoch {epoch+1} 모델 저장됨")
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{config['training']['max_epoch']}, Loss: {train_loss:.4f}")
    
    # 최종 모델 저장
    if use_multi_gpu:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    
    torch.save(state_dict, os.path.join(savepath, 'model_final.pth'))
    print(f"\n[COMPLETE] End-to-End 훈련 완료!")
    print(f"모델 저장됨: {savepath}")

if __name__ == "__main__":
    main()

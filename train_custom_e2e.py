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

# Loss 클래스 정의 (utils.py에 없음)
class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()

    def distance(self, x, y):
        d = torch.cdist(x, y, p=2)
        return d

    def forward(self, feats, margin=100.0):
        bs = len(feats)
        n_feats = feats[:bs // 2]
        a_feats = feats[bs // 2:]
        n_d = self.distance(n_feats, n_feats)
        a_d = self.distance(n_feats, a_feats)
        n_d_max, _ = torch.max(n_d, dim=0)
        a_d_min, _ = torch.min(a_d, dim=0)
        a_d_min = margin - a_d_min
        a_d_min = torch.max(torch.zeros(bs // 2).to(feats.device), a_d_min)
        return torch.mean(n_d_max) + torch.mean(a_d_min)

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.triplet = TripletLoss()

    def forward(self, scores, feats, targets, alpha=0.01):
        loss_ce = self.criterion(scores, targets)
        loss_triplet = self.triplet(feats)
        return loss_ce, alpha * loss_triplet

class EndToEndDataset(torch.utils.data.Dataset):
    """End-to-End 학습을 위한 데이터셋 (이미지 → 라벨)"""
    
    def __init__(self, data_list_path, transform=None, test_mode=False):
        self.data_list = []
        self.labels = []
        self.categories = []
        self.transform = transform
        self.test_mode = test_mode
        
        # 데이터 리스트 로드
        with open(data_list_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # |로 구분된 형식 처리
                    if '|' in line:
                        parts = line.split('|')
                        if len(parts) >= 3:
                            image_path = parts[0]
                            label_str = parts[1].lower()
                            category = parts[2] if len(parts) > 2 else "unknown"
                            
                            # 라벨 변환: normal -> 0, abnormal -> 1
                            if label_str in ['normal', '0']:
                                label = 0
                            elif label_str in ['abnormal', '1']:
                                label = 1
                            else:
                                print(f"⚠️ 알 수 없는 라벨: {label_str}, 건너뜀")
                                continue
                            
                            # Windows 경로 처리
                            image_path = image_path.replace('\\', '/')
                            
                            # 이미지 파일 존재 확인 (Windows 경로 지원)
                            if os.path.exists(image_path):
                                self.data_list.append(image_path)
                                self.labels.append(label)
                                self.categories.append(category)
                            else:
                                print(f"⚠️ 이미지 파일을 찾을 수 없음: {image_path}")
                                continue
                    else:
                        # 공백으로 구분된 형식 처리 (기존 방식)
                        parts = line.split()
                        if len(parts) >= 3:
                            image_path = parts[0]
                            try:
                                label = int(parts[1])
                                category = parts[2] if len(parts) > 2 else "unknown"
                                
                                # Windows 경로 처리
                                image_path = image_path.replace('\\', '/')
                                
                                if os.path.exists(image_path):
                                    self.data_list.append(image_path)
                                    self.labels.append(label)
                                    self.categories.append(category)
                                else:
                                    print(f"⚠️ 이미지 파일을 찾을 수 없음: {image_path}")
                                    continue
                            except ValueError:
                                print(f"⚠️ 라벨을 정수로 변환할 수 없음: {parts[1]}, 건너뜀")
                                continue
        
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
            # 간단한 CNN으로 대체하여 차원 문제 해결
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(256, 512, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
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
        
        # Feature 차원 맞추기 (CNN 출력 -> STEAD 입력)
        self.feature_adapter = nn.Linear(self.feature_dim, 400)
        
        # STEAD 모델의 init_dim에 맞추기 위한 어댑터
        self.dim_adapter = nn.Linear(400, 32)  # 400 -> 32 (tiny 모델)
    
    def forward(self, x):
        # 배치 크기가 1인 경우 복제하여 2로 만들기 (BatchNorm 문제 해결)
        if x.size(0) == 1:
            x = x.repeat(2, 1, 1, 1)
            need_squeeze = True
        else:
            need_squeeze = False
        
        # CNN 특징 추출
        features = self.feature_extractor(x)  # [B, 512] 또는 [B, 256]
        
        # 차원 맞추기 (CNN 출력 -> STEAD 입력)
        features = self.feature_adapter(features)  # [B, 400]
        
        # STEAD 모델의 init_dim에 맞추기
        features = self.dim_adapter(features)  # [B, 32]
        
        # STEAD 모델 입력 형태로 변환: [B, 32] -> [B, 32, 1, 1, 1] (B, C, T, H, W)
        features = features.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # [B, 32, 1, 1, 1]
        
        # STEAD 모델 통과
        output = self.stead_model(features)
        
        # 배치 크기가 1이었던 경우 첫 번째 결과만 반환
        if need_squeeze:
            if isinstance(output, tuple):
                output = (output[0][0:1], output[1][0:1])
            else:
                output = output[0:1]
        
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
    print(f"  - 검증: {config['data'].get('valid_list', '없음')}")
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
    parser.add_argument('--config', type=str, default='config_e2e_tiny.json', help='설정 파일 경로')
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
        
        # STEAD 모델은 (logits, features) 튜플을 반환
        if isinstance(outputs, tuple):
            logits, features = outputs
        else:
            logits = outputs
            features = None
        
        # Loss 계산
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        
        logits = logits.view(-1)  # 모든 차원을 1차원으로 평탄화
        labels = labels.view(-1)  # 모든 차원을 1차원으로 평탄화
        
        bce_loss = nn.BCEWithLogitsLoss()(logits, labels.float())
        
        # Triplet Loss (간단한 버전)
        if logits.size(0) > 1:
            # 정상/비정상 샘플 분리
            normal_mask = labels == 0
            abnormal_mask = labels == 1
            
            if normal_mask.sum() > 0 and abnormal_mask.sum() > 0:
                normal_features = logits[normal_mask]
                abnormal_features = logits[abnormal_mask]
                
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
            
            # STEAD 모델은 (logits, features) 튜플을 반환
            if isinstance(outputs, tuple):
                logits, features = outputs
            else:
                logits = outputs
            
            pred = torch.sigmoid(logits.squeeze())
            
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
        # 훈련 데이터
        train_loader = DataLoader(
            EndToEndDataset(train_list, transform=transform, test_mode=False),
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=0,  # Windows 호환성
            pin_memory=False,
            drop_last=True  # BatchNorm 문제 방지
        )
        
        # 검증 데이터 (validation set)
        valid_list = config['data'].get('valid_list', train_list)  # 기본값은 train_list
        valid_loader = DataLoader(
            EndToEndDataset(valid_list, transform=transform, test_mode=True),
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=0,  # Windows 호환성
            pin_memory=False,
            drop_last=True  # BatchNorm 문제 방지
        )
        
        # 테스트 데이터
        test_loader = DataLoader(
            EndToEndDataset(test_list, transform=transform, test_mode=True),
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=0,  # Windows 호환성
            pin_memory=False,
            drop_last=True  # BatchNorm 문제 방지
        )
        
        print(f"훈련 데이터: {len(train_loader.dataset)}개")
        print(f"검증 데이터: {len(valid_loader.dataset)}개")
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
    valid_info = {"epoch": [], "valid_AUC": [], "valid_PR": []}
    
    # Early Stopping 설정
    best_valid_auc = 0.0
    patience = 10
    patience_counter = 0
    
    print("\n=== End-to-End 훈련 시작 ===")
    for epoch in tqdm.tqdm(range(config['training']['max_epoch']), desc="전체 진행률"):
        # 훈련
        train_loss = train(train_loader, model, optimizer, scheduler, device, epoch, use_multi_gpu)
        
        # 검증 (매 에포크마다)
        valid_roc_auc, valid_pr_auc = test(valid_loader, model, device)
        valid_info["epoch"].append(epoch + 1)
        valid_info["valid_AUC"].append(valid_roc_auc)
        valid_info["valid_PR"].append(valid_pr_auc)
        print(f"Epoch {epoch+1} - 검증: ROC AUC: {valid_roc_auc:.4f}, PR AUC: {valid_pr_auc:.4f}")
        
        # Best Model 저장 (검증 성능 기준)
        if valid_roc_auc > best_valid_auc:
            best_valid_auc = valid_roc_auc
            patience_counter = 0
            
            # Best Model 저장
            if use_multi_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            torch.save(state_dict, os.path.join(savepath, 'model_best.pth'))
            print(f"[BEST] 새로운 최고 성능 모델 저장: ROC AUC {valid_roc_auc:.4f}")
        else:
            patience_counter += 1
        
        # Early Stopping 체크
        if patience_counter >= patience:
            print(f"[STOP] {patience} 에포크 동안 성능 개선 없음. 조기 종료.")
            break
        
        # 테스트 (매 10 에포크마다)
        if (epoch + 1) % 10 == 0:
            test_roc_auc, test_pr_auc = test(test_loader, model, device)
            test_info["epoch"].append(epoch + 1)
            test_info["test_AUC"].append(test_roc_auc)
            test_info["test_PR"].append(test_roc_auc)
            print(f"Epoch {epoch+1} - 테스트: ROC AUC: {test_roc_auc:.4f}, PR AUC: {test_pr_auc:.4f}")
        
        # 주기적 모델 저장
        if (epoch + 1) % 20 == 0:
            if use_multi_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            torch.save(state_dict, os.path.join(savepath, f'model_epoch_{epoch+1}.pth'))
            print(f"[SAVE] Epoch {epoch+1} 모델 저장됨")
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{config['training']['max_epoch']}, Loss: {train_loss:.4f}, Best Valid AUC: {best_valid_auc:.4f}, Patience: {patience_counter}/{patience}")
    
    # 최종 모델 저장
    if use_multi_gpu:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    
    torch.save(state_dict, os.path.join(savepath, 'model_final.pth'))
    
    # 훈련 결과 요약
    print(f"\n=== 훈련 결과 요약 ===")
    print(f"최고 검증 ROC AUC: {best_valid_auc:.4f}")
    print(f"최종 테스트 ROC AUC: {test_info['test_AUC'][-1] if test_info['test_AUC'] else 'N/A'}")
    print(f"총 훈련 에포크: {len(valid_info['valid_AUC'])}")
    
    print(f"\n[COMPLETE] End-to-End 훈련 완료!")
    print(f"모델 저장됨: {savepath}")
    print(f"  - Best Model: model_best.pth (검증 성능 기준)")
    print(f"  - Final Model: model_final.pth (마지막 에포크)")
    print(f"  - 주기적 저장: model_epoch_X.pth (20 에포크마다)")

if __name__ == "__main__":
    main()

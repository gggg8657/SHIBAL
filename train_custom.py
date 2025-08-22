#!/usr/bin/env python3
"""
커스텀 데이터셋 STEAD 훈련 스크립트
세그먼트 정보를 활용한 이상 탐지 모델 훈련
config.json 파일에서 설정을 읽어옵니다.
멀티 GPU 지원 (GPU 0, 1번 병렬 처리)
여러 설정 파일을 사용하여 동시 학습 가능
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DataParallel
from tqdm import tqdm
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import os
import datetime
import random
import numpy as np
import json
import argparse
from torch.utils.data import DataLoader
from segment_dataset import SegmentDataset
from model import Model
from utils import save_best_record
from timm.scheduler.cosine_lr import CosineLRScheduler
import sys
# Windows 콘솔(cp949 등)에서 이모지 출력 시 UnicodeEncodeError 방지
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

def parse_args():
    """커맨드라인 아규먼트 파싱"""
    parser = argparse.ArgumentParser(description='커스텀 STEAD 훈련')
    parser.add_argument('--config', default='config.json', help='설정 파일 경로 (기본값: config.json)')
    parser.add_argument('--gpu_ids', type=str, default=None, help='사용할 GPU ID (예: 0,1 또는 0)')
    parser.add_argument('--batch_size', type=int, default=None, help='배치 크기 (config.json 오버라이드)')
    parser.add_argument('--lr', type=float, default=None, help='학습률 (config.json 오버라이드)')
    parser.add_argument('--max_epoch', type=int, default=None, help='최대 에포크 (config.json 오버라이드)')
    return parser.parse_args()

def load_config(config_path='config.json'):
    """config.json 파일을 로드합니다."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"[OK] 설정 파일 로드: {config_path}")
        return config
    except FileNotFoundError:
        print(f"❌ 설정 파일을 찾을 수 없습니다: {config_path}")
        print("config.json 파일을 생성하거나 경로를 확인하세요.")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ JSON 파싱 오류: {e}")
        return None

def print_config(config):
    """설정을 출력합니다."""
    print("\n=== 훈련 설정 ===")
    print(f"훈련 데이터: {config['data']['train_list']}")
    print(f"테스트 데이터: {config['data']['test_list']}")
    print(f"모델 경로: {config['data']['model_path']}")
    print(f"모델 아키텍처: {config['training']['model_arch']}")
    print(f"배치 크기: {config['training']['batch_size']}")
    print(f"학습률: {config['training']['lr']}")
    print(f"최대 에포크: {config['training']['max_epoch']}")
    print(f"저장 디렉토리: {config['training']['save_dir']}")
    print("=" * 30)

def setup_gpu(config, gpu_ids_arg=None):
    """GPU 설정을 확인하고 설정합니다."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"[GPU] 사용 가능한 GPU: {gpu_count}개")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        
        # 커맨드라인 아규먼트 우선 적용
        if gpu_ids_arg:
            gpu_ids = [int(x.strip()) for x in gpu_ids_arg.split(',')]
            use_multi_gpu = len(gpu_ids) > 1
            print(f"[GPU] 커맨드라인 GPU 설정: {gpu_ids}")
        else:
            # config.json에서 GPU 설정 읽기
            gpu_config = config.get('gpu', {})
            use_multi_gpu = gpu_config.get('use_multi_gpu', True)
            gpu_ids = gpu_config.get('gpu_ids', [0, 1])
            auto_detect = gpu_config.get('auto_detect', True)
            
            if auto_detect and gpu_count >= 2 and use_multi_gpu:
                gpu_ids = [0, 1]  # 최대 2개 GPU 사용
            elif not use_multi_gpu:
                gpu_ids = [0]
        
        if len(gpu_ids) > 1:
            print(f"[GPU] 멀티 GPU 모드 활성화 (GPU {gpu_ids} 병렬)")
            device = torch.device(f'cuda:{gpu_ids[0]}')
            return device, True, len(gpu_ids), gpu_ids
        else:
            print("[GPU] 단일 GPU 모드 (GPU 0번만 사용)")
            device = torch.device('cuda:0')
            return device, False, 1, [0]
    else:
        print("⚠️ GPU를 사용할 수 없습니다. CPU 모드로 실행합니다.")
        return torch.device('cpu'), False, 0, []

class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()

    def distance(self, x, y):
        d = torch.cdist(x, y, p=2)
        return d

    def forward(self, feats, margin = 100.0):
        bs = len(feats)
        n_feats = feats[:bs // 2]
        a_feats = feats[bs // 2:]
        n_d = self.distance(n_feats, n_feats)
        a_d = self.distance(n_feats, a_feats)
        n_d_max, _ = torch.max(n_d, dim=0)
        a_d_min, _ = torch.min(a_d, dim=0)
        a_d_min = margin - a_d_min
        a_d_min = torch.max(torch.zeros(bs // 2).cuda(), a_d_min)
        return torch.mean(n_d_max) + torch.mean(a_d_min)

class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.triplet = TripletLoss()

    def forward(self, scores, feats, targets, alpha = 0.01):
        loss_ce = self.criterion(scores, targets)
        loss_triplet = self.triplet(feats)
        return loss_ce, alpha * loss_triplet

def train(loader, model, optimizer, scheduler, device, epoch, use_multi_gpu=False):
    """훈련 함수"""
    with torch.set_grad_enabled(True):
        model.train()
        pred = []
        label = []
        for step, (ninput, nlabel, ainput, alabel, ncategory, acategory) in tqdm(enumerate(loader), desc=f"Epoch {epoch}"):
            input = torch.cat((ninput, ainput), 0).to(device, non_blocking=True)
            
            scores, feats = model(input) 
            pred += scores.cpu().detach().tolist()
            labels = torch.cat((nlabel, alabel), 0).to(device, non_blocking=True)
            label += labels.cpu().detach().tolist()

            loss_ce, loss_con = Loss()(scores.squeeze(), feats, labels)
            loss = loss_ce + loss_con

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            scheduler.step_update(epoch * len(loader) + step)
        
        fpr, tpr, _ = roc_curve(label, pred)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(label, pred)
        pr_auc = auc(recall, precision)
        print(f'train_pr_auc : {pr_auc:.4f}')
        print(f'train_roc_auc : {roc_auc:.4f}')
        return loss.item()

def test(dataloader, model, config, device, use_multi_gpu=False):
    """테스트 함수"""
    model.to(device)
    with torch.no_grad():
        model.eval()
        pred = []
        labels = []
        for _, inputs in tqdm(enumerate(dataloader), desc="테스트"):
            if len(inputs) == 4:  # features, label, category, description
                features, label, category, description = inputs
            else:
                features, label = inputs
            
            # 라벨을 1D 리스트로 강제 변환
            labels_1d = label.view(-1).detach().cpu().numpy().tolist()
            labels += labels_1d
            input = features.to(device, non_blocking=True)
            
            scores, _ = model(input)
            # 예측을 1D 리스트로 강제 변환 (logits 그대로 사용)
            scores_1d = scores.view(-1).detach().cpu().numpy().tolist()
            pred += scores_1d
        
        fpr, tpr, _ = roc_curve(labels, pred)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(labels, pred)
        pr_auc = auc(recall, precision)
        print(f'pr_auc : {pr_auc:.4f}')
        print(f'roc_auc : {roc_auc:.4f}')
        return roc_auc, pr_auc

def init_weights(m):
    """가중치 초기화"""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv1d):
         torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Conv2d):
         torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')     
    elif isinstance(m, nn.Conv3d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

def save_config(save_path, args):
    """설정 저장"""
    path = save_path+'/'
    os.makedirs(path, exist_ok=True)
    f = open(path + "config_{}.txt".format(datetime.datetime.now()), 'w')
    for key in vars(args).keys():
        f.write('{}: {}'.format(key, vars(args)[key]))
        f.write('\n')
    f.close()

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
        print("먼저 preprocess_segments.py를 실행하여 데이터를 준비하세요.")
        return
    
    if not os.path.exists(test_list):
        print(f"❌ 테스트 데이터 리스트를 찾을 수 없습니다: {test_list}")
        print("먼저 preprocess_segments.py를 실행하여 데이터를 준비하세요.")
        return
    
    # 시드 설정
    random.seed(2025)
    np.random.seed(2025)
    torch.cuda.manual_seed(2025)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # 데이터 로더 생성
    try:
        # SegmentDataset에 필요한 args 객체 생성
        class Args:
            def __init__(self, config):
                self.rgb_list = config['data']['train_list']
                self.test_rgb_list = config['data']['test_list']
                self.batch_size = config['training']['batch_size']
        
        args_obj = Args(config)
        
        # 멀티 GPU 사용 시 배치 크기 조정
        if use_multi_gpu:
            effective_batch_size = config['training']['batch_size'] // gpu_count
            print(f"[GPU] {gpu_count}개 사용으로 배치 크기 조정: {config['training']['batch_size']} -> {effective_batch_size} (GPU당)")
        else:
            effective_batch_size = config['training']['batch_size']
        
        # Windows 호환성을 위한 DataLoader 설정
        is_windows = os.name == 'nt'
        
        train_loader = DataLoader(
            SegmentDataset(args_obj, test_mode=False),
            batch_size=effective_batch_size // 2,
            num_workers=0 if is_windows else min(2, os.cpu_count() or 2),  # Windows에서는 0
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=2 if not is_windows else None,  # Windows에서는 None
            shuffle=True,
        )
        test_loader = DataLoader(
            SegmentDataset(args_obj, test_mode=True),
            batch_size=effective_batch_size,
            num_workers=0 if is_windows else min(2, os.cpu_count() or 2),  # Windows에서는 0
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=2 if not is_windows else None,  # Windows에서는 None
            shuffle=False,
        )
        
        print(f"훈련 데이터: {len(train_loader.dataset)}개")
        print(f"테스트 데이터: {len(test_loader.dataset)}개")
        
    except Exception as e:
        print(f"❌ 데이터 로더 생성 실패: {e}")
        return
    
    # 모델 생성
    model_config = config['model']
    if config['training']['model_arch'] == 'base':
        model = Model(dropout=config['training']['dropout_rate'], 
                     attn_dropout=config['training']['attn_dropout_rate'])
    elif config['training']['model_arch'] == 'fast' or config['training']['model_arch'] == 'tiny':
        model = Model(dropout=config['training']['dropout_rate'], 
                     attn_dropout=config['training']['attn_dropout_rate'], 
                     ff_mult=model_config['ff_mult'], 
                     dims=tuple(model_config['dims']), 
                     depths=tuple(model_config['depths']))
    else:
        print("❌ 모델 아키텍처를 인식할 수 없습니다")
        return
    
    model.apply(init_weights)
    
    # 프리트레인드 모델 로드
    model_path = config['data']['model_path']
    if model_path and model_path != "null" and os.path.exists(model_path):
        try:
            model_ckpt = torch.load(model_path, map_location=device)
            model.load_state_dict(model_ckpt, strict=False)
            print(f"[OK] 프리트레인드 모델 로드: {model_path}")
        except Exception as e:
            print(f"❌ 프리트레인드 모델 로드 실패: {e}")
            print("새로운 모델로 시작합니다.")
    else:
        print(f"[INFO] From Scratch 학습 모드")
        print("새로운 모델로 시작합니다.")
    
    # 멀티 GPU 설정
    if use_multi_gpu:
        model = DataParallel(model, device_ids=gpu_ids)
        print(f"[GPU] DataParallel 활성화 (GPU {gpu_ids} 병렬)")
    
    model = model.to(device)
    
    # 체크포인트 저장 디렉토리 생성 (설정 파일명 포함)
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    savepath = os.path.join(config['training']['save_dir'], 
                           f"{config_name}_{config['training']['lr']}_{config['training']['batch_size']}_{config['training']['comment']}")
    os.makedirs(savepath, exist_ok=True)
    
    # 옵티마이저 및 스케줄러 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=0.2)
    
    num_steps = len(train_loader)
    scheduler = CosineLRScheduler(
            optimizer,
            t_initial=config['training']['max_epoch'] * num_steps,
            cycle_mul=1.,
            lr_min=config['training']['lr'] * 0.2,
            warmup_lr_init=config['training']['lr'] * 0.01,
            warmup_t=config['training']['warmup'] * num_steps,
            cycle_limit=20,
            t_in_epochs=False,
            warmup_prefix=True,
            cycle_decay=0.95,
        )
    
    # 훈련 루프
    test_info = {"epoch": [], "test_AUC": [], "test_PR":[]}
    loss_criterion = Loss().to(device)
    
    print("\n=== 훈련 시작 ===")
    for step in tqdm(range(0, config['training']['max_epoch']), 
                     total=config['training']['max_epoch'], desc="전체 진행률"):
        cost = train(train_loader, model, optimizer, scheduler, device, step, use_multi_gpu)
        scheduler.step(step + 1)
        
        # 테스트
        auc, pr_auc = test(test_loader, model, config, device, use_multi_gpu)
        
        test_info["epoch"].append(step)
        test_info["test_AUC"].append(auc)
        test_info["test_PR"].append(pr_auc)
        
        # 모델 저장 (DataParallel 사용 시 module 접근)
        if use_multi_gpu:
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()
            
        torch.save(model_state_dict, 
                  f'{savepath}/{config["training"]["model_name"]}{step}-x3d.pkl')
        save_best_record(test_info, os.path.join(savepath, f'{step}-step.txt'))
        
        print(f"Epoch {step}: AUC={auc:.4f}, PR_AUC={pr_auc:.4f}")
    
    # 최종 모델 저장
    if use_multi_gpu:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
        
    torch.save(model_state_dict, 
              f'{savepath}/{config["training"]["model_name"]}final.pkl')
    print(f"\n[OK] 훈련 완료! 모델이 {savepath}에 저장되었습니다.")

if __name__ == "__main__":
    main()

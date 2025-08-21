#!/usr/bin/env python3
"""
커스텀 데이터셋 STEAD 훈련 스크립트
세그먼트 정보를 활용한 이상 탐지 모델 훈련
config.json 파일에서 설정을 읽어옵니다.
"""

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import os
import datetime
import random
import numpy as np
import json
from torch.utils.data import DataLoader
from segment_dataset import SegmentDataset
from model import Model
from utils import save_best_record
from timm.scheduler.cosine_lr import CosineLRScheduler

def load_config(config_path='config.json'):
    """config.json 파일을 로드합니다."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✅ 설정 파일 로드: {config_path}")
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

def train(loader, model, optimizer, scheduler, device, epoch):
    """훈련 함수"""
    with torch.set_grad_enabled(True):
        model.train()
        pred = []
        label = []
        for step, (ninput, nlabel, ainput, alabel, ncategory, acategory) in tqdm(enumerate(loader), desc=f"Epoch {epoch}"):
            input = torch.cat((ninput, ainput), 0).to(device)
            
            scores, feats = model(input) 
            pred += scores.cpu().detach().tolist()
            labels = torch.cat((nlabel, alabel), 0).to(device)
            label += labels.cpu().detach().tolist()

            loss_criterion = Loss()
            loss_ce, loss_con = loss_criterion(scores.squeeze(), feats, labels)
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

def test(dataloader, model, args, device = 'cuda'):
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
            
            labels += label.cpu().detach().tolist()
            input = features.to(device)
            scores, feat = model(input)
            scores = torch.nn.Sigmoid()(scores).squeeze()
            pred_ = scores.cpu().detach().tolist()
            pred += pred_
        
        fpr, tpr, threshold = roc_curve(labels, pred)
        roc_auc = auc(fpr, tpr)
        precision, recall, th = precision_recall_curve(labels, pred)
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
    # 설정 파일 로드
    config = load_config()
    if config is None:
        return
    
    print_config(config)
    
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 데이터 로더 생성
    try:
        # SegmentDataset에 필요한 args 객체 생성
        class Args:
            def __init__(self, config):
                self.rgb_list = config['data']['train_list']
                self.test_rgb_list = config['data']['test_list']
                self.batch_size = config['training']['batch_size']
        
        args = Args(config)
        
        train_loader = DataLoader(SegmentDataset(args, test_mode=False),
                                   batch_size=config['training']['batch_size'] // 2)
        test_loader = DataLoader(SegmentDataset(args, test_mode=True),
                                 batch_size=config['training']['batch_size'])
        
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
    if os.path.exists(model_path):
        try:
            model_ckpt = torch.load(model_path, map_location=device)
            model.load_state_dict(model_ckpt, strict=False)
            print(f"✅ 프리트레인드 모델 로드: {model_path}")
        except Exception as e:
            print(f"❌ 프리트레인드 모델 로드 실패: {e}")
            print("새로운 모델로 시작합니다.")
    else:
        print(f"⚠️ 프리트레인드 모델을 찾을 수 없습니다: {model_path}")
        print("새로운 모델로 시작합니다.")
    
    model = model.to(device)
    
    # 체크포인트 저장 디렉토리 생성
    savepath = os.path.join(config['training']['save_dir'], 
                           f"{config['training']['lr']}_{config['training']['batch_size']}_{config['training']['comment']}")
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
    
    print("\n=== 훈련 시작 ===")
    for step in tqdm(range(0, config['training']['max_epoch']), 
                     total=config['training']['max_epoch'], desc="전체 진행률"):
        cost = train(train_loader, model, optimizer, scheduler, device, step)
        scheduler.step(step + 1)
        
        # 테스트
        auc, pr_auc = test(test_loader, model, config, device)
        
        test_info["epoch"].append(step)
        test_info["test_AUC"].append(auc)
        test_info["test_PR"].append(pr_auc)
        
        # 모델 저장
        torch.save(model.state_dict(), 
                  f'{savepath}/{config["training"]["model_name"]}{step}-x3d.pkl')
        save_best_record(test_info, os.path.join(savepath, f'{step}-step.txt'))
        
        print(f"Epoch {step}: AUC={auc:.4f}, PR_AUC={pr_auc:.4f}")
    
    # 최종 모델 저장
    torch.save(model.state_dict(), 
              f'{savepath}/{config["training"]["model_name"]}final.pkl')
    print(f"\n✅ 훈련 완료! 모델이 {savepath}에 저장되었습니다.")

if __name__ == "__main__":
    main()

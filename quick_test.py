#!/usr/bin/env python3
"""
빠른 테스트 스크립트
saved_models의 프리트레인드 웨이트로 간단한 성능 확인
"""

import torch
import numpy as np
from model import Model
import option
from segment_dataset import SegmentDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

def quick_test():
    """빠른 테스트 수행"""
    args = option.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"디바이스: {device}")
    
    # 모델 로드
    if args.model_arch == 'base':
        model = Model(dropout=args.dropout_rate, attn_dropout=args.attn_dropout_rate)
    elif args.model_arch == 'fast' or args.model_arch == 'tiny':
        model = Model(dropout=args.dropout_rate, attn_dropout=args.attn_dropout_rate, 
                     ff_mult=1, dims=(32,32), depths=(1,1))
    
    # 저장된 모델 로드 (기본값: tiny 모델)
    model_path = 'saved_models/888tiny.pkl'
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        print("사용 가능한 모델:")
        for file in os.listdir('saved_models'):
            if file.endswith('.pkl'):
                print(f"  - saved_models/{file}")
        return
    
    # 모델 파일명에서 아키텍처 자동 감지
    if 'base' in model_path or '913' in model_path:
        detected_arch = 'base'
        model = Model(dropout=args.dropout_rate, attn_dropout=args.attn_dropout_rate)
    elif 'tiny' in model_path or '888' in model_path:
        detected_arch = 'tiny'
        model = Model(dropout=args.dropout_rate, attn_dropout=args.attn_dropout_rate, 
                     ff_mult=1, dims=(32,32), depths=(1,1))
    else:
        detected_arch = args.model_arch
    
    print(f"감지된 모델 아키텍처: {detected_arch}")
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ 모델 로드 완료: {model_path} (strict=False)")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return
    
    model = model.to(device)
    model.eval()
    
    # 테스트 데이터 로더
    try:
        test_dataset = SegmentDataset(args, test_mode=True)
        if len(test_dataset) == 0:
            print("❌ 테스트 데이터를 찾을 수 없습니다.")
            print("custom_data/custom_test.txt 파일을 먼저 생성하세요.")
            return
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size, 
            shuffle=False
        )
        
        print(f"테스트 데이터: {len(test_dataset)}개")
        
    except Exception as e:
        print(f"❌ 데이터 로더 생성 실패: {e}")
        return
    
    # 테스트 수행
    predictions = []
    labels = []
    categories = []
    
    print("테스트 진행 중...")
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 4:  # features, label, category, description
                features, label, category, description = batch
            else:
                features, label = batch
                category = ['unknown'] * len(label)
            
            features = features.to(device)
            scores, _ = model(features)
            scores = torch.nn.Sigmoid()(scores).squeeze()
            
            predictions.extend(scores.cpu().numpy())
            labels.extend(label.numpy())
            categories.extend(category)
    
    # 결과 계산
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # ROC AUC
    from sklearn.metrics import roc_auc_score, precision_recall_auc_score
    roc_auc = roc_auc_score(labels, predictions)
    
    # 정확도
    binary_preds = (predictions > 0.5).astype(int)
    accuracy = np.mean(binary_preds == labels)
    
    print(f"\n=== 빠른 테스트 결과 ===")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"정확도: {accuracy:.4f}")
    print(f"총 샘플: {len(labels)}")
    print(f"정상: {np.sum(labels == 0)}")
    print(f"비정상: {np.sum(labels == 1)}")
    
    # 카테고리별 성능
    unique_categories = set(categories)
    print(f"\n=== 카테고리별 성능 ===")
    for cat in unique_categories:
        cat_indices = [i for i, c in enumerate(categories) if c == cat]
        if len(cat_indices) > 0:
            cat_preds = predictions[cat_indices]
            cat_labels = labels[cat_indices]
            try:
                cat_auc = roc_auc_score(cat_labels, cat_preds)
                print(f"{cat}: AUC={cat_auc:.4f} (샘플={len(cat_indices)})")
            except:
                print(f"{cat}: AUC 계산 불가 (샘플={len(cat_indices)})")
    
    # 간단한 시각화
    try:
        plt.figure(figsize=(10, 4))
        
        # ROC 곡선
        plt.subplot(1, 2, 1)
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(labels, predictions)
        plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        
        # 예측 분포
        plt.subplot(1, 2, 2)
        plt.hist(predictions[labels == 0], alpha=0.7, label='Normal', bins=20)
        plt.hist(predictions[labels == 1], alpha=0.7, label='Anomaly', bins=20)
        plt.xlabel('Prediction Score')
        plt.ylabel('Count')
        plt.title('Prediction Distribution')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('quick_test_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\n📊 시각화 결과가 'quick_test_results.png'에 저장되었습니다.")
        
    except Exception as e:
        print(f"시각화 실패: {e}")

if __name__ == "__main__":
    quick_test()

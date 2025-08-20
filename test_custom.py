#!/usr/bin/env python3
"""
커스텀 데이터셋 테스트 스크립트
saved_models의 프리트레인드 웨이트 사용
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, precision_recall_curve, classification_report
from tqdm import tqdm
import argparse
import json
import os
import sys
from segment_dataset import SegmentDataset
from model import Model
import option

class CustomTester:
    def __init__(self, model_path, args, device='cuda'):
        self.device = device
        self.args = args
        self.model = self.load_model(model_path)
        
    def load_model(self, model_path):
        """훈련된 모델 로드"""
        # 모델 파일명에서 아키텍처 자동 감지
        if 'base' in model_path or '913' in model_path:
            detected_arch = 'base'
        elif 'tiny' in model_path or '888' in model_path:
            detected_arch = 'tiny'
        else:
            detected_arch = self.args.model_arch
        
        print(f"감지된 모델 아키텍처: {detected_arch}")
        
        if detected_arch == 'base':
            model = Model(dropout=self.args.dropout_rate, attn_dropout=self.args.attn_dropout_rate)
        elif detected_arch == 'fast' or detected_arch == 'tiny':
            model = Model(dropout=self.args.dropout_rate, attn_dropout=self.args.attn_dropout_rate, 
                        ff_mult=1, dims=(32,32), depths=(1,1))
        else:
            raise ValueError("Model architecture not recognized")
        
        print(f"모델 로드 중: {model_path}")
        
        # 모델 가중치 로드 시 오류 처리
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict, strict=False)
            print("✅ 모델 로드 완료 (strict=False)")
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            print("모델 아키텍처를 다시 확인하세요.")
            raise e
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def test_custom_data(self, test_loader, save_results=True, output_dir="./test_results"):
        """커스텀 데이터 테스트 수행"""
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'predictions': [],
            'labels': [],
            'categories': [],
            'descriptions': [],
            'features': [],
            'segment_performance': {}
        }
        
        print("=== 커스텀 데이터 테스트 시작 ===")
        
        with torch.no_grad():
            for batch_idx, (features, labels, categories, descriptions) in enumerate(tqdm(test_loader, desc="테스트 진행")):
                features = features.to(self.device)
                
                # 모델 예측
                scores, feat = self.model(features)
                scores = torch.nn.Sigmoid()(scores).squeeze()
                
                # 결과 수집
                results['predictions'].extend(scores.cpu().numpy())
                results['labels'].extend(labels.numpy())
                results['categories'].extend(categories)
                results['descriptions'].extend(descriptions)
                results['features'].extend(feat.cpu().numpy())
        
        # 전체 성능 계산
        overall_metrics = self.calculate_metrics(results['predictions'], results['labels'])
        
        # 세그먼트별 성능 분석
        segment_metrics = self.analyze_segment_performance(results)
        
        # 결과 출력
        self.print_results(overall_metrics, segment_metrics)
        
        # 시각화
        if save_results:
            self.visualize_results(results, overall_metrics, segment_metrics, output_dir)
            self.save_detailed_results(results, overall_metrics, segment_metrics, output_dir)
        
        return results, overall_metrics, segment_metrics
    
    def calculate_metrics(self, predictions, labels):
        """기본 성능 지표 계산"""
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        # ROC AUC
        fpr, tpr, _ = roc_curve(labels, predictions)
        roc_auc = auc(fpr, tpr)
        
        # PR AUC
        precision, recall, _ = precision_recall_curve(labels, predictions)
        pr_auc = auc(recall, precision)
        
        # 분류 보고서
        binary_predictions = (predictions > 0.5).astype(int)
        classification_rep = classification_report(labels, binary_predictions, 
                                               target_names=['Normal', 'Anomaly'], 
                                               output_dict=True)
        
        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'fpr': fpr,
            'tpr': tpr,
            'precision': precision,
            'recall': recall,
            'classification_report': classification_rep,
            'binary_predictions': binary_predictions
        }
    
    def analyze_segment_performance(self, results):
        """세그먼트별 성능 분석"""
        segment_metrics = {}
        
        for category in set(results['categories']):
            # 해당 카테고리의 인덱스 찾기
            cat_indices = [i for i, cat in enumerate(results['categories']) if cat == category]
            
            if len(cat_indices) < 5:  # 너무 적은 샘플은 스킵
                continue
            
            cat_predictions = [results['predictions'][i] for i in cat_indices]
            cat_labels = [results['labels'][i] for i in cat_indices]
            
            # 카테고리별 성능 계산
            try:
                fpr, tpr, _ = roc_curve(cat_labels, cat_predictions)
                roc_auc = auc(fpr, tpr)
                
                precision, recall, _ = precision_recall_curve(cat_labels, cat_predictions)
                pr_auc = auc(recall, precision)
                
                segment_metrics[category] = {
                    'roc_auc': roc_auc,
                    'pr_auc': pr_auc,
                    'sample_count': len(cat_indices),
                    'anomaly_count': sum(cat_labels),
                    'normal_count': len(cat_labels) - sum(cat_labels)
                }
            except:
                segment_metrics[category] = {
                    'roc_auc': 0.0,
                    'pr_auc': 0.0,
                    'sample_count': len(cat_indices),
                    'anomaly_count': sum(cat_labels),
                    'normal_count': len(cat_labels) - sum(cat_labels)
                }
        
        return segment_metrics
    
    def print_results(self, overall_metrics, segment_metrics):
        """결과 출력"""
        print("\n" + "="*50)
        print("전체 성능")
        print("="*50)
        print(f"ROC AUC: {overall_metrics['roc_auc']:.4f}")
        print(f"PR AUC: {overall_metrics['pr_auc']:.4f}")
        print(f"정확도: {overall_metrics['classification_report']['accuracy']:.4f}")
        print(f"정밀도: {overall_metrics['classification_report']['1']['precision']:.4f}")
        print(f"재현율: {overall_metrics['classification_report']['1']['recall']:.4f}")
        print(f"F1-Score: {overall_metrics['classification_report']['1']['f1-score']:.4f}")
        
        print("\n" + "="*50)
        print("세그먼트별 성능")
        print("="*50)
        for category, metrics in sorted(segment_metrics.items()):
            print(f"\n{category}:")
            print(f"  샘플 수: {metrics['sample_count']}")
            print(f"  정상: {metrics['normal_count']}, 비정상: {metrics['anomaly_count']}")
            print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
            print(f"  PR AUC: {metrics['pr_auc']:.4f}")
    
    def visualize_results(self, results, overall_metrics, segment_metrics, output_dir):
        """결과 시각화"""
        # 1. ROC 곡선
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(overall_metrics['fpr'], overall_metrics['tpr'], 
                label=f'Overall (AUC = {overall_metrics["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        
        # 2. PR 곡선
        plt.subplot(2, 2, 2)
        plt.plot(overall_metrics['recall'], overall_metrics['precision'], 
                label=f'Overall (AUC = {overall_metrics["pr_auc"]:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        
        # 3. 세그먼트별 ROC AUC 비교
        plt.subplot(2, 2, 3)
        categories = list(segment_metrics.keys())
        roc_aucs = [segment_metrics[cat]['roc_auc'] for cat in categories]
        
        bars = plt.bar(categories, roc_aucs)
        plt.ylabel('ROC AUC')
        plt.title('세그먼트별 ROC AUC')
        plt.xticks(rotation=45)
        
        # 색상으로 성능 구분
        for i, (bar, auc_val) in enumerate(zip(bars, roc_aucs)):
            if auc_val >= 0.8:
                bar.set_color('green')
            elif auc_val >= 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # 4. 세그먼트별 샘플 수
        plt.subplot(2, 2, 4)
        sample_counts = [segment_metrics[cat]['sample_count'] for cat in categories]
        plt.bar(categories, sample_counts)
        plt.ylabel('샘플 수')
        plt.title('세그먼트별 샘플 수')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/custom_test_results.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. UMAP 임베딩 시각화
        try:
            import umap
            feats = np.array(results['features'])
            fit = umap.UMAP()
            reduced_feats = fit.fit_transform(feats)
            
            plt.figure(figsize=(10, 8))
            labels = np.array(results['labels'])
            categories = np.array(results['categories'])
            
            # 정상/비정상으로 색상 구분
            plt.scatter(reduced_feats[labels == 0, 0], reduced_feats[labels == 0, 1], 
                       c='tab:blue', label='Normal', marker='o', alpha=0.7)
            plt.scatter(reduced_feats[labels == 1, 0], reduced_feats[labels == 1, 1], 
                       c='tab:red', label='Anomaly', marker='*', alpha=0.7)
            
            plt.title('UMAP Embedding of Custom Data Features')
            plt.xlabel('UMAP Dimension 1')
            plt.ylabel('UMAP Dimension 2')
            plt.legend()
            plt.savefig(f"{output_dir}/custom_umap_embedding.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except ImportError:
            print("UMAP이 설치되지 않아 임베딩 시각화를 건너뜁니다.")
    
    def save_detailed_results(self, results, overall_metrics, segment_metrics, output_dir):
        """상세 결과 저장"""
        # JSON 형태로 결과 저장
        output_results = {
            'overall_metrics': overall_metrics,
            'segment_metrics': segment_metrics,
            'detailed_predictions': []
        }
        
        # 개별 예측 결과 저장
        for i in range(len(results['predictions'])):
            output_results['detailed_predictions'].append({
                'prediction': float(results['predictions'][i]),
                'label': int(results['labels'][i]),
                'category': results['categories'][i],
                'description': results['descriptions'][i]
            })
        
        with open(f"{output_dir}/custom_detailed_results.json", 'w', encoding='utf-8') as f:
            json.dump(output_results, f, indent=2, ensure_ascii=False)
        
        # CSV 형태로도 저장
        try:
            import pandas as pd
            df = pd.DataFrame({
                'prediction': results['predictions'],
                'label': results['labels'],
                'category': results['categories'],
                'description': results['descriptions']
            })
            df.to_csv(f"{output_dir}/custom_test_predictions.csv", index=False, encoding='utf-8')
        except ImportError:
            print("pandas가 설치되지 않아 CSV 저장을 건너뜁니다.")
        
        print(f"\n상세 결과가 {output_dir}에 저장되었습니다.")

def main():
    parser = argparse.ArgumentParser(description='커스텀 데이터셋 STEAD 테스트')
    parser.add_argument('--model_path', required=True, help='훈련된 모델 경로 (saved_models 폴더 내)')
    parser.add_argument('--test_list', default='custom_data/custom_test.txt', help='테스트 데이터 리스트')
    parser.add_argument('--output_dir', default='./test_results', help='결과 저장 디렉토리')
    parser.add_argument('--batch_size', type=int, default=16, help='배치 크기')
    parser.add_argument('--model_arch', default='tiny', help='모델 아키텍처 (base, fast, tiny)')
    
    args = parser.parse_args()
    
    # 모델 경로 확인
    if not os.path.exists(args.model_path):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {args.model_path}")
        print("saved_models 폴더 내의 모델 파일 경로를 확인하세요.")
        return
    
    # STEAD 옵션 설정
    stead_args = option.parse_args()
    stead_args.test_rgb_list = args.test_list
    stead_args.batch_size = args.batch_size
    stead_args.model_arch = args.model_arch
    
    # 테스트 실행
    try:
        tester = CustomTester(args.model_path, stead_args)
        
        # 테스트 데이터 로더 생성
        from torch.utils.data import DataLoader
        test_dataset = SegmentDataset(stead_args, test_mode=True)
        
        if len(test_dataset) == 0:
            print("❌ 테스트 데이터를 찾을 수 없습니다.")
            print(f"테스트 리스트 파일을 확인하세요: {args.test_list}")
            return
        
        test_loader = DataLoader(test_dataset, batch_size=stead_args.batch_size, 
                               shuffle=False, num_workers=0)
        
        print(f"테스트 데이터: {len(test_dataset)}개")
        
        # 테스트 수행
        results, overall_metrics, segment_metrics = tester.test_custom_data(
            test_loader, save_results=True, output_dir=args.output_dir
        )
        
        print("\n=== 테스트 완료 ===")
        
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

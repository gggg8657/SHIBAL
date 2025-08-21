import torch.utils.data as data
import numpy as np
import torch
import random
import os

class SegmentDataset(data.Dataset):
    def __init__(self, args, test_mode=False):
        self.test_mode = test_mode
        
        if test_mode:
            self.data_list_file = args.test_rgb_list
        else:
            self.data_list_file = args.rgb_list
        
        self.load_data_list()
        
        # 실제 데이터 수에 맞게 동적으로 계산
        normal_data = [path for path, info in self.segment_info.items() if info['label'] == 0.0]
        abnormal_data = [path for path, info in self.segment_info.items() if info['label'] == 1.0]
        
        self.n_len = len(normal_data)    # 정상 데이터 수
        self.a_len = len(abnormal_data)  # 비정상 데이터 수
        
        print(f"📊 데이터 분포: 정상 {self.n_len}개, 비정상 {self.a_len}개")
    
    def load_data_list(self):
        """세그먼트 정보가 포함된 데이터 리스트 로드"""
        self.data_list = []
        self.segment_info = {}
        
        if not os.path.exists(self.data_list_file):
            print(f"데이터 리스트 파일이 없습니다: {self.data_list_file}")
            return
        
        valid_count = 0
        missing_count = 0
        
        with open(self.data_list_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 4:
                    feature_path = parts[0]
                    category = parts[1]
                    label = float(parts[2])
                    description = parts[3]
                    
                    # 파일 존재 여부 확인
                    if os.path.exists(feature_path):
                        self.data_list.append(feature_path)
                        self.segment_info[feature_path] = {
                            'category': category,
                            'label': label,
                            'description': description
                        }
                        valid_count += 1
                    else:
                        missing_count += 1
                elif len(parts) == 1:
                    # 기존 형식 지원 (경로만 있는 경우)
                    feature_path = parts[0]
                    label = 0.0 if "Normal" in feature_path else 1.0
                    
                    if os.path.exists(feature_path):
                        self.data_list.append(feature_path)
                        self.segment_info[feature_path] = {
                            'category': 'unknown',
                            'label': label,
                            'description': 'unknown'
                        }
                        valid_count += 1
                    else:
                        missing_count += 1
        
        print(f"✅ 유효한 데이터: {valid_count}개")
        if missing_count > 0:
            print(f"❌ 누락된 데이터: {missing_count}개")
    
    def __getitem__(self, index):
        if not self.test_mode:
            # 훈련 모드: 정상/비정상 쌍으로 반환
            if index == 0:
                # 정상/비정상 데이터 인덱스 분리
                normal_indices = [i for i, path in enumerate(self.data_list) 
                                if self.segment_info[path]['label'] == 0.0]
                abnormal_indices = [i for i, path in enumerate(self.data_list) 
                                  if self.segment_info[path]['label'] == 1.0]
                
                self.n_ind = normal_indices.copy()
                self.a_ind = abnormal_indices.copy()
                random.shuffle(self.n_ind)
                random.shuffle(self.a_ind)
            
            # 인덱스가 범위를 벗어나면 처음부터 다시 시작
            if not self.n_ind or not self.a_ind:
                normal_indices = [i for i, path in enumerate(self.data_list) 
                                if self.segment_info[path]['label'] == 0.0]
                abnormal_indices = [i for i, path in enumerate(self.data_list) 
                                  if self.segment_info[path]['label'] == 1.0]
                
                self.n_ind = normal_indices.copy()
                self.a_ind = abnormal_indices.copy()
                random.shuffle(self.n_ind)
                random.shuffle(self.a_ind)
            
            nindex = self.n_ind.pop()
            aindex = self.a_ind.pop()
            
            # 정상 데이터
            npath = self.data_list[nindex]
            nfeatures = np.load(npath, allow_pickle=True).astype(np.float32)
            nlabel = self.segment_info[npath]['label']
            ncategory = self.segment_info[npath]['category']
            
            # 비정상 데이터
            apath = self.data_list[aindex]
            afeatures = np.load(apath, allow_pickle=True).astype(np.float32)
            alabel = self.segment_info[apath]['label']
            acategory = self.segment_info[apath]['category']
            
            return nfeatures, nlabel, afeatures, alabel, ncategory, acategory
        
        else:
            # 테스트 모드: 단일 데이터 반환
            path = self.data_list[index]
            features = np.load(path, allow_pickle=True).astype(np.float32)
            label = self.segment_info[path]['label']
            category = self.segment_info[path]['category']
            description = self.segment_info[path]['description']
            
            return features, label, category, description
    
    def __len__(self):
        if self.test_mode:
            return len(self.data_list)
        else:
            # 정상과 비정상 데이터 중 더 작은 값만큼 반환
            return min(self.a_len, self.n_len)

# 기존 데이터셋과의 호환성을 위한 래퍼
class Dataset(data.Dataset):
    def __init__(self, args, test_mode=False):
        self.segment_dataset = SegmentDataset(args, test_mode)
        self.test_mode = test_mode
    
    def __getitem__(self, index):
        if not self.test_mode:
            nfeatures, nlabel, afeatures, alabel, ncategory, acategory = self.segment_dataset[index]
            return nfeatures, nlabel, afeatures, alabel
        else:
            features, label, category, description = self.segment_dataset[index]
            return features, label
    
    def __len__(self):
        return len(self.segment_dataset)

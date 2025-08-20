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
        self.n_len = 800  # 정상 데이터 수
        self.a_len = len(self.data_list) - self.n_len
    
    def load_data_list(self):
        """세그먼트 정보가 포함된 데이터 리스트 로드"""
        self.data_list = []
        self.segment_info = {}
        
        if not os.path.exists(self.data_list_file):
            print(f"데이터 리스트 파일이 없습니다: {self.data_list_file}")
            return
        
        with open(self.data_list_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 4:
                    feature_path = parts[0]
                    category = parts[1]
                    label = float(parts[2])
                    description = parts[3]
                    
                    self.data_list.append(feature_path)
                    self.segment_info[feature_path] = {
                        'category': category,
                        'label': label,
                        'description': description
                    }
                elif len(parts) == 1:
                    # 기존 형식 지원 (경로만 있는 경우)
                    feature_path = parts[0]
                    label = 0.0 if "Normal" in feature_path else 1.0
                    
                    self.data_list.append(feature_path)
                    self.segment_info[feature_path] = {
                        'category': 'unknown',
                        'label': label,
                        'description': 'unknown'
                    }
    
    def __getitem__(self, index):
        if not self.test_mode:
            # 훈련 모드: 정상/비정상 쌍으로 반환
            if index == 0:
                self.n_ind = list(range(self.a_len, len(self.data_list)))
                self.a_ind = list(range(self.a_len))
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

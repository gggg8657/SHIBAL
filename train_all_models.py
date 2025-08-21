#!/usr/bin/env python3
"""
8개 모델 동시 학습 스크립트
uncropped/cropped + tiny/base + finetune/scratch 조합으로 총 8개 모델을 동시 학습
"""

import subprocess
import os
import time
import signal
import sys
from concurrent.futures import ProcessPoolExecutor
import argparse

def train_single_model(config_file, gpu_id=None, additional_args=None):
    """단일 모델 학습 함수"""
    cmd = ["python3", "train_custom.py", "--config", config_file]
    
    if gpu_id is not None:
        cmd.extend(["--gpu_ids", str(gpu_id)])
    
    if additional_args:
        cmd.extend(additional_args)
    
    print(f"🚀 시작: {config_file} (GPU: {gpu_id if gpu_id is not None else 'auto'})")
    print(f"명령어: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600*24)  # 24시간 타임아웃
        if result.returncode == 0:
            print(f"✅ 완료: {config_file}")
            return True
        else:
            print(f"❌ 실패: {config_file}")
            print(f"에러: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ 타임아웃: {config_file}")
        return False
    except Exception as e:
        print(f"❌ 예외: {config_file} - {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='8개 모델 동시 학습')
    parser.add_argument('--mode', choices=['all', 'finetune', 'scratch', 'uncropped', 'cropped'], 
                       default='all', help='학습 모드 (기본값: all)')
    parser.add_argument('--gpu_assignment', nargs='+', default=None,
                       help='GPU 할당 (예: 0 1 0 1 0 1 0 1)')
    parser.add_argument('--parallel', action='store_true', default=True,
                       help='병렬 실행 (기본값: True)')
    parser.add_argument('--max_workers', type=int, default=8,
                       help='최대 워커 수 (기본값: 8)')
    
    args = parser.parse_args()
    
    # 8개 모델 설정 파일 리스트
    all_configs = [
        # 파인튜닝 모델들
        "config_uncropped_tiny_finetune.json",
        "config_uncropped_base_finetune.json", 
        "config_cropped_tiny_finetune.json",
        "config_cropped_base_finetune.json",
        # From scratch 모델들
        "config_uncropped_tiny_scratch.json",
        "config_uncropped_base_scratch.json",
        "config_cropped_tiny_scratch.json", 
        "config_cropped_base_scratch.json"
    ]
    
    # 모드별 설정 파일 선택
    if args.mode == 'finetune':
        configs = all_configs[:4]  # 파인튜닝만
        print("🎯 파인튜닝 모델 4개 학습")
    elif args.mode == 'scratch':
        configs = all_configs[4:]  # From scratch만
        print("🎯 From scratch 모델 4개 학습")
    elif args.mode == 'uncropped':
        configs = [all_configs[0], all_configs[1], all_configs[4], all_configs[5]]  # uncropped만
        print("🎯 Uncropped 모델 4개 학습")
    elif args.mode == 'cropped':
        configs = [all_configs[2], all_configs[3], all_configs[6], all_configs[7]]  # cropped만
        print("🎯 Cropped 모델 4개 학습")
    else:  # all
        configs = all_configs
        print("🎯 모든 모델 8개 학습")
    
    print(f"설정 파일들: {configs}")
    print(f"병렬 실행: {args.parallel}")
    print(f"최대 워커 수: {args.max_workers}")
    
    # GPU 할당 설정
    gpu_assignments = []
    if args.gpu_assignment:
        for gpu_str in args.gpu_assignment:
            if ',' in gpu_str:
                gpu_assignments.append(gpu_str)  # 멀티 GPU
            else:
                gpu_assignments.append(int(gpu_str))  # 단일 GPU
    else:
        # 자동 할당: GPU 0, 1번 번갈아가며 할당
        for i in range(len(configs)):
            if i % 2 == 0:
                gpu_assignments.append("0")  # GPU 0번
            else:
                gpu_assignments.append("1")  # GPU 1번
    
    print(f"GPU 할당: {gpu_assignments}")
    
    # 모델 정보 출력
    print("\n📋 학습할 모델 목록:")
    for i, config in enumerate(configs):
        model_type = "파인튜닝" if "finetune" in config else "From Scratch"
        data_type = "Uncropped" if "uncropped" in config else "Cropped"
        arch_type = "Tiny" if "tiny" in config else "Base"
        gpu_id = gpu_assignments[i] if i < len(gpu_assignments) else "auto"
        print(f"  {i+1}. {data_type} {arch_type} {model_type} (GPU: {gpu_id})")
    
    if args.parallel:
        # 병렬 실행
        print("\n🔄 병렬 실행 시작...")
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = []
            for config_file, gpu_id in zip(configs, gpu_assignments):
                future = executor.submit(train_single_model, config_file, gpu_id)
                futures.append(future)
            
            # 결과 수집
            results = []
            for future in futures:
                results.append(future.result())
        
        # 결과 요약
        success_count = sum(results)
        total_count = len(results)
        print(f"\n📊 학습 완료 요약:")
        print(f"성공: {success_count}/{total_count}")
        print(f"실패: {total_count - success_count}/{total_count}")
        
        # 성공한 모델 목록
        print(f"\n✅ 성공한 모델들:")
        for i, (config, result) in enumerate(zip(configs, results)):
            if result:
                model_type = "파인튜닝" if "finetune" in config else "From Scratch"
                data_type = "Uncropped" if "uncropped" in config else "Cropped"
                arch_type = "Tiny" if "tiny" in config else "Base"
                print(f"  {data_type} {arch_type} {model_type}")
        
    else:
        # 순차 실행
        print("\n🔄 순차 실행 시작...")
        results = []
        for config_file, gpu_id in zip(configs, gpu_assignments):
            result = train_single_model(config_file, gpu_id)
            results.append(result)
            time.sleep(5)  # 5초 대기
        
        # 결과 요약
        success_count = sum(results)
        total_count = len(results)
        print(f"\n📊 학습 완료 요약:")
        print(f"성공: {success_count}/{total_count}")
        print(f"실패: {total_count - success_count}/{total_count}")

if __name__ == "__main__":
    main()

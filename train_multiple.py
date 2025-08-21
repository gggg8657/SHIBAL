#!/usr/bin/env python3
"""
여러 모델 동시 학습 스크립트
여러 설정 파일을 사용하여 병렬로 모델을 학습합니다.
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
    cmd = [sys.executable, "train_custom.py", "--config", config_file]
    
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
    parser = argparse.ArgumentParser(description='여러 모델 동시 학습')
    parser.add_argument('--configs', nargs='+', default=['config_tiny.json', 'config_base.json'], 
                       help='사용할 설정 파일들')
    parser.add_argument('--gpu_assignment', nargs='+', default=None,
                       help='GPU 할당 (예: 0 1 또는 0,1 0,1)')
    parser.add_argument('--parallel', action='store_true', default=True,
                       help='병렬 실행 (기본값: True)')
    parser.add_argument('--max_workers', type=int, default=2,
                       help='최대 워커 수 (기본값: 2)')
    
    args = parser.parse_args()
    
    print("🎯 여러 모델 동시 학습 시작")
    print(f"설정 파일들: {args.configs}")
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
        for i in range(len(args.configs)):
            if i % 2 == 0:
                gpu_assignments.append("0")  # GPU 0번
            else:
                gpu_assignments.append("1")  # GPU 1번
    
    print(f"GPU 할당: {gpu_assignments}")
    
    if args.parallel:
        # 병렬 실행
        print("\n🔄 병렬 실행 시작...")
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = []
            for config_file, gpu_id in zip(args.configs, gpu_assignments):
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
        
    else:
        # 순차 실행
        print("\n🔄 순차 실행 시작...")
        results = []
        for config_file, gpu_id in zip(args.configs, gpu_assignments):
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

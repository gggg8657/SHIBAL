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
from pathlib import Path

def train_single_model(config_file, gpu_id=None, additional_args=None, stream=False, log_dir="logs"):
    """단일 모델 학습 함수"""
    cmd = [sys.executable, "train_custom.py", "--config", config_file]
    
    if gpu_id is not None:
        cmd.extend(["--gpu_ids", str(gpu_id)])
        visible = str(gpu_id)
    else:
        visible = ""
    
    if additional_args:
        cmd.extend(additional_args)
    
    # 로그 디렉토리 준비
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    cfg_name = os.path.splitext(os.path.basename(config_file))[0]
    log_path = os.path.join(log_dir, f"{cfg_name}.log")
    
    print(f"🚀 시작: {config_file} (GPU: {gpu_id if gpu_id is not None else 'auto'})")
    print(f"명령어: {' '.join(cmd)}")
    print(f"📄 로그: {log_path}")
    
    # 환경변수 설정 (GPU 고정)
    env = os.environ.copy()
    if visible:
        env["CUDA_VISIBLE_DEVICES"] = visible
    
    try:
        if stream:
            # 실시간 스트리밍 + 파일 저장
            with open(log_path, "w", encoding="utf-8", errors="ignore") as lf:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
                for line in proc.stdout:
                    sys.stdout.write(line)
                    lf.write(line)
                proc.wait()
                rc = proc.returncode
        else:
            # 캡처 + 파일 저장
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600*24, env=env)
            with open(log_path, "w", encoding="utf-8", errors="ignore") as lf:
                lf.write(result.stdout or "")
                lf.write("\n==== STDERR ====\n")
                lf.write(result.stderr or "")
            rc = result.returncode
        
        if rc == 0:
            print(f"✅ 완료: {config_file}")
            return True
        else:
            print(f"❌ 실패: {config_file}")
            print(f"🔎 자세한 로그: {log_path}")
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
    parser.add_argument('--max_workers', type=int, default=2,
                       help='최대 워커 수 (기본값: 2)')
    parser.add_argument('--stream', action='store_true', default=False,
                       help='실시간 로그 스트리밍 (기본값: False)')
    parser.add_argument('--log_dir', type=str, default='logs', help='로그 저장 디렉토리')
    
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
        configs = all_configs[:4]
        print("🎯 파인튜닝 모델 4개 학습")
    elif args.mode == 'scratch':
        configs = all_configs[4:]
        print("🎯 From scratch 모델 4개 학습")
    elif args.mode == 'uncropped':
        configs = [all_configs[0], all_configs[1], all_configs[4], all_configs[5]]
        print("🎯 Uncropped 모델 4개 학습")
    elif args.mode == 'cropped':
        configs = [all_configs[2], all_configs[3], all_configs[6], all_configs[7]]
        print("🎯 Cropped 모델 4개 학습")
    else:
        configs = all_configs
        print("🎯 모든 모델 8개 학습")
    
    print(f"설정 파일들: {configs}")
    print(f"병렬 실행: {args.parallel}")
    print(f"최대 워커 수: {args.max_workers}")
    
    # GPU 할당 설정
    gpu_assignments = []
    if args.gpu_assignment:
        for gpu_str in args.gpu_assignment:
            gpu_assignments.append(str(gpu_str))
    else:
        # 자동 할당: GPU 0, 1번 번갈아가며 할당
        for i in range(len(configs)):
            gpu_assignments.append("0" if i % 2 == 0 else "1")
    
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
        print("\n🔄 병렬 실행 시작...")
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = []
            for config_file, gpu_id in zip(configs, gpu_assignments):
                futures.append(executor.submit(train_single_model, config_file, gpu_id, None, args.stream, args.log_dir))
            results = [f.result() for f in futures]
    else:
        print("\n🔄 순차 실행 시작...")
        results = []
        for config_file, gpu_id in zip(configs, gpu_assignments):
            ok = train_single_model(config_file, gpu_id, None, args.stream, args.log_dir)
            results.append(ok)
            time.sleep(2)
    
    # 결과 요약
    success_count = sum(results)
    total_count = len(results)
    print(f"\n📊 학습 완료 요약:")
    print(f"성공: {success_count}/{total_count}")
    print(f"실패: {total_count - success_count}/{total_count}")
    
    print(f"\n✅ 로그 디렉토리: {os.path.abspath(args.log_dir)}")

if __name__ == "__main__":
    main()

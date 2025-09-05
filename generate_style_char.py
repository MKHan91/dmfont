"""
DM-Font를 이용한 진법언해 스타일 한글 11,172자 생성 스크립트

이 스크립트는 훈련된 DM-Font 모델을 사용하여 진법언해 고서체 스타일로
한글 완성형 전체 11,172자를 생성하는 도구입니다.

주요 기능:
- 진법언해 스타일 샘플 이미지들을 참조 스타일로 사용
- 한글 완성형 전체 범위(가~힣) 커버
- 배치 단위 처리로 메모리 효율적 생성
- 생성 실패한 글자들 추적 및 로깅
"""
import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import cv2
from torchvision import transforms

from models import MACore
from datasets import kor_decompose as kor
import utils


def get_all_korean_chars():
    """
    한글 유니코드 전체 11,172자를 생성하는 함수
    
    한글 완성형 범위:
    - 시작: 가(U+AC00, 44032)
    - 끝: 힣(U+D7A3, 55203)
    - 총 개수: 11,172자
    
    이 범위는 현대 한글의 모든 가능한 조합을 포함하며,
    초성(19개) × 중성(21개) × 종성(28개, 빈 종성 포함) = 11,172자
    
    Returns
    -------
    list
        한글 완성형 전체 11,172자의 문자 리스트
    """
    # 완성형 한글: 가(AC00) ~ 힣(D7A3) = 11,172자
    return [chr(code) for code in range(0xAC00, 0xD7A4)]


def load_jinbeop_style_samples(style_dir):
    """
    진법언해 스타일 샘플 글자들을 로드하는 함수
    
    진법언해 폰트의 기존 샘플 이미지들을 읽어와서 
    스타일 참조용으로 전처리합니다.
    
    Parameters
    ----------
    style_dir : str
        진법언해 스타일 글자 이미지들이 있는 디렉토리 경로
        각 파일은 "{문자}.png" 형식이어야 함
    
    Returns
    -------
    tuple
        (available_chars, style_images)
        - available_chars: 사용 가능한 문자들의 리스트
        - style_images: 전처리된 스타일 이미지 텐서들의 리스트
    """
    # 디렉토리에서 파일명(확장자 제외)으로 문자 추출
    # 하드코딩된 경로를 사용 - 실제로는 매개변수 style_dir를 사용해야 함
    style_chars = [fname[:-4] for fname in os.listdir("/home/dev/dmfont/datasets/Jinbeop_font_image")]
    style_images = []
    available_chars = []
    
    # 이미지 전처리 변환 정의
    # DM-Font 모델에서 사용하는 표준 전처리 파이프라인
    transform = transforms.Compose([
        transforms.ToTensor(),          # PIL Image -> Tensor 변환, [0,1] 범위
        transforms.Normalize([0.5], [0.5])  # [0,1] -> [-1,1] 정규화 (GAN 표준)
    ])
    
    # 각 문자에 대해 이미지 로드 및 전처리
    for char in style_chars:
        style_path = Path(style_dir) / f"{char}.png"
        
        if style_path.exists():
            # 이미지를 그레이스케일로 로드
            img = cv2.imread(str(style_path), cv2.IMREAD_GRAYSCALE)

            # 모델 입력 크기(128x128)로 리사이즈
            # LANCZOS4: 고품질 리샘플링 방법 (문자 이미지에 적합)
            img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LANCZOS4)
            
            # PyTorch 텐서로 변환 및 정규화
            img = transform(img)
            
            style_images.append(img)
            available_chars.append(char)
    
    return available_chars, style_images


def generate_all_korean_chars(model, style_chars, style_images, output_dir):
    """
    DM-Font 모델을 사용하여 모든 한글 11,172자를 생성하는 함수
    
    이 함수는 다음과 같은 과정으로 동작합니다:
    1. 스타일 이미지들을 모델에 인코딩 (한 번만 실행)
    2. 모든 한글 문자를 배치별로 처리
    3. 각 문자에 대해 자소 분해 후 생성
    4. 생성된 이미지를 파일로 저장
    
    Parameters
    ----------
    model : MACore
        훈련된 DM-Font 생성자 모델
    style_chars : list
        스타일 참조용 문자들의 리스트
    style_images : list
        전처리된 스타일 이미지 텐서들의 리스트
    output_dir : str
        생성된 이미지들을 저장할 출력 디렉토리
    """
    
    # 출력 디렉토리 생성
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 모든 한글 문자 목록 가져오기
    all_chars = get_all_korean_chars()
    
    print(f"총 {len(all_chars)}자 생성 시작...")
    
    # 모델을 평가 모드로 설정 (dropout, batch normalization 비활성화)
    model.eval()
    
    # 그라디언트 계산 비활성화 (추론 시 메모리 절약 및 속도 향상)
    with torch.no_grad():
        # === 스타일 인코딩 단계 (한 번만 실행) ===
        
        # 스타일 폰트 ID 생성 (모든 스타일 샘플이 같은 폰트(진법언해)에서 온다고 가정)
        style_ids = torch.tensor([0] * len(style_chars), device='cuda')  # 폰트 ID 0 = 진법언해
        
        # 스타일 글자들의 자소 분해
        # 한국어 자소 분해: 각 문자를 초성, 중성, 종성으로 분리
        style_comp_ids = []
        for char in style_chars:
            comps = kor.decompose(char)  # [초성_id, 중성_id, 종성_id] 반환
            style_comp_ids.append(comps)
        style_comp_ids = torch.tensor(style_comp_ids, device='cuda')
        
        # 스타일 이미지들을 GPU로 이동
        style_imgs = torch.stack(style_images).cuda()
        
        # 스타일 정보를 모델의 메모리에 저장
        # 이 단계에서 스타일 특징들이 추출되어 모델 내부 메모리에 저장됨
        model.encode_write(style_ids, style_comp_ids, style_imgs)
        
        # === 배치별 문자 생성 단계 ===
        
        # 배치 크기 설정 (GPU 메모리에 따라 조정 가능)
        batch_size = 64  # 한 번에 64자씩 처리
        n_batches = (len(all_chars) + batch_size - 1) // batch_size  # 전체 배치 수 계산
        
        # 통계 변수 초기화
        success_count = 0
        failed_chars = []
        
        # 배치별 처리 루프
        for batch_idx in tqdm(range(n_batches), desc="생성 중"):
            # 현재 배치의 시작/끝 인덱스 계산
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(all_chars))
            batch_chars = all_chars[start_idx:end_idx]
            
            # 현재 배치의 결과를 저장할 리스트
            batch_results = []
            batch_valid_chars = []
            
            # 배치 내 각 문자에 대해 생성 수행
            for char in batch_chars:
                try:
                    # 한글 자소 분해
                    comps = kor.decompose(char)
                    comps_tensor = torch.tensor(comps, device='cuda').unsqueeze(0)  # 배치 차원 추가
                    
                    # 타겟 폰트 ID (진법언해 폰트)
                    trg_id = torch.tensor([0], device='cuda')
                    
                    # DM-Font 모델을 사용한 폰트 생성
                    # 메모리에 저장된 스타일 정보를 읽어와서 새로운 문자 생성
                    generated = model.read_decode(trg_id, comps_tensor)
                    
                    batch_results.append(generated[0])  # 첫 번째 (유일한) 결과
                    batch_valid_chars.append(char)
                    
                except Exception as e:
                    # 생성 실패 시 오류 로깅
                    print(f"생성 실패: {char} - {str(e)}")
                    failed_chars.append(char)
                    continue
            
            # === 배치 결과 저장 ===
            for char, img_tensor in zip(batch_valid_chars, batch_results):
                # 유니코드 값을 16진수로 변환하여 파일명에 포함
                unicode_hex = f"{ord(char):04X}"  # 4자리 대문자 16진수
                filename = f"jinbeop_{char}_{unicode_hex}.png"
                filepath = output_dir / filename
                
                # 텐서를 이미지 파일로 저장
                # utils.save_tensor_to_image는 [-1,1] 범위의 텐서를 [0,255] PNG로 변환
                utils.save_tensor_to_image(img_tensor, filepath)
                success_count += 1
        
        # === 최종 결과 출력 ===
        print(f"\n생성 완료!")
        print(f"성공: {success_count}자")
        print(f"실패: {len(failed_chars)}자")
        
        # 실패한 글자들이 있으면 상세 정보 출력
        if failed_chars:
            # 실패 목록이 많으면 처음 10개만 출력
            if len(failed_chars) > 10:
                print(f"실패한 글자들: {failed_chars[:10]}...")
            else:
                print(f"실패한 글자들: {failed_chars}")
            
            # 실패한 글자 목록을 파일로 저장 (추후 분석용)
            with open(output_dir / "failed_chars.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(failed_chars))


def main():
    """
    메인 실행 함수
    
    명령줄 인자를 파싱하고 전체 생성 파이프라인을 실행합니다.
    """
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description="진법언해 스타일 한글 전체 생성 도구")
    parser.add_argument("--model_path", 
                        default="/home/dev/dmfont/experiments/checkpoints/250831_05-11-52_test/last.pth",
                        help="학습된 DM-Font 모델 체크포인트 경로")
    parser.add_argument("--style_dir", 
                        default="/home/dev/dmfont/datasets/Jinbeop_font_image",
                        help="진법언해 스타일 샘플 이미지 디렉토리")
    parser.add_argument("--output_dir", 
                        default="/home/dev/dmfont/results/Jinbeop",
                        help="생성된 글자들을 저장할 디렉토리")
    parser.add_argument("--config_path", 
                        default="cfgs/kor.yaml",
                        help="모델 설정 파일 경로")
    
    args = parser.parse_args()
    
    # === 설정 파일 로드 ===
    from sconf import Config
    cfg = Config(args.config_path)
    
    # === 모델 초기화 및 로드 ===
    print("모델 로딩 중...")
    
    # 한국어 자소 관련 상수
    n_comps = kor.N_COMPONENTS    # 전체 자소 개수 (초성+중성+종성)
    n_comp_types = 3              # 자소 타입 수 (초성, 중성, 종성)
    
    # 모델 하이퍼파라미터 추출
    g_kwargs = cfg.get('g_args', {})
    
    # DM-Font 생성자 모델 초기화
    gen = MACore(
        1, cfg['C'], 1,             # 입력 채널, 특징 채널, 출력 채널
        **g_kwargs,                 # 추가 생성자 인자들
        n_comps=n_comps,            # 전체 구성 요소 수
        n_comp_types=n_comp_types,  # 구성 요소 타입 수
        language='kor'              # 한국어 설정
    )
    gen.cuda()  # 모델을 GPU로 이동
    
    # === 체크포인트에서 모델 가중치 로드 ===
    print(f"체크포인트 로딩: {args.model_path}")
    ckpt = torch.load(args.model_path, map_location='cuda')
    
    # EMA(Exponential Moving Average) 모델 사용
    # EMA는 일반적으로 더 안정적이고 좋은 생성 품질을 제공
    gen.load_state_dict(ckpt['generator_ema'])
    print("모델 로딩 완료!")
    
    # === 스타일 샘플 로드 ===
    print("진법언해 스타일 샘플 로딩 중...")
    style_chars, style_images = load_jinbeop_style_samples(args.style_dir)
    
    # 스타일 샘플 유효성 검사
    if len(style_chars) == 0:
        print("ERROR: 스타일 샘플을 찾을 수 없습니다!")
        print(f"경로를 확인하세요: {args.style_dir}")
        return
    
    print(f"로드된 스타일 샘플: {len(style_chars)}개")
    print(f"샘플 문자들: {style_chars}")
    
    # === 전체 한글 생성 실행 ===
    print("\n한글 11,172자 생성을 시작합니다...")
    generate_all_korean_chars(gen, style_chars, style_images, args.output_dir)
    
    print(f"\n모든 작업이 완료되었습니다!")
    print(f"결과 저장 위치: {args.output_dir}")


if __name__ == "__main__":
    main()

# === 실행 예시 ===
# 기본 설정으로 실행:
# python generate_jinbeop_complete.py

# 사용자 정의 설정으로 실행:
# python generate_jinbeop_complete.py \
#     --model_path /home/dev/dmfont/checkpoints/250717_04-46-31_test/150000-test.pth \
#     --style_dir ./jinbeop_style_samples \
#     --output_dir ./jinbeop_complete_11172

# === 주요 특징 ===
# 1. 메모리 효율성: 배치 단위 처리로 GPU 메모리 사용량 최적화
# 2. 오류 처리: 생성 실패한 문자들 추적 및 로깅
# 3. 진행률 표시: tqdm을 사용한 실시간 진행률 표시
# 4. 파일명 표준화: 유니코드 값을 포함한 체계적인 파일명
# 5. 결과 검증: 성공/실패 통계 및 실패 문자 목록 저장
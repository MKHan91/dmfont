"""
DM-Font를 이용한 진법언해 스타일 한글 11,172자 생성 스크립트
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
    """한글 유니코드 전체 11,172자 생성"""
    # 완성형 한글: 가(AC00) ~ 힣(D7A3) = 11,172자
    return [chr(code) for code in range(0xAC00, 0xD7A4)]

def load_jinbeop_style_samples(style_dir):
    """진법언해 스타일 샘플 글자들 로드
    Args:
        style_dir: 진법언해 스타일 글자 이미지들이 있는 디렉토리
    """
    style_chars = [fname[:-4] for fname in os.listdir("/home/dev/dmfont/datasets/Jinbeop_font_image")]
    style_images = []
    available_chars = []
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    for char in style_chars:
        style_path = Path(style_dir) / f"{char}.png"
        if style_path.exists():
            # 이미지 로드 및 전처리
            img = cv2.imread(str(style_path), cv2.IMREAD_GRAYSCALE)

            img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LANCZOS4)
            img = transform(img)
            
            style_images.append(img)
            available_chars.append(char)
    
    return available_chars, style_images

def generate_all_korean_chars(model, style_chars, style_images, output_dir):
    """모든 한글 11,172자 생성"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 모든 한글 문자 가져오기
    all_chars = get_all_korean_chars()
    
    print(f"총 {len(all_chars)}자 생성 시작...")
    
    # 스타일 인코딩 (한 번만 실행)
    model.eval()
    with torch.no_grad():
        # 스타일 정보를 메모리에 저장
        style_ids = torch.tensor([0] * len(style_chars), device='cuda')  # 폰트 ID (진법언해)
        
        # 스타일 글자들의 자소 분해
        style_comp_ids = []
        for char in style_chars:
            comps = kor.decompose(char)
            style_comp_ids.append(comps)
        style_comp_ids = torch.tensor(style_comp_ids, device='cuda')
        
        # 스타일 이미지들을 GPU로
        style_imgs = torch.stack(style_images).cuda()
        
        # 스타일 인코딩
        model.encode_write(style_ids, style_comp_ids, style_imgs)
        
        # 배치별로 글자 생성
        batch_size = 64  # 메모리에 따라 조정
        n_batches = (len(all_chars) + batch_size - 1) // batch_size
        
        success_count = 0
        failed_chars = []
        
        for batch_idx in tqdm(range(n_batches), desc="생성 중"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(all_chars))
            batch_chars = all_chars[start_idx:end_idx]
            
            batch_results = []
            batch_valid_chars = []
            
            for char in batch_chars:
                try:
                    # 자소 분해
                    comps = kor.decompose(char)
                    comps_tensor = torch.tensor(comps, device='cuda').unsqueeze(0)
                    
                    # 타겟 ID (진법언해 폰트)
                    trg_id = torch.tensor([0], device='cuda')
                    
                    # 생성
                    generated = model.read_decode(trg_id, comps_tensor)
                    
                    batch_results.append(generated[0])
                    batch_valid_chars.append(char)
                    
                except Exception as e:
                    print(f"생성 실패: {char} - {str(e)}")
                    failed_chars.append(char)
                    continue
            
            # 배치 결과 저장
            for char, img_tensor in zip(batch_valid_chars, batch_results):
                # 유니코드 값으로 파일명 생성
                unicode_hex = f"{ord(char):04X}"
                filename = f"jinbeop_{char}_{unicode_hex}.png"
                filepath = output_dir / filename
                
                # 이미지 저장
                utils.save_tensor_to_image(img_tensor, filepath)
                success_count += 1
        
        print(f"\n생성 완료!")
        print(f"성공: {success_count}자")
        print(f"실패: {len(failed_chars)}자")
        
        if failed_chars:
            print(f"실패한 글자들: {failed_chars[:10]}..." if len(failed_chars) > 10 else f"실패한 글자들: {failed_chars}")
            
            # 실패한 글자 목록 저장
            with open(output_dir / "failed_chars.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(failed_chars))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/home/dev/dmfont/experiments/checkpoints/250831_05-11-52_test/last.pth",
                       help="학습된 DM-Font 모델 경로")
    parser.add_argument("--style_dir", default="/home/dev/dmfont/datasets/Jinbeop_font_image",
                       help="진법언해 스타일 샘플 이미지 디렉토리")
    parser.add_argument("--output_dir", default="/home/dev/dmfont/results/Jinbeop",
                       help="생성된 글자들을 저장할 디렉토리")
    parser.add_argument("--config_path", default="cfgs/kor.yaml",
                       help="모델 설정 파일 경로")
    
    args = parser.parse_args()
    
    # 설정 로드
    from sconf import Config
    cfg = Config(args.config_path)
    
    # 모델 로드
    print("모델 로딩 중...")
    n_comps = kor.N_COMPONENTS
    n_comp_types = 3
    
    g_kwargs = cfg.get('g_args', {})
    gen = MACore(
        1, cfg['C'], 1, 
        **g_kwargs, 
        n_comps=n_comps, 
        n_comp_types=n_comp_types,
        language='kor'
    )
    gen.cuda()
    
    # 체크포인트 로드
    ckpt = torch.load(args.model_path)
    gen.load_state_dict(ckpt['generator_ema'])  # EMA 모델 사용
    
    # 스타일 샘플 로드
    print("진법언해 스타일 샘플 로딩 중...")
    style_chars, style_images = load_jinbeop_style_samples(args.style_dir)
    
    if len(style_chars) == 0:
        print("ERROR: 스타일 샘플을 찾을 수 없습니다!")
        return
    
    print(f"로드된 스타일 샘플: {len(style_chars)}개 - {style_chars}")
    
    # 전체 한글 생성
    generate_all_korean_chars(gen, style_chars, style_images, args.output_dir)

if __name__ == "__main__":
    main()

# 실행 예시:
# python generate_jinbeop_complete.py \
#     --model_path /home/dev/dmfont/checkpoints/250717_04-46-31_test/150000-test.pth \
#     --style_dir ./jinbeop_style_samples \
#     --output_dir ./jinbeop_complete_11172
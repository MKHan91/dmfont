"""
DMFont - 딥러닝 기반 폰트 생성 모델의 평가 및 추론 코드
Copyright (c) 2020-present NAVER Corp.
MIT license

이 코드는 훈련된 DMFont 모델을 평가하고, 새로운 폰트 글리프를 생성하는 기능을 제공합니다.
주요 기능:
1. Cross-validation을 통한 정량적 평가 (L1, SSIM, MS-SSIM)
2. 시각적 비교를 위한 이미지 그리드 생성
3. 사용자 연구를 위한 폰트 생성
4. 2단계 추론 (인코딩 → 디코딩)
"""
from itertools import chain
from pathlib import Path
import json
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from sconf import Config

import utils
from logger import Logger

from models import MACore
from datasets import uniform_sample
from datasets import kor_decompose as kor
from datasets import thai_decompose as thai
from inference import (
    infer, get_val_loader,
    infer_2stage, get_val_encode_loader, get_val_decode_loader
)
from ssim import SSIM, MSSSIM


def torch_eval(val_fn):
    """
    평가 함수를 위한 데코레이터
    
    평가 중에는 모델을 eval 모드로 설정하고, gradient 계산을 비활성화합니다.
    평가 완료 후에는 다시 train 모드로 복원합니다.
    
    Args:
        val_fn: 평가 함수
        
    Returns:
        decorated: 데코레이팅된 함수
    """
    @torch.no_grad()
    def decorated(self, gen, *args, **kwargs):
        gen.eval()  # 평가 모드로 설정
        ret = val_fn(self, gen, *args, **kwargs)
        gen.train()  # 훈련 모드로 복원
        return ret
    return decorated


def get_all_hangul_chars():
    """
    모든 완성형 한글 문자들을 반환하는 함수
    
    유니코드 범위 0xAC00-0xD7A3 (가-힣)에 해당하는 모든 한글 문자를 생성합니다.
    
    Returns:
        list: 모든 한글 문자들의 리스트 ['가', '각', '간', ...]
    """
    return [chr(code) for code in range(0xAC00, 0xD7A4)]


# region - Evaluator
class Evaluator:
    """
    DMFont 평가기
    
    이 클래스는 다음과 같은 평가 기능을 제공합니다:
    1. 픽셀 레벨 평가 (L1, SSIM, MS-SSIM)
    2. 참조 스타일 샘플로부터 글리프 생성
    3. Cross-validation을 통한 정량적 성능 측정
    4. 시각적 비교를 위한 이미지 그리드 생성
    """
    
    def __init__(self, data, trn_avails, logger, writer, batch_size, transform,
                 content_font, language, meta, val_loaders, n_workers=2):
        """
        Evaluator 초기화
        
        Args:
            data: HDF5 데이터 객체
            trn_avails: 훈련 시 사용 가능한 폰트-문자 조합
            logger: 로거 객체
            writer: TensorBoard writer 또는 DiskWriter
            batch_size: 배치 크기
            transform: 이미지 전처리 변환
            content_font: 내용 폰트 (구조 정보용)
            language: 언어 코드 ('kor', 'thai' 등)
            meta: 메타데이터 (훈련/검증 폰트, 문자 정보)
            val_loaders: 검증용 데이터 로더들
            n_workers: 데이터 로딩 워커 수
        """
        self.data = data
        self.logger = logger
        self.writer = writer
        self.batch_size = batch_size
        self.transform = transform
        self.n_workers = n_workers
        self.unify_resize_method = True  # 공정한 평가를 위한 크기 통일 옵션

        self.trn_avails = trn_avails
        self.val_loaders = val_loaders
        self.content_font = content_font
        self.language = language
        
        # 언어별 컴포넌트 타입 수 설정
        if self.language == 'kor':
            self.n_comp_types = 3  # 초성, 중성, 종성
        elif self.language == 'thai':
            self.n_comp_types = 4  # 태국어 컴포넌트
        else:
            raise ValueError(f"Unsupported language: {self.language}")

        # 이미지 품질 평가 메트릭 설정
        self.SSIM = SSIM().cuda()  # Structural Similarity Index
        weights = [0.25, 0.3, 0.3, 0.15]  # MS-SSIM 가중치 (다중 스케일)
        self.MSSSIM = MSSSIM(weights=weights).cuda()

        # Cross-validation 배치 수 설정 (가장 작은 로더 기준)
        n_batches = [len(loader) for loader in self.val_loaders.values()]
        self.n_cv_batches = min(n_batches)
        self.logger.info("# of cross-validation batches = {}".format(self.n_cv_batches))

        # CV 시각화용 문자와 폰트 샘플링
        n_chars = 16  # 시각화할 문자 수
        n_fonts = 16  # 시각화할 폰트 수
        seen_chars = uniform_sample(meta['train']['chars'], n_chars//2)      # 훈련 시 본 문자
        unseen_chars = uniform_sample(meta['valid']['chars'], n_chars//2)    # 새로운 문자
        unseen_fonts = uniform_sample(meta['valid']['fonts'], n_fonts)       # 새로운 폰트

        self.cv_comparable_fonts = unseen_fonts
        self.cv_comparable_chars = seen_chars + unseen_chars

        # 비교 가능한 폰트-문자 조합 설정
        allchars = meta['train']['chars'] + meta['valid']['chars']
        self.cv_comparable_avails = {
            font: allchars
            for font in self.cv_comparable_fonts
        }

    def validation(self, gen, step, extra_tag=''):
        """
        전체 검증 실행
        
        1. 비교 가능한 검증 세트 평가
        2. 모든 검증 로더에 대한 교차 검증
        3. TensorBoard에 결과 로깅
        
        Args:
            gen: 생성자 모델
            step: 현재 훈련 스텝
            extra_tag: 추가 태그 (예: '_EMA')
            
        Returns:
            dict: 평가 결과 딕셔너리
        """
        # 비교 가능한 검증 세트로 시각적 평가
        self.comparable_validset_validation(gen, step, True, 'comparable_val'+extra_tag)

        plot_dic = {}
        # 각 검증 로더에 대해 정량적 평가 수행
        for tag, loader in self.val_loaders.items():
            tag = tag + extra_tag
            l1, ssim, msssim = self.cross_validation(
                gen, step, loader, tag, n_batches=self.n_cv_batches
            )
            # TensorBoard용 결과 저장
            plot_dic[f'val/{tag}/l1'] = l1
            plot_dic[f'val/{tag}/ssim'] = ssim
            plot_dic[f'val/{tag}/ms-ssim'] = msssim if not np.isnan(msssim) else 0.
        
        self.writer.add_scalars(plot_dic, step)
        return plot_dic

    @torch_eval
    def comparable_validset_validation(self, gen, step, compare_inputs=False, tag='comparable_val'):
        """
        CV에서 비교 가능한 검증 세트 평가
        
        선택된 폰트와 문자들에 대해 비교 가능한 그리드를 생성하여
        시각적으로 생성 품질을 평가합니다.
        
        Args:
            gen: 생성자 모델
            step: 현재 훈련 스텝
            compare_inputs: 입력 이미지도 비교할지 여부
            tag: 로깅 태그
        """
        comparable_grid = self.comparable_validation(
            gen, self.cv_comparable_avails, self.cv_comparable_fonts, self.cv_comparable_chars,
            n_max_match=1, compare_inputs=compare_inputs
        )
        self.writer.add_image(tag, comparable_grid, global_step=step)

    @torch_eval
    def comparable_validation(self, gen, style_avails, target_fonts, target_chars, n_max_match=3,
                              compare_inputs=False):
        """
        타겟 폰트와 문자들에 대한 수평 비교
        
        각 폰트별로 같은 문자들을 생성하여 수평으로 비교할 수 있는 그리드를 만듭니다.
        
        Args:
            gen: 생성자 모델
            style_avails: 스타일로 사용 가능한 폰트-문자 조합
            target_fonts: 타겟 폰트들
            target_chars: 타겟 문자들
            n_max_match: 최대 매칭 수
            compare_inputs: 입력 이미지도 포함할지 여부
            
        Returns:
            torch.Tensor: 비교 가능한 이미지 그리드
        """
        # 검증용 데이터 로더 생성
        loader = get_val_loader(
            self.data, target_fonts, target_chars, style_avails,
            B=self.batch_size, n_max_match=n_max_match, transform=self.transform,
            content_font=self.content_font, language=self.language, n_workers=self.n_workers
        )
        
        # 추론 실행 - 생성된 이미지들 [B, 1, 128, 128]
        out = infer(gen, loader)

        # 참조용 원본 문자 이미지들 가져오기
        refs = self.get_charimages(target_fonts, target_chars)

        # 비교할 배치들 구성 (참조 + 생성된 이미지)
        compare_batches = [refs, out]
        if compare_inputs:
            compare_batches += self.get_inputimages(loader)  # 입력 이미지도 추가

        nrow = len(target_chars)  # 그리드의 열 수
        comparable_grid = utils.make_comparable_grid(*compare_batches, nrow=nrow)

        return comparable_grid

    # region - cross valid
    @torch_eval
    def cross_validation(self, gen, step, loader, tag, n_batches, n_log=64, save_dir=None):
        """
        분할된 교차 검증 세트를 사용한 검증
        
        정량적 메트릭(L1, SSIM, MS-SSIM)을 계산하고 시각적 결과를 로깅합니다.
        
        Args:
            gen: 생성자 모델
            step: 현재 훈련 스텝
            loader: 검증 데이터 로더
            tag: 로깅 태그
            n_batches: 평가할 배치 수
            n_log: 로깅할 이미지 수
            save_dir: 이미지 저장 디렉토리 (옵션)
            
        Returns:
            tuple: (L1 평균, SSIM 평균, MS-SSIM 평균)
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        outs = []  # 생성된 이미지들
        trgs = []  # 타겟 이미지들
        n_accum = 0  # 누적된 이미지 수

        # 평가 메트릭들을 위한 평균 미터
        losses = utils.AverageMeters("l1", "ssim", "msssim")
        
        # 전체 한글 문자 리스트 (주석 처리된 코드에서 사용)
        hangul_chars = get_all_hangul_chars()
        
        for i, (style_ids, style_comp_ids, style_imgs,
                trg_ids, trg_comp_ids, content_imgs, trg_imgs) in enumerate(loader):
            if i == n_batches:
                break

            # 데이터를 GPU로 이동
            style_ids = style_ids.cuda()
            style_comp_ids = style_comp_ids.cuda()
            style_imgs = style_imgs.cuda()
            trg_ids = trg_ids.cuda()
            trg_comp_ids = trg_comp_ids.cuda()
            trg_imgs = trg_imgs.cuda()

            # 생성자의 2단계 과정
            # 1단계: 스타일 인코딩 및 메모리에 저장
            gen.encode_write(style_ids, style_comp_ids, style_imgs)
            
            # # --------------------------- 추가된 코드 (주석 처리됨) ---------------------------
            # # 모든 한글 문자에 대해 생성을 시도하는 코드
            # # 이 부분은 특정 폰트에 대해 전체 한글을 생성할 때 사용
            # for idx, char in enumerate(hangul_chars):
            #     comps = kor.decompose(char)  # 문자를 자소로 분해
            #     comps = torch.tensor(comps, device='cuda')
            #     comps = comps.unsqueeze(0)
            #
            #     try:
            #         gen_out = gen.read_decode(trg_ids[0].unsqueeze(0), comps)
            #     except KeyError:
            #         continue  # 해당 자소 조합이 없으면 건너뛰기
            # # ----------------------------------------------------------------------
            
            # 2단계: 타겟 문자 생성
            out = gen.read_decode(trg_ids, trg_comp_ids)
            B = len(out)

            # 로깅용 이미지 수집
            if n_accum < n_log:
                trgs.append(trg_imgs)
                outs.append(out)
                n_accum += B

                if n_accum >= n_log:
                    # 결과 로깅
                    outs = torch.cat(outs)[:n_log]
                    trgs = torch.cat(trgs)[:n_log]
                    self.merge_and_log_image(tag, outs, trgs, step)

            # 픽셀 레벨 손실 계산
            l1, ssim, msssim = self.get_pixel_losses(out, trg_imgs, self.unify_resize_method)
            losses.updates({
                "l1": l1.item(),
                "ssim": ssim.item(),
                "msssim": msssim.item()
            }, B)

            # 이미지 저장 (옵션)
            if save_dir:
                font_ids = trg_ids.detach().cpu().numpy()
                images = out.detach().cpu()  # [B, 1, 128, 128]
                char_comp_ids = trg_comp_ids.detach().cpu().numpy()  # [B, n_comp_types]
                
                for font_id, image, comp_ids in zip(font_ids, images, char_comp_ids):
                    font_name = loader.dataset.fonts[font_id]  # 폰트명.ttf
                    font_name = Path(font_name).stem  # 확장자 제거
                    (save_dir / font_name).mkdir(parents=True, exist_ok=True)
                    
                    # 언어별 문자 재구성
                    if self.language == 'kor':
                        char = kor.compose(*comp_ids)  # 자소 조합으로 문자 재구성
                    elif self.language == 'thai':
                        char = thai.compose_ids(*comp_ids)

                    # 유니코드 문자열로 파일명 생성
                    uni = "".join([f'{ord(each):04X}' for each in char])
                    path = save_dir / font_name / "{}_{}.png".format(font_name, uni)
                    utils.save_tensor_to_image(image, path)

        # 검증 결과 로깅
        self.logger.info(
            "  [Valid] {tag:30s} | Step {step:7d}  L1 {L.l1.avg:7.4f}  SSIM {L.ssim.avg:7.4f}"
            "  MSSSIM {L.msssim.avg:7.4f}"
            .format(tag=tag, step=step, L=losses))

        return losses.l1.avg, losses.ssim.avg, losses.msssim.avg

    def get_pixel_losses(self, out, trg_imgs, unify):
        """
        픽셀 레벨 손실 메트릭들 계산
        
        Args:
            out: 생성된 이미지들
            trg_imgs: 타겟 GT(Ground Truth) 이미지들
            unify: 공정한 평가를 위한 글리프 크기 및 리사이즈 방법 통일 여부
                   논문에서 사용된 공정한 평가 설정
            
        Returns:
            tuple: (L1 손실, SSIM, MS-SSIM)
        """
        def unify_resize_method(img):
            """
            공정한 평가를 위해 다양한 글리프 크기와 리사이즈 방법을 통일
            
            Args:
                img: 입력 이미지
                
            Returns:
                torch.Tensor: 통일된 크기의 이미지
            """
            size = img.size(-1)
            if size == 128:
                # 128x128 → 64x64 → normalize → 128x128로 bicubic 업스케일링
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize([64, 64]),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])
                img = torch.stack([transform(_img) for _img in img.cpu()]).cuda()

            # 2배 업스케일링 (bicubic interpolation 사용)
            img = F.interpolate(img, scale_factor=2.0, mode='bicubic', align_corners=True)
            return img

        # 크기 통일이 요청된 경우 적용
        if unify:
            out = unify_resize_method(out)
            trg_imgs = unify_resize_method(trg_imgs)

        # 세 가지 메트릭 계산
        l1 = F.l1_loss(out, trg_imgs)        # L1 거리 (픽셀별 절대값 차이)
        ssim = self.SSIM(out, trg_imgs)      # 구조적 유사도
        msssim = self.MSSSIM(out, trg_imgs)  # 다중 스케일 구조적 유사도

        return l1, ssim, msssim

    @torch_eval
    def handwritten_validation_2stage(self, gen, step, fonts, style_chars, target_chars,
                                      comparable=False, save_dir=None, tag='hw_validation_2stage'):
        """
        2단계 필기체 검증
        
        1단계: 스타일 문자들로 인코딩
        2단계: 타겟 문자들로 디코딩
        
        이 방법은 실제 사용 시나리오를 시뮬레이션합니다.
        
        Args:
            gen: 생성자 모델
            step: 현재 훈련 스텝
            fonts: 폰트명 리스트
            style_chars: 스타일 참조용 문자들
            target_chars: 생성할 타겟 문자들
            comparable: 비교 그리드 생성 여부
            save_dir: 저장 디렉토리 (None이면 그리드만 생성)
            tag: 로깅 태그
        """
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        outs = []
        
        # 각 폰트에 대해 2단계 추론 수행
        for font_name in tqdm(fonts):
            # 1단계: 인코딩 로더 - 스타일 문자들로 폰트 스타일 학습
            encode_loader = get_val_encode_loader(
                self.data, font_name, style_chars, self.language, self.transform
            )
            # 2단계: 디코딩 로더 - 타겟 문자들 생성
            decode_loader = get_val_decode_loader(target_chars, self.language)
            
            # 2단계 추론 실행
            out = infer_2stage(gen, encode_loader, decode_loader)
            outs.append(out)

            # 개별 이미지 저장 (요청된 경우)
            if save_dir:
                for char, glyph in zip(target_chars, out):
                    # 유니코드 문자열로 파일명 생성
                    uni = "".join([f'{ord(each):04X}' for each in char])
                    path = save_dir / font_name / "{}_{}.png".format(font_name, uni)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    utils.save_tensor_to_image(glyph, path)

        if save_dir:  # 저장 모드인 경우 그리드 생성 생략
            return

        # 결과들을 하나의 텐서로 결합
        out = torch.cat(outs)
        
        if comparable:
            # 참조용 원본 문자 이미지들 가져오기
            refs = self.get_charimages(fonts, target_chars)
            nrow = len(target_chars)
            grid = utils.make_comparable_grid(refs, out, nrow=nrow)
        else:
            # 단순 그리드 생성
            grid = utils.to_grid(out, 'torch', nrow=len(target_chars))

        # 태그에 처음 4개 문자 추가
        tag = tag + ''.join(target_chars[:4])
        self.writer.add_image(tag, grid, global_step=step)

    def get_inputimages(self, val_loader):
        """
        검증 로더에서 입력 이미지들을 통합하여 반환
        
        스타일 이미지들을 컴포넌트 타입별로 분리하여 반환합니다.
        
        Args:
            val_loader: 검증 데이터 로더
            
        Returns:
            list: 컴포넌트 타입별로 분리된 배치들
        """
        inputs = []
        
        # 모든 배치에서 스타일 이미지들 수집
        for style_ids, style_comp_ids, style_imgs, trg_ids, trg_comp_ids, content_imgs \
                in val_loader:
            inputs.append(style_imgs)

        inputs = torch.cat(inputs)
        shape = inputs.shape
        
        # [B*n_comp_types, 1, H, W] → [B, n_comp_types, 1, H, W]로 재구성
        inputs = inputs.view(shape[0]//self.n_comp_types, self.n_comp_types, *shape[1:])
        
        # 각 컴포넌트 타입별로 분리
        batches = [inputs[:, i] for i in range(self.n_comp_types)]

        return batches

    def get_charimages(self, fonts, chars, empty_header=False, as_tensor=True):
        """
        self.data에서 문자 이미지들을 가져오는 함수
        
        Args:
            fonts: 폰트명 리스트
            chars: 문자 리스트
            empty_header: 빈 헤더 추가 여부
            as_tensor: 텐서로 반환할지 여부
            
        Returns:
            2차원 리스트 또는 5차원 텐서:
            리스트: [
                [charimage1, charimage2, ...] (font1),
                ...
            ]
            또는 텐서: [n_fonts, n_chars, 1, 128, 128]
        """
        empty_box = torch.ones(1, 128, 128)  # 빈 이미지 (흰색)
        
        # 각 폰트별로 각 문자의 이미지 수집
        charimages = [
            [self.data.get(font_name, char, empty_box) for char in chars]
            for font_name in fonts
        ]

        # 빈 헤더 추가 (요청된 경우)
        if empty_header:
            header = [empty_box for _ in chars]
            charimages.insert(0, header)

        # 텐서로 변환 (요청된 경우)
        if as_tensor:
            charimages = torch.stack(list(chain.from_iterable(charimages)))

        return charimages

    def merge_and_log_image(self, name, out, target, step):
        """
        출력과 타겟을 2열 그리드로 병합하고 로깅
        
        생성된 이미지와 실제 이미지를 나란히 배치하여 시각적 비교가 가능하도록 합니다.
        
        Args:
            name: 로그 이름
            out: 생성된 이미지들
            target: 타겟 이미지들
            step: 현재 스텝
        """
        merge = utils.make_merged_grid([out, target], merge_dim=2)
        self.writer.add_image(name, merge, global_step=step)


# region - EVAL
def eval_ckpt():
    """
    체크포인트 평가 메인 함수
    
    훈련된 모델을 로드하여 다양한 평가 모드로 평가를 수행합니다:
    - eval: 기본 검증 및 정량적 평가
    - cv-save: 교차 검증 결과 저장
    - user-study: 사용자 연구용 샘플 생성
    - user-study-save: 사용자 연구용 전체 결과 저장
    """
    from train import (
        setup_language_dependent, setup_data, setup_cv_dset_loader,
        get_dset_loader
    )

    logger = Logger.get()

    # 명령행 인수 파싱
    parser = argparse.ArgumentParser('DMFont-eval')
    parser.add_argument(
        "--name", 
        help="사용자 연구 생성 결과의 디렉토리명으로 사용될 이름",
        default='JinbeopUnhae'
    )
    parser.add_argument(
        "--resume", 
        help="로드할 체크포인트 경로",
        default='/home/dev/dmfont/experiments/checkpoints/250831_05-11-52_test/last.pth'
    )
    parser.add_argument(
        "--img_dir", 
        help="이미지 저장 디렉토리", 
        default='./results'
    )
    parser.add_argument(
        "--config_paths", 
        nargs="+", 
        help="설정 파일 경로들",
        default=['cfgs/kor.yaml']
    )
    parser.add_argument(
        "--show", 
        action="store_true", 
        default=False,
        help="설정만 표시하고 종료"
    )
    parser.add_argument(
        "--mode", 
        default="user-study",
        help="평가 모드 선택: "
             "eval (기본값) - 비교 가능한 그리드 생성 및 픽셀 레벨 CV 점수 계산, "
             "cv-save - CV의 모든 타겟 문자 생성 및 저장, "
             "user-study - 랜덤 샘플링된 타겟 문자들에 대한 비교 가능한 그리드 생성, "
             "user-study-save - 사용자 연구의 모든 타겟 문자 생성 및 저장"
    )
    parser.add_argument(
        "--deterministic", 
        default=False, 
        action="store_true",
        help="결정적(deterministic) 평가 모드 활성화"
    )
    parser.add_argument(
        "--debug", 
        default=False, 
        action="store_true",
        help="디버그 모드 (배치 수 제한)"
    )
    args, left_argv = parser.parse_known_args()

    # 설정 파일 로드 및 명령행 인수로 업데이트
    cfg = Config(*args.config_paths)
    cfg.argv_update(left_argv)

    # CUDA 벤치마크 모드 활성화 (성능 최적화)
    torch.backends.cudnn.benchmark = True

    # 데이터 디렉토리 경로 설정
    cfg['data_dir'] = Path(cfg['data_dir'])

    # 설정만 표시하고 종료하는 모드
    if args.show:
        exit()

    # 시드 설정 (재현 가능한 결과를 위해)
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    random.seed(cfg['seed'])

    # 결정적 평가 모드 설정
    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        cfg['n_workers'] = 0  # 멀티프로세싱 비활성화
        logger.info("#" * 80)
        logger.info("# 결정적 옵션이 활성화되었습니다!")
        logger.info("# 결정적 평가기는 결정적 교차 검증만 보장합니다")
        logger.info("#" * 80)
    else:
        torch.backends.cudnn.benchmark = True

    # 스타일 믹싱 모드 검증 (해당하는 경우)
    if args.mode.startswith('mix'):
        assert cfg['g_args']['style_enc']['use'], \
                "스타일 믹싱은 스타일 인코더 모델에서만 사용 가능합니다"

    #####################################
    # 데이터셋 설정
    ####################################
    
    # 언어 의존적 값들 설정
    content_font, n_comp_types, n_comps = setup_language_dependent(cfg)

    # 이미지 전처리 변환 설정
    transform = transforms.Compose([
        transforms.ToTensor(),              # PIL 이미지 → 텐서
        transforms.Normalize([0.5], [0.5])  # [0,1] → [-1,1] 정규화
    ])

    # HDF5 데이터와 메타데이터 설정
    hdf5_data, meta = setup_data(cfg, transform)

    # 훈련 데이터셋 및 로더 설정
    trn_dset, loader = get_dset_loader(
        hdf5_data, meta['train']['fonts'], meta['train']['chars'], transform, True, cfg,
        content_font=content_font
    )

    # 교차 검증용 데이터 로더들 설정
    val_loaders = setup_cv_dset_loader(
        hdf5_data, meta, transform, n_comp_types, content_font, cfg
    )

    #####################################
    # 모델 설정
    ####################################
    
    # 생성자만 설정 (평가 시에는 판별자 불필요)
    g_kwargs = cfg.get('g_args', {})
    gen = MACore(
        1, cfg['C'], 1, **g_kwargs, 
        n_comps=n_comps, 
        n_comp_types=n_comp_types,
        language=cfg['language']
    )
    gen.cuda()

    # 체크포인트 로드
    ckpt = torch.load(args.resume)
    logger.info("기본값으로 EMA 생성자 사용")
    gen.load_state_dict(ckpt['generator_ema'])  # EMA 가중치 로드

    step = ckpt['epoch']  # 실제로는 스텝 번호
    loss = ckpt['loss']   # 저장된 손실값

    logger.info("체크포인트가 {}에서 복원되었습니다 (Step {}, Loss {:7.3f})".format(
        args.resume, step, loss))

    # 디스크 기반 writer 설정 (이미지 저장용)
    writer = utils.DiskWriter(args.img_dir, 0.6)

    # 평가기 설정
    evaluator = Evaluator(
        hdf5_data, trn_dset.avails, logger, writer, cfg['batch_size'],
        content_font=content_font, transform=transform, language=cfg['language'],
        val_loaders=val_loaders, meta=meta
    )
    
    # 전체 데이터 평가를 위해 배치 수 제한 해제
    evaluator.n_cv_batches = -1
    logger.info("전체 데이터 평가를 위해 n_cv_batches = -1로 업데이트")
    
    # 디버그 모드인 경우 배치 수 제한
    if args.debug:
        evaluator.n_cv_batches = 10
        logger.info("!!! 디버그 모드: n_cv_batches = 10 !!!")

    #####################################
    # 평가 모드별 실행
    #####################################
    
    if args.mode == 'eval':
        """
        기본 평가 모드
        - 정량적 메트릭 계산 (L1, SSIM, MS-SSIM)
        - 비교 가능한 이미지 그리드 생성
        - TensorBoard에 결과 로깅
        """
        logger.info("검증을 시작합니다...")
        dic = evaluator.validation(gen, step)
        logger.info("검증이 완료되었습니다. 결과 이미지가 {}에 저장되었습니다".format(args.img_dir))
    
    # region user-study
    elif args.mode.startswith('user-study'):
        """
        사용자 연구 모드
        - 실제 필기체 폰트에 대한 평가
        - 2단계 추론 사용 (인코딩 → 디코딩)
        """
        # 한국어 비정제 메타데이터 로드
        meta = json.load(open('meta/kor-unrefined.json'))
        target_chars = meta['target_chars']  # 생성할 타겟 문자들
        style_chars = meta['style_chars']    # 스타일 참조용 문자들
        fonts = meta['fonts']                # 평가할 폰트들

        if args.mode == 'user-study':
            """
            사용자 연구 - 샘플 생성 모드
            - 타겟 문자 중 20개만 샘플링하여 생성
            - 비교 가능한 그리드 형태로 결과 표시
            """
            sampled_target_chars = uniform_sample(target_chars, 20)
            logger.info("한국어 비정제 생성을 시작합니다...")
            logger.info("샘플링된 문자들 = {}".format(sampled_target_chars))

            evaluator.handwritten_validation_2stage(
                gen, step, fonts, style_chars, sampled_target_chars,
                comparable=True, tag='userstudy-{}'.format(args.name)
            )
            
        elif args.mode == 'user-study-save':
            """
            사용자 연구 - 전체 저장 모드
            - 모든 타겟 문자에 대해 생성
            - 개별 이미지 파일로 저장
            """
            logger.info("한국어 비정제 생성 및 저장을 시작합니다...")
            save_dir = Path(args.img_dir) / "{}-{}".format(args.name, step)
            evaluator.handwritten_validation_2stage(
                gen, step, fonts, style_chars, target_chars,
                comparable=True, save_dir=save_dir
            )
        logger.info("검증이 완료되었습니다. 결과 이미지가 {}에 저장되었습니다".format(args.img_dir))
        
    elif args.mode == 'cv-save':
        """
        교차 검증 저장 모드
        - 모든 검증 로더의 결과를 생성
        - 개별 이미지 파일로 저장
        - 정량적 메트릭도 함께 계산
        """
        save_dir = Path(args.img_dir) / "cv_images_{}".format(step)
        logger.info("CV 결과를 {}에 저장합니다...".format(save_dir))
        utils.rm(save_dir)  # 기존 디렉토리 삭제
        
        # 각 검증 로더별로 결과 생성 및 저장
        for tag, loader in val_loaders.items():
            l1, ssim, msssim = evaluator.cross_validation(
                gen, step, loader, tag, n_batches=evaluator.n_cv_batches, 
                save_dir=(save_dir / tag)
            )
    else:
        raise ValueError(f"지원되지 않는 모드: {args.mode}")


if __name__ == "__main__":
    eval_ckpt()
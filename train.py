"""
DMFont
Copyright (c) 2020-present NAVER Corp.
MIT license

DMFont 모델의 훈련 스크립트
Multi-Content Font Style Transfer를 위한 GAN 기반 모델 학습
"""
import sys
import json
from pathlib import Path
import argparse
import random

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
import numpy as np
from sconf import Config, dump_args

from logger import Logger
from models import MACore, Discriminator, AuxClassifier
from models.modules import weights_init
from datasets import HDF5Data, get_ma_dataset, get_ma_val_dataset
import datasets.kor_decompose as kor
import datasets.thai_decompose as thai
import utils
from trainer import Trainer, load_checkpoint
from evaluator import Evaluator


# region - dataset loader
def get_dset_loader(data, avail_fonts, avail_chars, transform, shuffle, cfg, content_font=None):
    """
    훈련용 데이터셋 로더를 생성하는 함수
    
    Parameters
    ----------
    data : HDF5Data
        HDF5 형식의 폰트 데이터
    avail_fonts : list
        사용 가능한 폰트 목록
    avail_chars : list
        사용 가능한 문자 목록
    transform : torchvision.transforms
        이미지 전처리 변환
    shuffle : bool
        데이터 셔플 여부
    cfg : Config
        설정 객체
    content_font : str, optional
        컨텐츠 폰트명 (기본 스타일 참조용)
    
    Returns
    -------
    tuple
        (dataset, dataloader) 튜플
    """
    # 다중 컨텐츠 데이터셋 생성
    dset, collate_fn = get_ma_dataset(
        data,
        avail_fonts,
        avail_chars=avail_chars,
        transform=transform,
        **cfg.get('dset_args', {}),  # 데이터셋 추가 인자들
        content_font=content_font,
        language=cfg['language']
    )
    
    # 데이터 로더 생성
    loader = DataLoader(dset, batch_size=cfg['batch_size'], shuffle=shuffle,
                        num_workers=cfg['n_workers'], collate_fn=collate_fn)

    return dset, loader


def get_val_dset_loader(data, avail_fonts, avail_chars, trn_avail_chars, transform,
                        batch_size, n_workers=2, n_max_match=3, content_font=None, language=None):
    """
    검증용 데이터셋 로더를 생성하는 함수
    
    Parameters
    ----------
    data : HDF5Data
        HDF5 형식의 폰트 데이터
    avail_fonts : list
        검증에 사용할 폰트 목록
    avail_chars : list
        검증에 사용할 문자 목록
    trn_avail_chars : list
        훈련에 사용된 문자 목록 (스타일 참조용)
    transform : torchvision.transforms
        이미지 전처리 변환
    batch_size : int
        배치 크기
    n_workers : int
        데이터 로딩 워커 수
    n_max_match : int
        최대 매칭 수 (스타일 참조 이미지 개수)
    content_font : str, optional
        컨텐츠 폰트명
    language : str, optional
        언어 설정
    
    Returns
    -------
    tuple
        (dataset, dataloader) 튜플
    """
    # 스타일 폰트별로 사용 가능한 문자 매핑
    # 모든 스타일 폰트에서 훈련 문자들을 사용할 수 있다고 가정
    style_avails = {
        font_name: trn_avail_chars for font_name in avail_fonts
    }
    
    # 검증용 다중 컨텐츠 데이터셋 생성
    dset, collate_fn = get_ma_val_dataset(
        data,
        avail_fonts,
        avail_chars,
        style_avails,
        n_max_match=n_max_match,    # 스타일 참조 이미지 최대 개수
        transform=transform,
        ret_targets=True,           # 타겟 이미지도 반환
        first_shuffle=True,         # 첫 번째 에포크에서 셔플
        content_font=content_font,
        language=language
    )
    
    # 검증용 데이터 로더 (셔플하지 않음)
    loader = DataLoader(dset, batch_size=batch_size, shuffle=False,
                        num_workers=n_workers, collate_fn=collate_fn)

    return dset, loader


def setup_args_and_config():
    """
    명령줄 인자와 설정을 파싱하고 설정하는 함수
    
    Returns
    -------
    tuple
        (args, config) 파싱된 인자와 설정 객체
    """
    parser = argparse.ArgumentParser('MaHFG')
    parser.add_argument("--name", default='test', help="실험 이름")
    parser.add_argument("--config_paths", nargs='+', default=['cfgs/kor.yaml'], 
                        help="설정 파일 경로들")
    parser.add_argument("--show", action="store_true", default=False, 
                        help="설정만 출력하고 종료")
    parser.add_argument("--resume", default=None, help="재개할 체크포인트 경로")
    parser.add_argument("--log_lv", default='info', help="로그 레벨")
    parser.add_argument("--debug", default=False, action="store_true", 
                        help="디버그 모드 활성화")
    parser.add_argument("--tb-image", default=False, action="store_true",
                        help="텐서보드에 이미지 로그 작성")
    parser.add_argument("--deterministic", default=False, action="store_true",
                        help="결정론적 실행 (재현가능한 결과)")

    # 알려진 인자와 남은 인자를 분리
    args, left_argv = parser.parse_known_args()
    assert not args.name.endswith(".yaml")

    # 설정 파일 로드 및 명령줄 인자로 업데이트
    cfg = Config(*args.config_paths, colorize_modified_item=True)
    cfg.argv_update(left_argv)

    # 디버그 모드 설정
    if args.debug:
        cfg['print_freq'] = 1       # 매 스텝마다 출력
        cfg['tb_freq'] = 1          # 매 스텝마다 텐서보드 로그
        cfg['max_iter'] = 10        # 최대 10 이터레이션만 실행
        cfg['val_freq'] = 5         # 5 스텝마다 검증
        cfg['save_freq'] = 10       # 10 스텝마다 저장
        args.name += "_debug"       # 실험 이름에 디버그 표시
        args.tb_image = True        # 이미지 로그 활성화
        args.log_lv = 'debug'       # 디버그 로그 레벨

    # 데이터 디렉토리를 Path 객체로 변환
    cfg['data_dir'] = Path(cfg['data_dir'])

    # 저장 빈도가 검증 빈도의 배수인지 확인
    assert cfg['save_freq'] % cfg['val_freq'] == 0

    return args, cfg


def setup_language_dependent(cfg):
    """
    언어별 종속적인 설정을 초기화하는 함수
    
    Parameters
    ----------
    cfg : Config
        설정 객체
    
    Returns
    -------
    tuple
        (content_font, n_comp_types, n_comps)
        - content_font: 기본 컨텐츠 폰트명
        - n_comp_types: 구성 요소 타입 수
        - n_comps: 전체 구성 요소 수
    """
    if cfg['language'] == 'kor':
        # 한국어 설정
        # content_font = "JinbeopUnhae.ttf"  # 진법언해 폰트 (주석처리됨)
        content_font = "NanumBarunpenR.ttf"   # 나눔바른펜 폰트 사용
        n_comp_types = 3  # 초성, 중성, 종성
        n_comps = kor.N_COMPONENTS  # 한글 구성 요소 총 개수
    elif cfg['language'] == 'thai':
        # 태국어 설정
        content_font = "NotoSansThai-Regular.ttf"  # 노토 산스 태국어 폰트
        n_comp_types = 4  # 자음, 상단, 최상단, 하단
        n_comps = thai.N_COMPONENTS  # 태국어 구성 요소 총 개수
    else:
        raise ValueError(cfg['language'])

    return content_font, n_comp_types, n_comps


def setup_data(cfg, val_transform):
    """
    데이터와 메타데이터를 설정하는 함수
    
    Parameters
    ----------
    cfg : Config
        설정 객체
    val_transform : torchvision.transforms
        검증용 이미지 변환
    
    Returns
    -------
    tuple
        (hdf5_data, meta_data)
        - hdf5_data: HDF5 데이터 객체
        - meta_data: 메타데이터 딕셔너리
    """
    # 데이터 디렉토리에서 모든 HDF5 파일 찾기
    hdf5_paths = list(cfg['data_dir'].glob("*.hdf5"))
    # HDF5 데이터 객체 생성
    hdf5_data = HDF5Data(hdf5_paths, val_transform, language=cfg['language'])

    # 메타데이터 로드 (훈련/검증 폰트와 문자 분할 정보)
    meta = json.load(open(cfg['data_meta']))

    return hdf5_data, meta


def setup_cv_dset_loader(hdf5_data, meta, val_transform, n_comp_types, content_font, cfg):
    """
    교차 검증용 데이터셋 로더들을 설정하는 함수
    
    다양한 조합의 검증 시나리오를 테스트:
    - 보인 폰트 + 안 보인 문자
    - 안 보인 폰트 + 보인 문자  
    - 안 보인 폰트 + 안 보인 문자
    
    Parameters
    ----------
    hdf5_data : HDF5Data
        HDF5 데이터 객체
    meta : dict
        메타데이터
    val_transform : torchvision.transforms
        검증용 이미지 변환
    n_comp_types : int
        구성 요소 타입 수
    content_font : str
        컨텐츠 폰트명
    cfg : Config
        설정 객체
    
    Returns
    -------
    dict
        검증 로더들의 딕셔너리
    """
    trn_chars = meta['train']['chars']  # 훈련용 문자들
    val_chars = meta['valid']['chars']  # 검증용 문자들
    
    # 검증용 배치 크기 (일반적으로 훈련보다 크게 설정)
    batch_size = cfg['batch_size'] * 3
    n_workers = cfg['n_workers']
    n_max_match = n_comp_types  # 검증 데이터셋용 최대 매칭 수

    # 주석처리된 검증 시나리오들:
    # 1. 보인 폰트, 안 보인 문자 (seen fonts, unseen chars)
    # sfuc_dset, sfuc_loader = get_val_dset_loader(
    #     hdf5_data, meta['train']['fonts'], meta['valid']['chars'], trn_chars, val_transform,
    #     batch_size, n_workers, n_max_match, content_font, cfg['language']
    # )
    
    # 2. 안 보인 폰트, 보인 문자 (unseen fonts, seen chars)
    # ufsc_dset, ufsc_loader = get_val_dset_loader(
    #     hdf5_data, meta['valid']['fonts'], meta['train']['chars'], trn_chars, val_transform,
    #     batch_size, n_workers, n_max_match, content_font, cfg['language']
    # )
    
    # 검증용 문자에서 예외 문자들 제거 (한글 자모 단독 문자들)
    # 이는 일반적으로 단독으로는 사용되지 않는 한글 자모들
    exception_str = ["\u3131", "\u3132", "\u3134", "\u3137", "\u3138", "\u3139", "\u3141", "\u3142", 
                     "\u3143", "\u3145", "\u3146", "\u3147", "\u3148", "\u3149", "\u314a", "\u314b", 
                     "\u314c", "\u314d", "\u314e", "\u314f", "\u3150", "\u3151", "\u3152", "\u3153", 
                     "\u3154", "\u3155", "\u3156", "\u3157", "\u3158", "\u3159", "\u315a", "\u315b", 
                     "\u315c", "\u315d", "\u315e", "\u315f", "\u3160", "\u3161", "\u3162", "\u3163"]
    
    # 예외 문자들을 제외한 검증 문자 목록 생성
    meta['valid']['chars'] = [char for char in meta['valid']['chars'] if not char in exception_str]
    val_chars = meta['valid']['chars']
    
    # 3. 안 보인 폰트, 안 보인 문자 (가장 어려운 시나리오)
    ufuc_dset, ufuc_loader = get_val_dset_loader(
        hdf5_data, meta['valid']['fonts'], meta['valid']['chars'], val_chars, val_transform,
        batch_size, n_workers, n_max_match, content_font, cfg['language']
    )
    
    # 검증 로더들을 딕셔너리로 구성
    val_loaders = {
        # "SeenFonts-UnseenChars": sfuc_loader,      # 주석처리됨
        # "UnseenFonts-SeenChars": ufsc_loader,      # 주석처리됨
        "UnseenFonts-UnseenChars": ufuc_loader       # 현재 사용하는 검증 시나리오
    }

    return val_loaders


def main():
    """
    메인 훈련 함수
    
    전체 훈련 파이프라인을 실행:
    1. 인자 및 설정 파싱
    2. 로깅 및 실험 디렉토리 설정
    3. 데이터셋 및 데이터 로더 설정
    4. 모델 구축 및 초기화
    5. 옵티마이저 설정
    6. 체크포인트 복원 (선택적)
    7. 평가자 설정
    8. 훈련 시작
    """
    ############################
    # 인자 및 설정 파싱
    ############################
    args, cfg = setup_args_and_config()

    # 설정만 출력하고 종료하는 옵션
    if args.show:
        print("### Run Argv:\n> {}".format(' '.join(sys.argv)))
        print("### Run Arguments:")
        s = dump_args(args)
        print(s + '\n')
        print("### Configs:")
        print(cfg.dumps())
        sys.exit()

    # 고유한 실험 이름 생성 (타임스탬프 + 사용자 지정 이름)
    timestamp = utils.timestamp()
    unique_name = "{}_{}".format(timestamp, args.name)
    cfg['unique_name'] = unique_name  # 저장 디렉토리용
    cfg['name'] = args.name

    # 실험 디렉토리들 생성
    utils.makedirs('experiments/logs')
    utils.makedirs(Path('experiments/checkpoints', unique_name))

    # 로거 설정
    logger_path = Path('experiments/logs', f"{unique_name}.log")
    logger = Logger.get(file_path=logger_path, level=args.log_lv, colorize=True)

    # 텐서보드 라이터 설정
    image_scale = 0.6  # 이미지 스케일링
    writer_path = Path('experiments/runs', unique_name)
    if args.tb_image:
        # 텐서보드에 직접 이미지 쓰기
        writer = utils.TBWriter(writer_path, scale=image_scale)
    else:
        # 디스크에 이미지 저장하고 텐서보드에는 스칼라만
        image_path = Path('experiments/images', unique_name)
        writer = utils.TBDiskWriter(writer_path, image_path, scale=image_scale)

    # 기본 정보 로깅
    args_str = dump_args(args)
    logger.info("Run Argv:\n> {}".format(' '.join(sys.argv)))
    logger.info("Args:\n{}".format(args_str))
    logger.info("Configs:\n{}".format(cfg.dumps()))
    logger.info("Unique name: {}".format(unique_name))

    # 시드 설정 (재현 가능한 결과를 위해)
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    random.seed(cfg['seed'])

    # 결정론적 실행 모드
    if args.deterministic:
        # 완전히 재현 가능한 결과를 위한 설정
        # 성능은 다소 떨어질 수 있음
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        cfg['n_workers'] = 0  # 멀티프로세싱 비활성화
        logger.info("#" * 80)
        logger.info("# Deterministic option is activated !")
        logger.info("#" * 80)
    else:
        # 성능 최적화를 위한 설정
        torch.backends.cudnn.benchmark = True

    ############################
    # 데이터셋 및 로더 설정
    ############################
    logger.info("Get dataset ...")

    # 언어별 종속 설정
    content_font, n_comp_types, n_comps = setup_language_dependent(cfg)

    # 이미지 전처리 변환 설정
    # ToTensor: PIL Image -> Tensor 변환
    # Normalize: [0,1] -> [-1,1] 정규화 (GAN에서 일반적)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # mean=0.5, std=0.5로 [-1,1] 범위
    ])

    # 데이터 및 메타데이터 로드
    hdf5_data, meta = setup_data(cfg, transform)

    # 훈련용 데이터셋 및 로더 생성
    trn_dset, loader = get_dset_loader(
        hdf5_data, meta['train']['fonts'], meta['train']['chars'], transform, True, cfg,
        content_font=content_font
    )

    # 훈련 데이터셋 정보 로깅
    logger.info("### Training dataset ###")
    logger.info("# of avail fonts = {}".format(trn_dset.n_fonts))
    logger.info(f"Total {len(loader)} iterations per epochs")
    logger.info("# of avail items = {}".format(trn_dset.n_avails))
    logger.info(f"#fonts = {trn_dset.n_fonts}, #chars = {trn_dset.n_chars}")

    # 검증용 데이터셋들 설정
    val_loaders = setup_cv_dset_loader(
        hdf5_data, meta, transform, n_comp_types, content_font, cfg
    )
    
    # 현재 사용하는 검증 로더 (안 보인 폰트 + 안 보인 문자)
    ufuc_loader = val_loaders['UnseenFonts-UnseenChars']
    ufuc_dset = ufuc_loader.dataset

    # 검증 데이터셋 정보 로깅
    logger.info("### Cross-validation datasets ###")
    logger.info(
        "Unseen fonts, Unseen chars | "
        "#items = {}, #fonts = {}, #chars = {}, #steps = {}".format(
            len(ufuc_dset), len(ufuc_dset.fonts), len(ufuc_dset.chars), len(ufuc_loader)))

    ############################
    # 모델 구축
    ############################
    logger.info("Build model ...")
    
    # 생성자 (Generator) 모델
    g_kwargs = cfg.get('g_args', {})  # 생성자 추가 인자들
    gen = MACore(
        1, cfg['C'], 1,  # 입력 채널, 특징 채널, 출력 채널
        **g_kwargs, 
        n_comps=n_comps,                # 전체 구성 요소 수
        n_comp_types=n_comp_types,      # 구성 요소 타입 수
        language=cfg['language']
    )
    gen.cuda()  # GPU로 이동
    gen.apply(weights_init(cfg['init']))  # 가중치 초기화

    # 판별자 (Discriminator) 모델
    d_kwargs = cfg.get('d_args', {})  # 판별자 추가 인자들
    disc = Discriminator(cfg['C'], trn_dset.n_fonts, trn_dset.n_chars, **d_kwargs)
    disc.cuda()  # GPU로 이동
    disc.apply(weights_init(cfg['init']))  # 가중치 초기화

    # 보조 분류기 (Auxiliary Classifier) - 선택적
    if cfg['ac_w'] > 0.:
        # 보조 분류기 손실 가중치가 0보다 큰 경우에만 생성
        C = gen.mem_shape[0]  # 생성자의 메모리 특징 차원
        aux_clf = AuxClassifier(C, n_comps, **cfg['ac_args'])
        aux_clf.cuda()
        aux_clf.apply(weights_init(cfg['init']))
    else:
        aux_clf = None
        # 보조 분류기가 없으면 관련 손실도 0이어야 함
        assert cfg['ac_gen_w'] == 0., "ac_gen loss is only available with ac loss"

    # 옵티마이저 설정
    g_optim = optim.Adam(gen.parameters(), lr=cfg['g_lr'], betas=cfg['adam_betas'])
    d_optim = optim.Adam(disc.parameters(), lr=cfg['d_lr'], betas=cfg['adam_betas'])
    # 보조 분류기용 옵티마이저 (있는 경우에만)
    ac_optim = optim.Adam(aux_clf.parameters(), lr=cfg['g_lr'], betas=cfg['adam_betas']) \
               if aux_clf is not None else None

    # 체크포인트 복원
    st_step = 1  # 시작 스텝
    if args.resume:
        # 지정된 체크포인트에서 모델과 옵티마이저 상태 복원
        st_step, loss = load_checkpoint(args.resume, gen, disc, aux_clf, g_optim, d_optim, ac_optim)
        logger.info("Resumed checkpoint from {} (Step {}, Loss {:7.3f})".format(
            args.resume, st_step-1, loss))

    ############################
    # 검증 설정
    ############################
    # 평가자 객체 생성 (검증 및 결과 생성 담당)
    evaluator = Evaluator(
        hdf5_data, trn_dset.avails, logger, writer, cfg['batch_size'],
        content_font=content_font, transform=transform, language=cfg['language'],
        val_loaders=val_loaders, meta=meta
    )
    
    # 디버그 모드에서는 검증 배치 수 제한
    if args.debug:
        evaluator.n_cv_batches = 10
        logger.info("Change CV batches to 10 for debugging")

    ############################
    # 훈련 시작
    ############################
    # 트레이너 객체 생성 및 훈련 시작
    trainer = Trainer(
        gen, disc, g_optim, d_optim, aux_clf, ac_optim,
        writer, logger, evaluator, cfg
    )
    trainer.train(loader, st_step)  # 훈련 실행


if __name__ == "__main__":
    main()
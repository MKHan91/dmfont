"""
DMFont - 딥러닝 기반 폰트 생성 모델의 훈련 코드
Copyright (c) 2020-present NAVER Corp.
MIT license

이 코드는 스타일 폰트에서 타겟 문자로 새로운 폰트 글리프를 생성하는 GAN 기반 모델을 훈련합니다.
"""
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from datasets import cyclize
from models.memory import comp_id_to_addr
from criterions import hinge_g_loss, hinge_d_loss


def has_bn(model):
    """
    모델에 BatchNorm 레이어가 있는지 확인하는 함수
    
    Args:
        model: 검사할 신경망 모델
        
    Returns:
        bool: BatchNorm 레이어가 있으면 True, 없으면 False
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            return True
    return False


def load_checkpoint(path, gen, disc, aux_clf, g_optim, d_optim, ac_optim):
    """
    체크포인트 파일에서 모델과 옵티마이저 상태를 로드하는 함수
    
    Args:
        path: 체크포인트 파일 경로
        gen: 생성자 모델
        disc: 판별자 모델  
        aux_clf: 보조 분류기 모델
        g_optim: 생성자 옵티마이저
        d_optim: 판별자 옵티마이저
        ac_optim: 보조 분류기 옵티마이저
        
    Returns:
        tuple: (시작 에폭, 손실값)
    """
    ckpt = torch.load(path)
    
    # 생성자와 옵티마이저 상태 로드
    gen.load_state_dict(ckpt['generator'])
    g_optim.load_state_dict(ckpt['optimizer'])

    # 판별자가 있으면 로드
    if disc is not None:
        disc.load_state_dict(ckpt['discriminator'])
        d_optim.load_state_dict(ckpt['d_optimizer'])

    # 보조 분류기가 있으면 로드
    if aux_clf is not None:
        aux_clf.load_state_dict(ckpt['aux_clf'])
        ac_optim.load_state_dict(ckpt['ac_optimizer'])

    # NOTE: epoch는 실제로는 step을 의미함
    st_epoch = ckpt['epoch'] + 1
    loss = ckpt['loss']

    return st_epoch, loss


class Trainer:
    """
    DMFont 모델의 훈련을 담당하는 메인 클래스
    
    이 클래스는 다음과 같은 컴포넌트들을 관리합니다:
    - 생성자(Generator): 스타일 이미지에서 타겟 문자 생성
    - 판별자(Discriminator): 실제/가짜 이미지 구분
    - 보조 분류기(Auxiliary Classifier): 문자 컴포넌트 분류
    """
    
    def __init__(self, gen, disc, g_optim, d_optim, aux_clf, ac_optim,
                 writer, logger, evaluator, cfg):
        """
        Trainer 초기화
        
        Args:
            gen: 생성자 모델
            disc: 판별자 모델
            g_optim: 생성자 옵티마이저
            d_optim: 판별자 옵티마이저
            aux_clf: 보조 분류기 모델
            ac_optim: 보조 분류기 옵티마이저
            writer: TensorBoard writer
            logger: 로그 객체
            evaluator: 평가 객체
            cfg: 설정 딕셔너리
        """
        self.gen = gen  # 메인 생성자
        self.gen_ema = copy.deepcopy(self.gen)  # EMA(Exponential Moving Average) 생성자
        self.is_bn_gen = has_bn(self.gen)  # 생성자에 BatchNorm이 있는지 확인
        self.disc = disc  # 판별자
        self.g_optim = g_optim  # 생성자 옵티마이저
        self.d_optim = d_optim  # 판별자 옵티마이저
        self.aux_clf = aux_clf  # 보조 분류기
        self.ac_optim = ac_optim  # 보조 분류기 옵티마이저
        self.writer = writer  # TensorBoard writer
        self.logger = logger  # 로거
        self.evaluator = evaluator  # 평가기
        self.cfg = cfg  # 설정
        self.step = 1  # 현재 스텝
        self.language = cfg['language']  # 언어 설정

        # 손실 딕셔너리들 - 각 컴포넌트별로 분리
        self.g_losses = {}  # 생성자 손실들
        self.d_losses = {}  # 판별자 손실들
        self.ac_losses = {}  # 보조 분류기 손실들

    def clear_losses(self):
        """
        손실 딕셔너리들을 통합하고 초기화하는 함수
        
        Returns:
            dict: 모든 손실값들이 통합된 딕셔너리
        """
        # 생성자 손실들을 통합
        loss_dic = {k: v.item() for k, v in self.g_losses.items()}
        loss_dic['g_total'] = sum(loss_dic.values())
        
        # 판별자 손실들 추가
        loss_dic.update({k: v.item() for k, v in self.d_losses.items()})
        
        # 보조 분류기 손실들 추가
        loss_dic.update({k: v.item() for k, v in self.ac_losses.items()})

        # 손실 딕셔너리들 초기화
        self.g_losses = {}
        self.d_losses = {}
        self.ac_losses = {}

        return loss_dic

    def accum_g(self, decay=0.999):
        """
        생성자의 EMA(Exponential Moving Average) 업데이트
        
        EMA는 모델의 파라미터들의 이동평균을 계산하여 더 안정적인 생성을 돕습니다.
        
        Args:
            decay: EMA 감쇠율 (기본값: 0.999)
        """
        par1 = dict(self.gen_ema.named_parameters())  # EMA 모델 파라미터
        par2 = dict(self.gen.named_parameters())       # 현재 모델 파라미터

        # EMA 업데이트: EMA = decay * EMA + (1-decay) * current
        for k in par1.keys():
            par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

    def sync_g_ema(self, style_ids, style_comp_ids, style_imgs, trg_ids, trg_comp_ids):
        """
        EMA 생성자의 배치 정규화 running stats와 스펙트럴 정규화 업데이트
        
        Args:
            style_ids: 스타일 폰트 ID들
            style_comp_ids: 스타일 문자 컴포넌트 ID들  
            style_imgs: 스타일 이미지들
            trg_ids: 타겟 폰트 ID들
            trg_comp_ids: 타겟 문자 컴포넌트 ID들
        """
        org_train_mode = self.gen_ema.training  # 원래 훈련 모드 저장
        with torch.no_grad():
            self.gen_ema.train()  # 훈련 모드로 설정 (BN stats 업데이트를 위해)
            # 스타일 인코딩 및 메모리 작성
            self.gen_ema.encode_write(style_ids, style_comp_ids, style_imgs)
            # 타겟 문자 생성
            self.gen_ema.read_decode(trg_ids, trg_comp_ids)
        self.gen_ema.train(org_train_mode)  # 원래 모드로 복원

    # region - TRAIN
    def train(self, loader, st_step=1, val=None):
        """
        메인 훈련 루프
        
        Args:
            loader: 데이터 로더
            st_step: 시작 스텝 (기본값: 1)
            val: 검증 데이터 (옵션)
        """
        val = val or {}
        self.gen.train()   # 생성자를 훈련 모드로
        self.disc.train()  # 판별자를 훈련 모드로

        # 통계 추적을 위한 미터들
        losses = utils.AverageMeters("g_total", "pixel", "disc", "gen", "fm", "ac", "ac_gen")
        discs = utils.AverageMeters("real", "fake",
                                    "real_font", "real_char", "fake_font", "fake_char",
                                    "real_acc", "fake_acc", "real_font_acc", "real_char_acc",
                                    "fake_font_acc", "fake_char_acc")
        stats = utils.AverageMeters("B_style", "B_target", "ac_acc", "ac_gen_acc")

        self.step = st_step
        self.clear_losses()
        
        # region - load data
        self.logger.info("Start training ...")
        for (style_ids, style_char_ids, style_comp_ids, style_imgs,
             trg_ids, trg_char_ids, trg_comp_ids, trg_imgs, *content_imgs) in cyclize(loader):
            
            B = trg_imgs.size(0)  # 배치 크기
            stats.updates({
                "B_style": style_imgs.size(0),  # 스타일 배치 크기
                "B_target": B                    # 타겟 배치 크기
            })

            # 데이터를 GPU로 이동
            style_ids = style_ids.cuda()
            style_comp_ids = style_comp_ids.cuda()
            style_imgs = style_imgs.cuda()
            trg_ids = trg_ids.cuda()
            trg_char_ids = trg_char_ids.cuda()
            trg_comp_ids = trg_comp_ids.cuda()
            trg_imgs = trg_imgs.cuda()

            # === 생성자 Forward Pass ===
            # 스타일 이미지를 인코딩하고 메모리에 저장
            comp_feats = self.gen.encode_write(style_ids, style_comp_ids, style_imgs)
            # 메모리에서 읽어와서 타겟 문자 생성
            out = self.gen.read_decode(trg_ids, trg_comp_ids)

            # === 판별자 손실 계산 ===
            # 실제 이미지에 대한 판별자 출력 (특징도 함께 반환)
            real, real_font, real_char, real_feats = self.disc(
                trg_imgs, trg_ids, trg_char_ids, out_feats=True
            )
            # 가짜 이미지에 대한 판별자 출력 (gradient 차단)
            fake, fake_font, fake_char = self.disc(out.detach(), trg_ids, trg_char_ids)
            
            # 판별자 GAN 손실 추가
            self.add_gan_d_loss(real, real_font, real_char, fake, fake_font, fake_char)

            # 판별자 업데이트
            self.d_optim.zero_grad()
            self.d_backward()
            self.d_optim.step()

            # === 생성자 손실 계산 ===
            # 가짜 이미지에 대한 판별자 출력 (특징도 함께, gradient 흐름 허용)
            fake, fake_font, fake_char, fake_feats = self.disc(
                out, trg_ids, trg_char_ids, out_feats=True
            )
            # 생성자 GAN 손실 추가
            self.add_gan_g_loss(real, real_font, real_char, fake, fake_font, fake_char)

            # 특징 매칭 손실 추가 (실제와 가짜 특징 간의 차이)
            self.add_fm_loss(real_feats, fake_feats)

            # 판별자 통계 업데이트
            racc = lambda x: (x > 0.).float().mean().item()  # 실제 정확도
            facc = lambda x: (x < 0.).float().mean().item()  # 가짜 정확도
            discs.updates({
                "real": real.mean().item(),
                "fake": fake.mean().item(),
                "real_font": real_font.mean().item(),
                "real_char": real_char.mean().item(),
                "fake_font": fake_font.mean().item(),
                "fake_char": fake_char.mean().item(),
                'real_acc': racc(real),
                'fake_acc': facc(fake),
                'real_font_acc': racc(real_font),
                'real_char_acc': racc(real_char),
                'fake_font_acc': facc(fake_font),
                'fake_char_acc': facc(fake_char)
            }, B)

            # 픽셀 손실 추가 (L1 loss)
            self.add_pixel_loss(out, trg_imgs)

            # 생성자 옵티마이저 초기화 (보조 분류기 손실이 생성자에 영향을 주기 전에)
            self.g_optim.zero_grad()
            
            # NOTE: 보조 분류기 손실이 생성자에 gradient를 남기므로
            # g_optim.zero_grad()는 보조 분류기 손실 앞에,
            # g_backward()는 보조 분류기 손실 뒤에 위치해야 함
            if self.aux_clf is not None:
                # 보조 분류기 손실 계산 및 통계 업데이트
                self.add_ac_losses_and_update_stats(
                    comp_feats, style_comp_ids, out, trg_comp_ids, stats
                )

                # 보조 분류기 업데이트
                self.ac_optim.zero_grad()
                self.ac_backward(retain_graph=True)  # gradient graph 유지
                self.ac_optim.step()

            # 생성자 업데이트
            self.g_backward()
            self.g_optim.step()

            # 손실 딕셔너리 정리 및 통계 업데이트
            loss_dic = self.clear_losses()
            losses.updates(loss_dic, B)

            # 생성자 EMA 업데이트
            self.accum_g()
            if self.is_bn_gen:
                # BatchNorm이 있는 경우 EMA 동기화
                self.sync_g_ema(style_ids, style_comp_ids, style_imgs, trg_ids, trg_comp_ids)

            # === 로깅 및 평가 ===
            if self.step % self.cfg['tb_freq'] == 0:
                self.plot(losses, discs, stats)  # TensorBoard 플롯

            if self.step % self.cfg['print_freq'] == 0:
                self.log(losses, discs, stats)  # 콘솔 로그
                losses.resets()
                discs.resets()
                stats.resets()

            if self.step % self.cfg['val_freq'] == 0:
                epoch = self.step / len(loader)
                self.logger.info("Validation at Epoch = {:.3f}".format(epoch))
                
                # 생성된 이미지와 실제 이미지 비교 로그
                self.evaluator.merge_and_log_image('d1', out, trg_imgs, self.step)
                # 메인 생성자 평가
                self.evaluator.validation(self.gen, self.step)

                # BatchNorm이 없는 경우에만 여기서 EMA 동기화
                if not self.is_bn_gen:
                    self.sync_g_ema(style_ids, style_comp_ids, style_imgs, trg_ids, trg_comp_ids)
                # EMA 생성자 평가
                self.evaluator.validation(self.gen_ema, self.step, extra_tag='_EMA')

                # 체크포인트 저장 (저장 주기 == 검증 주기)
                self.save(
                    loss_dic['g_total'], self.cfg['save'],
                    self.cfg.get('save_freq', self.cfg['val_freq'])
                )

            # 최대 반복 횟수 도달 시 종료
            if self.step >= self.cfg['max_iter']:
                self.logger.info("Iteration finished.")
                break

            self.step += 1

    def add_pixel_loss(self, out, target):
        """
        픽셀 레벨 L1 손실 추가
        
        생성된 이미지와 실제 이미지 간의 직접적인 차이를 최소화합니다.
        
        Args:
            out: 생성된 이미지
            target: 실제 타겟 이미지
            
        Returns:
            torch.Tensor: 계산된 픽셀 손실
        """
        loss = F.l1_loss(out, target, reduction='mean') * self.cfg['pixel_w']
        self.g_losses['pixel'] = loss
        return loss

    def add_gan_g_loss(self, real, real_font, real_char, fake, fake_font, fake_char):
        """
        생성자 GAN 손실 추가
        
        Args:
            real, real_font, real_char: 실제 이미지에 대한 판별자 출력들
            fake, fake_font, fake_char: 가짜 이미지에 대한 판별자 출력들
            
        Returns:
            torch.Tensor: 계산된 생성자 손실
        """
        if self.cfg['gan_w'] == 0.:
            return 0.

        # Hinge loss 사용: 생성자는 판별자를 속이려고 함
        g_loss = hinge_g_loss(real_font.detach(), fake_font) + \
                 hinge_g_loss(real_char.detach(), fake_char)
        
        # 전체 이미지 판별자가 있는 경우
        if self.disc.use_rx:
            g_loss += hinge_g_loss(real.detach(), fake)
        
        g_loss *= self.cfg['gan_w']  # 가중치 적용
        self.g_losses['gen'] = g_loss
        return g_loss

    def add_gan_d_loss(self, real, real_font, real_char, fake, fake_font, fake_char):
        """
        판별자 GAN 손실 추가
        
        Args:
            real, real_font, real_char: 실제 이미지에 대한 판별자 출력들
            fake, fake_font, fake_char: 가짜 이미지에 대한 판별자 출력들
            
        Returns:
            torch.Tensor: 계산된 판별자 손실
        """
        if self.cfg['gan_w'] == 0.:
            return 0.

        # Hinge loss 사용: 판별자는 실제와 가짜를 구분하려고 함
        d_loss = hinge_d_loss(real_font, fake_font) + \
                 hinge_d_loss(real_char, fake_char)
        
        # 전체 이미지 판별자가 있는 경우
        if self.disc.use_rx:
            d_loss += hinge_d_loss(real, fake)
        
        d_loss *= self.cfg['gan_w']  # 가중치 적용
        self.d_losses['disc'] = d_loss
        return d_loss

    def add_fm_loss(self, real_feats, fake_feats):
        """
        특징 매칭(Feature Matching) 손실 추가
        
        판별자의 중간 특징들을 매칭시켜 더 자세한 특징을 학습하도록 합니다.
        
        Args:
            real_feats: 실제 이미지의 중간 특징들
            fake_feats: 가짜 이미지의 중간 특징들
            
        Returns:
            torch.Tensor: 계산된 특징 매칭 손실
        """
        if self.cfg['fm_w'] == 0.:
            return 0.

        fm_loss = 0.
        # 각 레이어의 특징들에 대해 L1 손실 계산
        for real_f, fake_f in zip(real_feats, fake_feats):
            fm_loss += F.l1_loss(real_f.detach(), fake_f)
        
        fm_loss = fm_loss / len(real_feats) * self.cfg['fm_w']  # 평균 및 가중치 적용
        self.g_losses['fm'] = fm_loss
        return fm_loss

    def add_ac_losses_and_update_stats(self, comp_feats, style_comp_ids, generated,
                                       trg_comp_ids, stats):
        """
        보조 분류기 손실들을 추가하고 통계를 업데이트
        
        두 가지 보조 분류기 손실을 계산합니다:
        1. 스타일 인코더 특징에 대한 분류 손실
        2. 생성된 이미지 인코더 특징에 대한 분류 손실
        
        Args:
            comp_feats: 스타일 컴포넌트 특징들
            style_comp_ids: 스타일 컴포넌트 ID들
            generated: 생성된 이미지
            trg_comp_ids: 타겟 컴포넌트 ID들
            stats: 통계 객체
        """
        # 1. ac(enc(x)) 손실: 스타일 특징에 대한 분류
        loss, acc = self.infer_ac(comp_feats, style_comp_ids)
        self.ac_losses['ac'] = loss * self.cfg['ac_w']
        stats.ac_acc.update(acc, style_comp_ids.numel())

        # 2. ac(enc(fake)) 손실: 생성된 이미지 특징에 대한 분류
        # 인코더가 cheating하는 것을 방지하기 위해 두 번째 인코더를 freeze
        with utils.temporary_freeze(self.gen.component_encoder):
            feats = self.gen.component_encoder(generated)

        gen_comp_feats = feats[-1]  # 마지막 레이어 특징 사용

        loss, acc = self.infer_ac(gen_comp_feats, trg_comp_ids)
        self.ac_losses['ac_gen'] = loss * self.cfg['ac_w']
        stats.ac_gen_acc.update(acc, trg_comp_ids.numel())

    def infer_ac(self, comp_feats, comp_ids):
        """
        보조 분류기 추론 및 손실 계산
        
        Args:
            comp_feats: 컴포넌트 특징들
            comp_ids: 컴포넌트 ID들
            
        Returns:
            tuple: (손실, 정확도)
        """
        # 컴포넌트 ID를 주소로 변환 (언어별 매핑)
        comp_addrs = comp_id_to_addr(comp_ids, self.language)

        # 배치와 시퀀스 차원을 평탄화
        comp_feats_flat = comp_feats.flatten(0, 1)
        comp_addrs_flat = comp_addrs.flatten(0, 1)

        # 보조 분류기로 예측
        aux_out = self.aux_clf(comp_feats_flat)
        # 교차 엔트로피 손실 계산
        loss = F.cross_entropy(aux_out, comp_addrs_flat)
        # 정확도 계산
        acc = utils.accuracy(aux_out, comp_addrs_flat)

        return loss, acc

    def d_backward(self):
        """
        판별자 역전파
        
        생성자를 freeze하고 판별자만 업데이트합니다.
        """
        with utils.temporary_freeze(self.gen):
            d_loss = sum(self.d_losses.values())
            d_loss.backward()

    def g_backward(self):
        """
        생성자 역전파
        
        판별자를 freeze하고 생성자만 업데이트합니다.
        """
        with utils.temporary_freeze(self.disc):
            g_loss = sum(self.g_losses.values())
            g_loss.backward()

    def ac_backward(self, retain_graph):
        """
        보조 분류기 역전파
        
        Args:
            retain_graph: gradient graph 유지 여부
        """
        if self.aux_clf is None:
            return

        # 메모리의 persistent_memory는 freeze
        org_grads = utils.freeze(self.gen.memory.persistent_memory)

        # 스타일 특징에 대한 보조 분류기 손실 역전파
        if 'ac' in self.ac_losses:
            self.ac_losses['ac'].backward(retain_graph=retain_graph)

        # 생성된 이미지 특징에 대한 보조 분류기 손실 역전파
        if 'ac_gen' in self.ac_losses:
            with utils.temporary_freeze(self.aux_clf):
                self.ac_losses['ac_gen'].backward(retain_graph=retain_graph)

        # freeze된 파라미터들 복원
        utils.unfreeze(self.gen.memory.persistent_memory, org_grads)

    def save(self, cur_loss, method, save_freq=None):
        """
        모델 체크포인트 저장
        
        Args:
            cur_loss: 현재 손실값
            method: 저장 방법
                - 'all': 스텝별로 체크포인트 저장
                - 'last': 'last.pth'로만 저장  
                - 'all-last': 둘 다 저장
            save_freq: 저장 주기 (옵션)
        """
        if method not in ['all', 'last', 'all-last']:
            return

        step_save = False  # 스텝별 저장 여부
        last_save = False  # last.pth 저장 여부
        
        if method == 'all' or (method == 'all-last' and self.step % save_freq == 0):
            step_save = True
        if method in ('last', 'all-last'):
            last_save = True
        assert step_save or last_save

        # 저장할 딕셔너리 구성
        save_dic = {
            'generator': self.gen.state_dict(),
            'generator_ema': self.gen_ema.state_dict(),
            'discriminator': self.disc.state_dict(),
            'd_optimizer': self.d_optim.state_dict(),
            'optimizer': self.g_optim.state_dict(),
            'epoch': self.step,  # NOTE: 실제로는 step
            'loss': cur_loss
        }

        # 보조 분류기가 있으면 추가
        if self.aux_clf is not None:
            save_dic['aux_clf'] = self.aux_clf.state_dict()
            save_dic['ac_optimizer'] = self.ac_optim.state_dict()

        # 파일 경로 설정
        ckpt_dir = Path("experiments/checkpoints", self.cfg['unique_name'])
        step_ckpt_name = "{:06d}-{}.pth".format(self.step, self.cfg['name'])
        last_ckpt_name = "last.pth"
        step_ckpt_path = ckpt_dir / step_ckpt_name
        last_ckpt_path = ckpt_dir / last_ckpt_name

        log = ""
        
        # 스텝별 저장
        if step_save:
            torch.save(save_dic, str(step_ckpt_path))
            log = "Checkpoint is saved to {}".format(step_ckpt_path)

            # last.pth도 저장하는 경우 심볼릭 링크 생성
            if last_save:
                utils.rm(last_ckpt_path)  # 기존 파일 삭제
                last_ckpt_path.symlink_to(step_ckpt_path.absolute())  # 심볼릭 링크 생성
                log += " w/ {} symlink".format(last_ckpt_name)

        # last.pth만 저장하는 경우
        if not step_save and last_save:
            utils.rm(last_ckpt_path)  # 기존 파일 삭제
            torch.save(save_dic, str(last_ckpt_path))
            log = "Checkpoint is saved to {}".format(last_ckpt_path)

        self.logger.info("{}\n".format(log))

    def plot(self, losses, discs, stats):
        """
        TensorBoard에 훈련 통계를 플롯
        
        Args:
            losses: 손실 통계 객체
            discs: 판별자 통계 객체  
            stats: 기타 통계 객체
        """
        # TensorBoard에 기록할 스칼라 값들
        tag_scalar_dic = {
            # 생성자 손실들
            'train/g_total_loss': losses.g_total.val,
            'train/pixel_loss': losses.pixel.val,

            # 판별자 및 생성자 GAN 손실
            'train/d_loss': losses.disc.val,
            'train/g_loss': losses.gen.val,
            
            # 판별자 출력값들 (실제/가짜)
            'train/d_real_font': discs.real_font.val,
            'train/d_real_char': discs.real_char.val,
            'train/d_fake_font': discs.fake_font.val,
            'train/d_fake_char': discs.fake_char.val,

            # 판별자 정확도들
            'train/d_real_font_acc': discs.real_font_acc.val,
            'train/d_real_char_acc': discs.real_char_acc.val,
            'train/d_fake_font_acc': discs.fake_font_acc.val,
            'train/d_fake_char_acc': discs.fake_char_acc.val
        }

        # 전체 이미지 판별자가 있는 경우 추가 통계
        if self.disc.use_rx:
            tag_scalar_dic.update({
                'train/d_real': discs.real.val,
                'train/d_fake': discs.fake.val,
                'train/d_real_acc': discs.real_acc.val,
                'train/d_fake_acc': discs.fake_acc.val
            })
            
        # 특징 매칭 손실이 사용되는 경우
        if self.cfg['fm_w'] > 0.:
            tag_scalar_dic['train/feature_matching'] = losses.fm.val

        # 보조 분류기가 있는 경우
        if self.aux_clf is not None:
            tag_scalar_dic.update({
                'train/ac_loss': losses.ac.val,           # 보조 분류기 손실
                'train/ac_acc': stats.ac_acc.val,         # 스타일 특징 분류 정확도
                'train/ac_gen_loss': losses.ac_gen.val,   # 생성 이미지 분류 손실
                'train/ac_gen_acc': stats.ac_gen_acc.val  # 생성 이미지 분류 정확도
            })

        # TensorBoard에 기록
        self.writer.add_scalars(tag_scalar_dic, self.step)

    def log(self, losses, discs, stats):
        """
        콘솔에 훈련 통계를 로그 출력
        
        Args:
            losses: 손실 통계 객체
            discs: 판별자 통계 객체
            stats: 기타 통계 객체
        """
        self.logger.info(
            "  Step {step:7d}: "                          # 현재 스텝
            "L1 {L.pixel.avg:7.4f}  "                     # 픽셀 L1 손실
            "D {L.disc.avg:7.3f}  "                       # 판별자 손실
            "G {L.gen.avg:7.3f}  "                        # 생성자 GAN 손실
            "FM {L.fm.avg:7.3f}  "                        # 특징 매칭 손실
            "AC {S.ac_acc.avg:5.1%}  "                    # 보조 분류기 정확도 (스타일)
            "AC_gen {S.ac_gen_acc.avg:5.1%}  "            # 보조 분류기 정확도 (생성)
            "R {D.real_acc.avg:7.3f}  "                   # 실제 이미지 판별 정확도
            "F {D.fake_acc.avg:7.3f}  "                   # 가짜 이미지 판별 정확도
            "R_font {D.real_font_acc.avg:7.3f}  "         # 실제 폰트 판별 정확도
            "F_font {D.fake_font_acc.avg:7.3f}  "         # 가짜 폰트 판별 정확도
            "R_char {D.real_char_acc.avg:7.3f}  "         # 실제 문자 판별 정확도
            "F_char {D.fake_char_acc.avg:7.3f}  "         # 가짜 문자 판별 정확도
            "B_stl {S.B_style.avg:5.1f}  "                # 스타일 배치 크기
            "B_trg {S.B_target.avg:5.1f}"                 # 타겟 배치 크기
            .format(step=self.step, L=losses, D=discs, S=stats))
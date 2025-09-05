"""
DMFont
Copyright (c) 2020-present NAVER Corp.
MIT license

폰트 데이터셋을 HDF5 형식으로 전처리하는 모듈
다양한 언어(한국어, 태국어)의 폰트에서 문자 이미지를 추출하여 머신러닝용 데이터셋으로 변환
"""
import os
import sys
import os.path as osp
sys.path.append(os.getcwd())
import json
from itertools import chain
from functools import reduce
from pathlib import Path
from tqdm import tqdm

import h5py as h5
import fire
import numpy as np
from PIL import Image, ImageDraw, ImageFont, features
from fontTools.ttLib import TTFont

from logger import Logger
from datasets import thai_decompose as thai


# 언어별 유니코드 범위 정의
# 각 언어에서 사용할 문자들의 유니코드 범위를 지정
CODE_RANGE = {
    # 한국어: ASCII 기본 문자, 한글 자모, 한글 완성형
    'kor': [[0x0021, 0x007E], [0x3131, 0x3163], [0xAC00, 0xD7A3]],
    # 태국어: 태국 문자 및 기호
    'thai': [[0x0E01, 0x0E3A], [0x0E3F, 0x0E5B]]
}


def get_code_points(language):
    """
    지정된 언어에 해당하는 모든 문자들의 집합을 반환
    
    Parameters
    ----------
    language : str
        언어 코드 ('kor' 또는 'thai')
    
    Returns
    -------
    codes : set
        해당 언어의 모든 문자들을 포함하는 집합
    """
    codes = set()
    code_range = CODE_RANGE[language]
    
    # 각 유니코드 범위에서 모든 문자를 추출
    for rangemin, rangemax in code_range:
        for codepoint in range(rangemin, rangemax+1):
            codes.add(chr(codepoint))

    return codes


# region - dump hdf5
def dump_to_hdf5(dump_path, font_name, images, chars, compression=None):
    """
    이미지와 문자 데이터를 HDF5 파일 형식으로 저장
    
    Parameters
    ----------
    dump_path : str
        저장할 HDF5 파일 경로
    font_name : str
        폰트 이름
    images : list
        PIL Image 객체들의 리스트
    chars : list
        각 이미지에 대응하는 문자 코드들의 리스트
    compression : str, optional
        압축 방식
    """
    with h5.File(dump_path, 'w') as f:
        # 데이터셋 그룹 생성
        dset = f.create_group('dataset')
        dset.attrs['font_name'] = font_name
        
        N = len(images)
        # 이미지 데이터를 NumPy 배열로 변환하여 저장
        dset.create_dataset('images', (N, 128, 128), np.uint8, compression=compression,
                            data=np.stack(images))
        
        # 문자 코드 데이터 저장
        data = np.array(chars)
        dset.create_dataset('chars', data.shape, int, compression=compression,
                            data=np.array(chars))


class FontProcessor(object):
    """
    폰트 파일에서 문자 이미지를 추출하고 전처리하는 클래스
    """
    
    def __init__(self, language, resize_method="bilinear", font_size_factor=2, sample_size=128):
        """
        FontProcessor 초기화
        
        Parameters
        ----------
        language : str
            처리할 언어 ('kor' 또는 'thai')
        resize_method : str
            이미지 리사이즈 방법 (기본값: "bilinear")
        font_size_factor : int
            폰트 크기 배율 (기본값: 2)
        sample_size : int
            최종 출력 이미지 크기 (기본값: 128x128)
        """
        # 태국어의 경우 raqm 라이브러리가 필요 (복합 문자 렌더링용)
        if language == 'thai':
            assert features.check('raqm'), 'Please install raqm first for thai font rendering'

        # 로거 설정 (에러 로그 기록용)
        self.logger = Logger.get(file_path='preparedata.log', level='error')

        self.language = language
        # 해당 언어의 대상 문자들 가져오기
        self.targetcodes = get_code_points(self.language)
        
        # 이미지 리사이즈 방법 설정
        if resize_method == 'bilinear':
            self.resize_method = Image.BILINEAR
        else:
            raise ValueError('Invalid resize method: {}'.format(resize_method))
            
        self.sample_size = sample_size
        # 실제 렌더링할 폰트 크기 (나중에 sample_size로 축소)
        self.font_size = self.sample_size * font_size_factor

    def ord(self, char):
        """
        문자를 해당 언어에 맞는 코드로 변환
        
        Parameters
        ----------
        char : str
            변환할 문자
        
        Returns
        -------
        int or list
            한국어의 경우 유니코드 값, 태국어의 경우 분해된 코드 리스트
        """
        if self.language == 'kor':
            return ord(char)
        elif self.language == 'thai':
            # 태국어는 복합 문자이므로 분해하여 처리
            return thai.decompose_ords(char)
        else:
            raise ValueError(self.language)

    def is_renderable_char(self, font, ch):
        """
        주어진 폰트에서 특정 문자가 렌더링 가능한지 확인
        
        Parameters
        ----------
        font : PIL.ImageFont
            확인할 폰트 객체
        ch : str
            확인할 문자
        
        Returns
        -------
        bool
            렌더링 가능하면 True, 불가능하면 False
        """
        # 태국어의 경우 문자 순서 수정
        ch = self.fix_char_order_if_thai(ch)
        
        try:
            # 문자의 바운딩 박스 크기 계산
            bbox = font.getbbox(ch)
            width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            size = width * height
        except OSError:
            # 폰트에서 해당 문자를 열 수 없는 경우
            self.logger.warning('{}, "{}" ({}) cannot be opened'.format(font, ch, self.ord(ch)))
            return False
            
        # 크기가 0인 경우 (빈 문자)
        if not size:
            bbox = font.getbbox(ch)
            width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            size = width * height
            self.logger.warning('{}, "{}" ({}) has size {}'.format(
                font, ch, self.ord(ch), size)
            )
            return False

        return True

    def avail_chars(self, targetfontpath, pilfont):
        """
        특정 폰트에서 사용 가능한 문자들을 반환
        
        Parameters
        ----------
        targetfontpath : str
            폰트 파일 경로
        pilfont : PIL.ImageFont
            PIL 폰트 객체
        
        Returns
        -------
        set
            렌더링 가능한 문자들의 집합
        """
        # 폰트 파일에서 지원하는 문자들 추출
        ttfont = TTFont(targetfontpath)
        existing_chars = {chr(key) for table in ttfont['cmap'].tables for key in table.cmap.keys()}
        
        # 실제로 렌더링 가능한 문자들만 필터링
        rendercheckedchars = {ch for ch in existing_chars if self.is_renderable_char(pilfont, ch)}

        return rendercheckedchars

    def get_charsize(self, char, font):
        """
        특정 문자의 렌더링 크기를 계산
        
        Parameters
        ----------
        char : str
            크기를 측정할 문자
        font : PIL.ImageFont
            사용할 폰트
        
        Returns
        -------
        tuple
            (width, height) 문자의 실제 크기
        """
        # 태국어의 경우 문자 순서 수정
        char = self.fix_char_order_if_thai(char)
        
        # 문자의 바운딩 박스 계산
        bbox = font.getbbox(char)  # returns (x0, y0, x1, y1)
        size_x = bbox[2] - bbox[0]  # width
        size_y = bbox[3] - bbox[1]  # height
        
        # 문자의 마스크에서 실제 오프셋 계산
        mask = font.getmask(char)
        bbox = mask.getbbox()  # returns (x0, y0, x1, y1)

        if bbox is not None:
            offset_x, offset_y = bbox[0], bbox[1]
        else:
            # 문자 출력이 없을 경우 기본값
            offset_x, offset_y = 0, 0

        return size_x-offset_x, size_y-offset_y

    def render_center_no_offset(self, char, font, fontmaxsize, size=128, margin=0):
        """
        문자를 중앙 정렬하여 렌더링하고 지정된 크기로 조정
        
        Parameters
        ----------
        char : str
            렌더링할 문자
        font : PIL.ImageFont
            사용할 폰트
        fontmaxsize : int
            해당 폰트에서 가장 큰 문자의 크기 (정규화 기준)
        size : int
            최종 출력 이미지 크기
        margin : float
            여백 비율
        
        Returns
        -------
        PIL.Image or False
            성공 시 렌더링된 이미지, 실패 시 False
        """
        # 태국어의 경우 문자 순서 수정
        char = self.fix_char_order_if_thai(char)
        
        # 문자의 바운딩 박스 계산
        bbox = font.getbbox(char)  # returns (x0, y0, x1, y1)
        if bbox is None:
            self.logger.warning(
                '{}, "{}" ({}) has no bounding box'.format(font, char, self.ord(char))
            )
            return False
        
        x0, y0, x1, y1 = bbox
        roi_w = x1 - x0  # 관심 영역의 너비
        roi_h = y1 - y0  # 관심 영역의 높이

        # 크기가 유효하지 않은 경우
        if roi_w <= 0 or roi_h <= 0:
            self.logger.warning(
                '{}, "{}" ({}) has non-positive size (w={}, h={})'.format(font, char, self.ord(char), roi_w, roi_h)
            )
            return False
        
        # 문자 크기에 맞는 이미지 생성 (흰색 배경)
        img = Image.new("L", (roi_w, roi_h), 255)
        draw = ImageDraw.Draw(img)
        # 오프셋을 보정하여 문자 그리기 (검은색)
        draw.text((-x0, -y0), char, font=font, fill=0)
        
        # 빈 이미지 확인 (검은 픽셀이 없는 경우)
        npimg = 255 - np.array(img)  # 색상 반전 (검은색이 255가 되도록)
        if not npimg.sum():
            self.logger.warning(
                '{}, "{}" ({}) is empty (no black pixels)'.format(font, char, self.ord(char))
            )
            return False

        # 비어있지 않은 영역만 크롭
        wsum = npimg.sum(0)  # 세로축 합계
        hsum = npimg.sum(1)  # 가로축 합계
        w_indices = wsum.nonzero()[0]  # 0이 아닌 열 인덱스
        h_indices = hsum.nonzero()[0]  # 0이 아닌 행 인덱스

        if len(w_indices) == 0 or len(h_indices) == 0:
            self.logger.warning(
                '{}, "{}" ({}) is empty after cropping'.format(font, char, self.ord(char))
            )
            return False

        # 실제 문자가 있는 영역의 경계 계산
        wmin, wmax = w_indices.min(), w_indices.max()
        hmin, hmax = h_indices.min(), h_indices.max()

        # 해당 영역만 크롭하고 색상 다시 반전
        npimg = 255 - npimg[hmin:hmax+1, wmin:wmax+1]

        # 여백을 포함한 캔버스 크기 계산
        canvas_size = int(fontmaxsize * (1 + margin))
        content_h, content_w = npimg.shape
        
        # 중앙 정렬을 위한 패딩 계산
        left_margin = (canvas_size - content_w) // 2
        right_margin = canvas_size - content_w - left_margin
        top_margin = (canvas_size - content_h) // 2
        bottom_margin = canvas_size - content_h - top_margin

        # 패딩 추가 (흰색으로 채움)
        npimg = np.pad(npimg, ((top_margin, bottom_margin), (left_margin, right_margin)),
                    'constant', constant_values=255)

        # 최종 크기로 리사이즈
        img = Image.fromarray(npimg).resize((size, size), resample=self.resize_method)

        return img

    def dump_fonts(self, fonts, dump_dir, compression=None):
        """
        여러 폰트 파일을 처리하여 HDF5 데이터셋으로 변환
        
        각 폰트에서 사용 가능한 모든 문자를 렌더링하고,
        폰트 내에서 가장 큰 문자 크기를 기준으로 정규화하여 저장
        
        Parameters
        ----------
        fonts : list
            처리할 폰트 파일 경로들의 리스트
        dump_dir : str or Path
            출력 디렉토리 경로
        compression : str, optional
            HDF5 압축 방식
        """
        self.logger.info('# Font candidates: {}'.format(len(fonts)))

        # 출력 디렉토리 생성
        dump_dir = Path(dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        assert dump_dir.is_dir()

        n_fonts = len(fonts)
        
        # 각 폰트에 대해 처리
        for i, targetfontpath in enumerate(fonts):
            targetfontname = os.path.basename(targetfontpath)  # 확장자 포함 파일명
            font_name = os.path.splitext(targetfontname)[0]  # 확장자 제외 파일명
            
            # 디버깅용: 특정 폰트만 처리하고 싶을 때 사용
            # if font_name != "UhBee Skyrain": continue
            
            # 출력 HDF5 파일 경로 설정
            hdf5_name = "{}.hdf5".format(font_name)
            dump_path = dump_dir / hdf5_name

            # 이미 처리된 폰트는 건너뛰기
            if dump_path.exists():
                continue

            # 폰트 로드
            font = ImageFont.truetype(targetfontpath, self.font_size)
            
            # 해당 폰트에서 사용 가능한 문자들 확인
            codepoints = self.avail_chars(targetfontpath, font)
            # 대상 문자들과 교집합 구하기 (우리가 원하는 문자 중 폰트에서 지원하는 것들)
            codepoints = codepoints & self.targetcodes
            
            # 언어별 유효성 검사
            if self.language == 'kor':
                if len(codepoints) == 0:
                    self.logger.error("Font {} don't have any valid chars".format(targetfontname))
                    continue
            elif self.language == 'thai':
                # 태국어의 경우 모든 구성 요소가 있어야 함
                if codepoints != self.targetcodes:
                    self.logger.error("Font {} don't have full components ({}, {})".format(
                        targetfontname, len(codepoints), len(self.targetcodes)))
                    continue
                # 태국어는 완성된 문자 집합 사용
                codepoints = list(thai.complete_chars())
            else:
                raise ValueError(self.language)

            # 해당 폰트에서 가장 큰 문자의 크기 계산 (정규화 기준점)
            sizes = [self.get_charsize(codepoint, font) for codepoint in codepoints]
            fontmaxsize = max(chain(*sizes))  # 모든 크기 중 최대값

            # 이미지와 문자 코드 저장할 리스트
            images = []
            chars = []
            
            # 각 문자에 대해 이미지 렌더링
            for codepoint in tqdm(codepoints, desc=f"{i+1}. {font_name}"):
                if not codepoint:
                    self.logger.error("Wrong codepoint: {}".format(codepoint))
                    raise ValueError(codepoint)

                # 디버깅용: 특정 문자에서 중단점 설정
                # if codepoint == '쟈':
                #     a=1
                
                # 문자 렌더링
                img = self.render_center_no_offset(codepoint, font, fontmaxsize,
                                                   size=self.sample_size, margin=0)
                if not img:
                    # 렌더링 실패한 문자는 건너뛰기
                    continue
                    
                images.append(img)
                chars.append(self.ord(codepoint))

            # HDF5 파일로 저장
            dump_to_hdf5(dump_path, targetfontname, images, chars, compression=compression)

            self.logger.info("[{:3d}/{:3d}] {} has {} valid chars and {} images...".format(
                i+1, n_fonts, font_name, len(codepoints), len(images)))

    def fix_char_order_if_thai(self, char):
        """
        태국어 문자의 구성 요소 순서를 올바른 렌더링 순서로 수정
        
        태국어는 복합 문자이므로 구성 요소의 순서가 렌더링에 영향을 줌
        자음 - 상단 - 최상단 - 하단 => 자음 - 하단 - 상단 - 최상단 순서로 변경
        
        Parameters
        ----------
        char : str
            수정할 태국어 문자
        
        Returns
        -------
        str
            올바른 순서로 정렬된 문자
        """
        if self.language == 'thai':
            # 태국어 문자 분해
            ords = thai.decompose_ords(char)
            # 올바른 순서로 재구성: [자음, 하단, 상단, 최상단]
            char = thai.compose(ords[0], ords[3], ords[1], ords[2])
        return char


def main(language, fonts_dir, meta_path, dump_dir):
    """
    메인 실행 함수
    
    메타데이터에서 지정된 폰트들을 읽어와서 HDF5 데이터셋으로 변환
    
    Parameters
    ----------
    language : str
        처리할 언어 ('kor' 또는 'thai')
    fonts_dir : str
        폰트 파일들이 있는 디렉토리 경로
    meta_path : str
        훈련/검증용 폰트 목록이 있는 메타데이터 파일 경로
    dump_dir : str
        출력 데이터셋 디렉토리 경로
    """
    fonts_dir = Path(fonts_dir)

    # 메타데이터에서 폰트 목록 로드
    meta = json.load(open(meta_path))
    # 훈련용과 검증용 폰트를 모두 합침
    allfonts = set(meta['train']['fonts'] + meta['valid']['fonts'])
    # allfonts = set(meta['valid']['fonts'])  # 검증용만 처리하고 싶을 때
    
    # 실제 존재하는 TTF 파일들 중에서 메타데이터에 있는 것들만 선택
    fonts = [
        str(fname) for fname in fonts_dir.rglob("*.ttf") if fname.name in allfonts
    ]
    
    # 디버깅용: 표준 폰트 목록과 사용자 정의 폰트 목록 비교
    standard_allfonts = sorted(os.listdir(f"{root_dir}/datasets/all_fonts"))
    custom_allfonts = sorted(allfonts)
    for custom in custom_allfonts:
        if not custom in standard_allfonts:
            print(custom)  # 누락된 폰트 출력
    
    # 메타데이터의 폰트 수와 실제 찾은 폰트 수가 일치하는지 확인
    assert len(allfonts) == len(fonts)

    # 폰트 프로세서 생성 및 실행
    processor = FontProcessor(language)
    processor.dump_fonts(fonts, dump_dir)


if __name__ == '__main__':
    """
    스크립트 직접 실행 시 실행되는 부분
    Fire를 사용하여 명령줄 인터페이스 제공하거나 직접 실행 가능
    """
    # 명령줄에서 fire 라이브러리를 통해 실행할 때 사용
    # fire.Fire(main)
    
    # 개발용 직접 실행 예시 (절대 경로)
    # main(language='kor', fonts_dir='/home/dev/dmfont/datasets/all_fonts',
    #      meta_path='/home/dev/dmfont/meta/kor_split2.json',
    #      dump_dir='/home/dev/dmfont/datasets/hdf5')
    
    # 현재 작업 디렉토리 기준 상대 경로로 실행
    root_dir = os.getcwd()
    main(language='kor', 
         fonts_dir=f'{root_dir}/datasets/all_fonts',        # 모든 폰트 파일들이 있는 디렉토리
         meta_path=f'{root_dir}/meta/kor_split2.json',      # 훈련/검증 폰트 분할 정보
         dump_dir=f'{root_dir}/datasets/hdf5')              # HDF5 출력 디렉토리
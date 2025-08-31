"""
DMFont
Copyright (c) 2020-present NAVER Corp.
MIT license
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


CODE_RANGE = {
    'kor': [[0x0021, 0x007E], [0x3131, 0x3163], [0xAC00, 0xD7A3]],
    'thai': [[0x0E01, 0x0E3A], [0x0E3F, 0x0E5B]]
}


def get_code_points(language):
    codes = set()
    code_range = CODE_RANGE[language]
    for rangemin, rangemax in code_range:
        for codepoint in range(rangemin, rangemax+1):
            codes.add(chr(codepoint))

    return codes


# region - dump hdf5
def dump_to_hdf5(dump_path, font_name, images, chars, compression=None):
    with h5.File(dump_path, 'w') as f:
        dset = f.create_group('dataset')
        dset.attrs['font_name'] = font_name
        N = len(images)
        dset.create_dataset('images', (N, 128, 128), np.uint8, compression=compression,
                            data=np.stack(images))
        data = np.array(chars)
        dset.create_dataset('chars', data.shape, int, compression=compression,
                            data=np.array(chars))


class FontProcessor(object):
    def __init__(self, language, resize_method="bilinear", font_size_factor=2, sample_size=128):
        if language == 'thai':
            assert features.check('raqm'), 'Please install raqm first for thai font rendering'

        self.logger = Logger.get(file_path='preparedata.log', level='error')

        self.language = language
        self.targetcodes = get_code_points(self.language)
        if resize_method == 'bilinear':
            self.resize_method = Image.BILINEAR
        else:
            raise ValueError('Invalid resize method: {}'.format(resize_method))
        self.sample_size = sample_size
        self.font_size = self.sample_size * font_size_factor

    def ord(self, char):
        if self.language == 'kor':
            return ord(char)
        elif self.language == 'thai':
            return thai.decompose_ords(char)
        else:
            raise ValueError(self.language)

    def is_renderable_char(self, font, ch):
        ch = self.fix_char_order_if_thai(ch)
        try:
            # size = reduce(lambda x, y: x * y, font.getsize(ch))
            bbox = font.getbbox(ch)
            width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            size = width * height
        except OSError:
            self.logger.warning('{}, "{}" ({}) cannot be opened'.format(font, ch, self.ord(ch)))
            return False
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
        ttfont = TTFont(targetfontpath)
        existing_chars = {chr(key) for table in ttfont['cmap'].tables for key in table.cmap.keys()}
        rendercheckedchars = {ch for ch in existing_chars if self.is_renderable_char(pilfont, ch)}

        return rendercheckedchars

    def get_charsize(self, char, font):
        char = self.fix_char_order_if_thai(char)
        
        bbox = font.getbbox(char)  # returns (x0, y0, x1, y1)
        size_x = bbox[2] - bbox[0]  # width
        size_y = bbox[3] - bbox[1]  # height
        
        # size_x, size_y = font.getsize(char)
        # offset_x, offset_y = font.getoffset(char)
        mask = font.getmask(char)
        bbox = mask.getbbox()  # returns (x0, y0, x1, y1)

        if bbox is not None:
            offset_x, offset_y = bbox[0], bbox[1]
        else:
            offset_x, offset_y = 0, 0  # 문자 출력이 없을 경우 기본값

        return size_x-offset_x, size_y-offset_y

    def render_center_no_offset(self, char, font, fontmaxsize, size=128, margin=0):
        char = self.fix_char_order_if_thai(char)
        # size_x, size_y = font.getsize(char)
        bbox = font.getbbox(char)  # returns (x0, y0, x1, y1)
        if bbox is None:
            self.logger.warning(
                '{}, "{}" ({}) has no bounding box'.format(font, char, self.ord(char))
            )
            return False
        
        x0, y0, x1, y1 = bbox
        roi_w = x1 - x0
        roi_h = y1 - y0

        if roi_w <= 0 or roi_h <= 0:
            self.logger.warning(
                '{}, "{}" ({}) has non-positive size (w={}, h={})'.format(font, char, self.ord(char), roi_w, roi_h)
            )
            return False
        
        img = Image.new("L", (roi_w, roi_h), 255)
        draw = ImageDraw.Draw(img)
        draw.text((-x0, -y0), char, font=font, fill=0)
        
        # Check for empty image
        npimg = 255 - np.array(img)
        if not npimg.sum():
            self.logger.warning(
                '{}, "{}" ({}) is empty (no black pixels)'.format(font, char, self.ord(char))
            )
            return False

        # Crop non-white region
        wsum = npimg.sum(0)
        hsum = npimg.sum(1)
        w_indices = wsum.nonzero()[0]
        h_indices = hsum.nonzero()[0]

        if len(w_indices) == 0 or len(h_indices) == 0:
            self.logger.warning(
                '{}, "{}" ({}) is empty after cropping'.format(font, char, self.ord(char))
            )
            return False

        wmin, wmax = w_indices.min(), w_indices.max()
        hmin, hmax = h_indices.min(), h_indices.max()

        npimg = 255 - npimg[hmin:hmax+1, wmin:wmax+1]

        # Calculate canvas with margin
        canvas_size = int(fontmaxsize * (1 + margin))
        content_h, content_w = npimg.shape
        left_margin = (canvas_size - content_w) // 2
        right_margin = canvas_size - content_w - left_margin
        top_margin = (canvas_size - content_h) // 2
        bottom_margin = canvas_size - content_h - top_margin

        # Pad and resize
        npimg = np.pad(npimg, ((top_margin, bottom_margin), (left_margin, right_margin)),
                    'constant', constant_values=255)

        img = Image.fromarray(npimg).resize((size, size), resample=self.resize_method)

        
        return img

    def dump_fonts(self, fonts, dump_dir, compression=None):
        """
        calculates maximum size of available codepoints
        target text character is rendered accordingly
        relative size within each font are maintained proportionally
        relative size across fonts are adjusted
        (Maximum size codepoint to be maintained in the canvas)
        """
        self.logger.info('# Font candidates: {}'.format(len(fonts)))

        dump_dir = Path(dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        assert dump_dir.is_dir()

        n_fonts = len(fonts)
        for i, targetfontpath in enumerate(fonts):
            targetfontname = os.path.basename(targetfontpath)  # w/ ext
            font_name = os.path.splitext(targetfontname)[0]  # w/o ext
            
            # if font_name != "UhBee Skyrain": continue
            
            hdf5_name = "{}.hdf5".format(font_name)
            dump_path = dump_dir / hdf5_name

            if dump_path.exists():
                continue

            font = ImageFont.truetype(targetfontpath, self.font_size)
            codepoints = self.avail_chars(targetfontpath, font)
            # available & desired fonts
            codepoints = codepoints & self.targetcodes  # avail chars
            if self.language == 'kor':
                if len(codepoints) == 0:
                    self.logger.error("Font {} don't have any valid chars".format(targetfontname))
                    continue
            elif self.language == 'thai':
                if codepoints != self.targetcodes:
                    self.logger.error("Font {} don't have full components ({}, {})".format(
                        targetfontname, len(codepoints), len(self.targetcodes)))
                    continue

                codepoints = list(thai.complete_chars())
            else:
                raise ValueError(self.language)

            # max rendered size
            sizes = [self.get_charsize(codepoint, font) for codepoint in codepoints]
            fontmaxsize = max(chain(*sizes))

            images = []
            chars = []
            c = 0
            for codepoint in tqdm(codepoints, desc=f"{i+1}. {font_name}"):
                if not codepoint:
                    self.logger.error("Wrong codepoint: {}".format(codepoint))
                    raise ValueError(codepoint)

                # if codepoint == '쟈':
                #     a=1
                img = self.render_center_no_offset(codepoint, font, fontmaxsize,
                                                   size=self.sample_size, margin=0)
                if not img:
                    continue
                images.append(img)
                chars.append(self.ord(codepoint))

            dump_to_hdf5(dump_path, targetfontname, images, chars, compression=compression)

            self.logger.info("[{:3d}/{:3d}] {} has {} valid chars and {} images...".format(
                i+1, n_fonts, font_name, len(codepoints), len(images)))

    def fix_char_order_if_thai(self, char):
        """ Fix character component order for correct rendering
            consonant - upper - highest - lower
         => consonant - lower - upper - highest
        """
        if self.language == 'thai':
            ords = thai.decompose_ords(char)
            char = thai.compose(ords[0], ords[3], ords[1], ords[2])
        return char


def main(language, fonts_dir, meta_path, dump_dir):
    """
    Args:
        language: kor / thai
        fonts_dir: font directory that has ttf files
        meta_path: meta file path
        dump_dir: dataset dir
    """
    fonts_dir = Path(fonts_dir)

    meta = json.load(open(meta_path))
    allfonts = set(meta['train']['fonts'] + meta['valid']['fonts'])
    # allfonts = set(meta['valid']['fonts'])
    fonts = [
        str(fname) for fname in fonts_dir.rglob("*.ttf") if fname.name in allfonts
    ]
    

    standard_allfonts = sorted(os.listdir(f"{root_dir}/datasets/all_fonts"))
    custom_allfonts = sorted(allfonts)
    for custom in custom_allfonts:
        if not custom in standard_allfonts:
            print(custom)
    
    assert len(allfonts) == len(fonts)

    processor = FontProcessor(language)
    processor.dump_fonts(fonts, dump_dir)


if __name__ == '__main__':
    # fire.Fire(main)
    
    # main(language='kor', fonts_dir='/home/dev/dmfont/datasets/all_fonts',
    #      meta_path='/home/dev/dmfont/meta/kor_split2.json',
    #      dump_dir='/home/dev/dmfont/datasets/hdf5')
    
    root_dir = os.getcwd()
    main(language='kor', fonts_dir=f'{root_dir}/datasets/all_fonts',
         meta_path=f'{root_dir}/meta/kor_split2.json',
         dump_dir=f'{root_dir}/datasets/hdf5')

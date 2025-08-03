import os
import os.path as osp

import numpy as np
import h5py as h5
import cv2
from glob import glob

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
        


# 17세기 한글 진법언해 목판본 서체 글자
# 유니코드 이스케이프(UTF-16 코드 유닛 형태)
# targets = ["바" ,"배" ,"백" ,"버" ,"법" ,"보" ,"본" ,"부" ,"비" ,"빛" ,"사" ,"상" ,"생" ,"서" ,"선" ,"성" ,"세" ,"속" ,"수" ,"쉬" ,"쉽" ,
#            "식" ,"신" ,"써" ,"쓰" ,"쓴" ,"아" ,"않" ,"알" ,"야" ,"어" ,"언" ,"얼" ,"없" ,"엇" ,"에" ,"여" ,"오" ,"외" ,"요" ,"우" ,"울" ,
#            "워" ,"으" ,"은" ,"의" ,"이" ,"인" ,"있" ,"자" ,"잘" ,"장" ,"저" ,"절" ,"정" ,"제" ,"조" ,"좋" ,"쥐" ,"즉" ,"지" ,"진" ,"집" ,
#            "째" ,"첫" ,"쳇" ,"총" ,"퀴" ,"타" ,"터" ,"통" ,"파" ,"푸" ,"하" ,"한" ,"함" ,"해" ,"헌" ,"헤" ,"형" ,"후" ,"히" ,"가" ,"각" ,
#            "같" ,"것" ,"게" ,"고" ,"교" ,"구" ,"국" ,"군" ,"귀" ,"글" ,"기" ,"까" ,"나" ,"남" ,"낳" ,"내" ,"녀" ,"는" ,"늠" ,"능" ,"니" ,
#            "다" ,"닭" ,"대" ,"더" ,"데" ,"도" ,"동" ,"두" ,"둘" ,"드" ,"들" ,"또" ,"띄" ,"라" ,"람" ,"랑" ,"로" ,"론" ,"른" ,"를" ,"릇" ,
#            "리" ,"린" ,"마" ,"만" ,"말" ,"면" ,"모" ,"목" ,"무" ,"문" ,"묾" ,"민"]
# targets = ["글", "기", "ㄲ", "까", "ㄴ", "나", "남", "낳", "내", "녀", "는", "늠", "능", 
#            "니", "ㄷ", "다", "닭", "대", "더", "데", "도", "동", "두", "둘", "드", "들", 
#            "ㄸ", "또", "띄", "ㄹ", "라", "람", "랑", "로", "론", "른", "를", "릇", "리", 
#            "린", "ㅁ", "마", "만", "말", "면", "모", "목", "무", "문", "묾", "민", "ㅂ", 
#            "바", "배", "백", "버", "법", "보", "본", "부", "비", "빛", "ㅃ", "ㅅ", "사", 
#            "상", "생", "서", "선", "성", "세", "속", "수", "쉬", "쉽", "식", "신", "ㅆ", 
#            "써", "쓰", "쓴", "ㅇ", "아", "않", "알", "야", "어", "언", "얼", "없", "엇", 
#            "에", "여", "오", "외", "요", "우", "울", "워", "으", "은", "의", "이", "인", 
#            "있", "ㅈ", "자", "잘", "장", "저", "절", "정", "제", "조", "좋", "쥐", "즉", 
#            "지", "진", "집", "ㅉ", "째", "ㅊ", "첫", "쳇", "총", "ㅋ", "퀴", "ㅌ", "타", 
#            "터", "통", "ㅍ", "파", "푸", "ㅎ", "하", "한", "함", "해", "헌", "헤", "형", 
#            "후", "히", "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", 
#            "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ", "ㄱ", "가", "각", 
#            "같", "것", "게", "고", "교", "구", "국", "군"]
targets = [ "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ",
           "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ", "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ",
           "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ",
           "ㅡ", "ㅢ", "ㅣ"]
target_dir = r"/home/dev/dmfont/datasets/Jinbeop_raw_font_image/"
dump_path = r'/home/dev/dmfont/datasets/hdf5/JinbeopUnhae_2.hdf5'
targetfontname = 'JinbeopUnhae.ttf'


chars = []
images = []
escaped_list = []
for c in targets:
    escaped = c.encode('unicode_escape').decode('ascii')
    escaped_list.append(escaped)
    
    target_image_path = osp.join(target_dir, c+'.png')
    img = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    images.append(img)

    chars.append(ord(c))
    
# dump_to_hdf5(dump_path, targetfontname, images, chars, compression=None)

# chars = [s.strip().strip("'") for s in escaped_list]
# print([f"\"{c}\"" for c in chars])

print(escaped_list)
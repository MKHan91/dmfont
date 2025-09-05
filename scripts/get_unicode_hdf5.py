import os
import os.path as osp

import numpy as np
import h5py as h5
import cv2
from glob import glob

def dump_to_hdf5(dump_path: list, 
                 font_name: str, 
                 images: list[np.ndarray], 
                 chars: list[int], compression=None) -> None:
    
    """
    Input
    -----
    dump_path   : hdf5로 저장 되는 파일 경로
    font_name   : ttf 확장자 이름인데, hdf5에 저장될 때 사용하는 이름으로 그냥 형식상 적어야 함.
                  형식상 JinbeopUnhae로 적음.
    images      : 진법언해 이미지 배열들
    chars       : 진법언해 글자들

    """
    
    # HDF5 파일을 쓰기 모드로 열기
    with h5.File(dump_path, 'w') as f:
        # 'dataset'이라는 그룹 생성 (HDF5에서 폴더 역할)
        dset = f.create_group('dataset')
        # 폰트 이름을 그룹의 속성(attribute)으로 저장
        dset.attrs['font_name'] = font_name
        # 이미지 개수 확인
        N = len(images)
        # 이미지 데이터셋 생성 및 저장
        # 형태: (N, 128, 128), 데이터 타입: uint8 (0-255)
        dset.create_dataset('images', (N, 128, 128), np.uint8, compression=compression,
                            data=np.stack(images))
        # 문자 코드 배열로 변환
        data = np.array(chars)
        # 문자 데이터셋 생성 및 저장
        # 각 이미지에 대응하는 유니코드 값들을 저장
        dset.create_dataset('chars', data.shape, int, compression=compression,
                            data=np.array(chars))
        


def load_jinbeop_char(directory: list) -> list:
    # 17세기 한글 진법언해 목판본 서체 글자
    # 유니코드 이스케이프(UTF-16 코드 유닛 형태)
    
    """
    Input
    -----
    directory: 진법 폰트 이미지가 있는 경로
    
    Output
    -----
    fileNames: 진법 언해 파일 이름들
    """
    
    # 디렉토리 내의 모든 파일명에서 확장자(.png) 제거
    # 예: "가.png" -> "가"
    fileNames = [fname[:-4] for fname in os.listdir(directory)]

    return fileNames




def main():
    """
    메인 실행 함수
    진법언해 폰트 이미지들을 읽어와서 HDF5 데이터셋으로 변환
    """
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
    # targets = [ "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ",
    #            "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ", "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ",
    #            "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ",
    #            "ㅡ", "ㅢ", "ㅣ"]
    # 디렉토리에서 자동으로 문자 목록 추출
    targets = load_jinbeop_char(directory=target_dir)

    # 데이터 저장을 위한 리스트 초기화
    chars = []          # 유니코드 값들을 저장할 리스트
    images = []         # 이미지 배열들을 저장할 리스트
    escaped_list = []   # 유니코드 이스케이프 문자열들 (디버깅용)
    
    # 각 문자에 대해 이미지 로드 및 전처리 수행
    for c in targets:
        # 문자를 유니코드 이스케이프 형태로 변환 (디버깅용)
        # 예: "가" -> "\\uac00"
        escaped = c.encode('unicode_escape').decode('ascii')
        escaped_list.append(escaped)
        
        # 해당 문자의 이미지 파일 경로 생성
        target_image_path = osp.join(target_dir, c+'.png')
        
        # 이미지를 그레이스케일로 로드
        img = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
        
        # 이미지를 128x128 크기로 리사이즈
        # 모든 이미지를 동일한 크기로 표준화
        img = cv2.resize(img, (128, 128))
        images.append(img)

        # 문자의 유니코드 값을 정수로 변환하여 저장
        # ord() 함수는 문자를 해당하는 유니코드 숫자로 변환
        chars.append(ord(c))
    
    # 전처리된 이미지와 문자 데이터를 HDF5 파일로 저장
    dump_to_hdf5(dump_path, targetfontname, images, chars, compression=None)

    print("Done!")
    

if __name__ == "__main__":
    """
    스크립트가 직접 실행될 때만 실행되는 부분
    필요한 경로들을 설정하고 main 함수 호출
    """
    # 현재 작업 디렉토리 경로 가져오기
    root_dir = os.getcwd()
    
    # 진법언해 폰트 이미지들이 저장된 디렉토리 경로
    target_dir = f"{root_dir}/datasets/Jinbeop_font_image"
    
    # 생성될 HDF5 파일의 저장 경로
    dump_path = f'{root_dir}/datasets/hdf5/JinbeopUnhae_ver2.hdf5'
    
    # 폰트 파일명 (메타데이터로 사용)
    targetfontname = 'JinbeopUnhae.ttf'
    
    # 메인 함수 실행
    main()
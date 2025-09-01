import argparse
import os
import os.path as osp
from sconf import Config
from pathlib import Path
import json

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import h5py as h5
import numpy as np

import utils
from datasets import HDF5Data, kor_decompose, uniform_sample, get_ma_val_dataset
from train import setup_language_dependent, setup_cv_dset_loader
from models import MACore



def setup(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--resume', default='/home/dev/dmfont/experiments/checkpoints/250831_05-11-52_test/last.pth')
    parser.add_argument("--img_dir", default='./results')
    parser.add_argument("--config_paths", nargs="+", default=['cfgs/kor.yaml'])
    
    args, left_argv = parser.parse_known_args()
    
    cfg = Config(*args.config_paths, colorize_modified_item=True)
    cfg.argv_update(left_argv)
    
    cfg['data_dir'] = Path(cfg['data_dir'])
    
    return args, cfg



def loader_validation(cfg, content_font="NanumBarunpenR.ttf"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    hdf5_paths = list(cfg['data_dir'].glob("*.hdf5"))
    hdf5_data = HDF5Data(hdf5_paths, transform, language=cfg['language'])
    meta = json.load(open(cfg['data_meta']))
    
    
    # n_chars = 16
    # n_fonts = 16
    
    exception_str = ["\u3131", "\u3132", "\u3134", "\u3137", "\u3138", "\u3139", "\u3141", "\u3142", 
                     "\u3143", "\u3145", "\u3146", "\u3147", "\u3148", "\u3149", "\u314a", "\u314b", 
                     "\u314c", "\u314d", "\u314e", "\u314f", "\u3150", "\u3151", "\u3152", "\u3153", 
                     "\u3154", "\u3155", "\u3156", "\u3157", "\u3158", "\u3159", "\u315a", "\u315b", 
                     "\u315c", "\u315d", "\u315e", "\u315f", "\u3160", "\u3161", "\u3162", "\u3163"]
    meta['valid']['chars'] = [char for char in meta['valid']['chars'] if not char in exception_str]
    
    seen_chars   = meta['train']['chars']
    unseen_chars = meta['valid']['chars']
    unseen_fonts = meta['valid']['fonts']

    # allchars = seen_chars + unseen_chars
    # allchars = meta['valid']['chars']
    style_avails = {font: unseen_chars for font in unseen_fonts}
    
    val_dset, collate_fn = get_ma_val_dataset(
        hdf5_data, unseen_fonts, unseen_chars, style_avails, 
        n_max_match=3, transform=transform,
        content_font=content_font, language='kor'
    )
    # valid_loader = DataLoader(val_dset, batch_size=cfg['batch_size'] * 3, shuffle=False,
    #                     num_workers=2, collate_fn=collate_fn)
    valid_loader = DataLoader(val_dset, batch_size=120, shuffle=False,
                        num_workers=2, collate_fn=collate_fn)
    
    return valid_loader, unseen_chars


@torch.no_grad()
def load_model(args, cfg):
    n_comps = kor_decompose.N_COMPONENTS
    n_comp_types = 3  # cho, jung, jong
    
    g_kwargs = cfg.get('g_args', {})
    gen = MACore(
        1, cfg['C'], 1, **g_kwargs, n_comps=n_comps, n_comp_types=n_comp_types,
        language=cfg['language']
    )
    gen.cuda()
    gen.eval()
    
    ckpt = torch.load(args.resume)
    gen.load_state_dict(ckpt['generator_ema'])
    
    return gen


def main():
    loader, unseen_chars  = loader_validation(cfg)
    valid_comp_ids = [kor_decompose.decompose(char) for char in unseen_chars]
    
    gen     = load_model(args, cfg)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    all_hangul_chars = [chr(code) for code in range(0xAC00, 0xD7A4)]
    for idx, char in enumerate(all_hangul_chars):
        print(f"{char} \t {idx+1}/{len(all_hangul_chars)}")
        
        trg_comp_id = kor_decompose.decompose(char)
        
        # if trg_comp_id in valid_comp_ids:
        trg_comp_id = torch.tensor(trg_comp_id, device='cuda')
        trg_comp_id = trg_comp_id.unsqueeze(0)

        style_ids = torch.tensor([0], device='cuda:0')
        style_comp_ids = trg_comp_id
        
        for hdf5_path in hdf5_paths:
            with h5.File(hdf5_path, 'r') as f:
                chars       = f['dataset']['chars'][:]
                char2idx    = {chr(ch): i for i, ch in enumerate(chars)}
                try:
                    cidx    = char2idx[char]
                except KeyError:
                    print(f'missing characters: {char}')
                    continue
                
                style_image = f['dataset']['images'][cidx]
                style_image  = transform(style_image)
                style_image  = style_image.unsqueeze(0)
                style_image  = torch.tensor(style_image, device='cuda:0')
                
                gen.encode_write(style_ids, style_comp_ids, style_image)

                trg_ids = style_ids
                gen_out = gen.read_decode(trg_ids, trg_comp_id)
                gen_out = gen_out[0]
                
                utils.save_tensor_to_image(gen_out, osp.join(save_dir, f"{char}.png"))
        
        # for path in hdf5_paths:
        #     with h5.File(path, 'r') as f:
        #         chars       = f['dataset']['chars'][:]
        #         char2idx    = {chr(ch): i for i, ch in enumerate(chars)}
        #         try:
        #             cidx    = char2idx[char]
        #         except KeyError:
        #             print(f'missing characters: {char}')
        #             continue
                    
        #         style_imgs  = f['dataset']['images'][cidx]
        #         style_imgs  = transform(style_imgs)
        #         style_imgs  = style_imgs.unsqueeze(0)
        #         style_imgs  = torch.tensor(style_imgs, device='cuda:0')
                
        #         gen.encode_write(style_ids, style_comp_ids, style_imgs)
        #         trg_ids = style_ids
        #         gen_out = gen.read_decode(trg_ids, trg_comp_id)
        #         gen_out = gen_out[0]

        #         utils.save_tensor_to_image(gen_out, f"/home/dev/dmfont/results/Jinbeop/{char}.png")
        #         break

    # COMPONENT_RANGE = (int('3131', 16), int('3163', 16))  # kr 자음/모음
    # JONG_LIST = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ',
    #             'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ',
    #             'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    # N_JONG = len(JONG_LIST)
    
    # exception_str = ["\u3131", "\u3132", "\u3134", "\u3137", "\u3138", "\u3139", "\u3141", "\u3142", 
    #                     "\u3143", "\u3145", "\u3146", "\u3147", "\u3148", "\u3149", "\u314a", "\u314b", 
    #                     "\u314c", "\u314d", "\u314e", "\u314f", "\u3150", "\u3151", "\u3152", "\u3153", 
    #                     "\u3154", "\u3155", "\u3156", "\u3157", "\u3158", "\u3159", "\u315a", "\u315b", 
    #                     "\u315c", "\u315d", "\u315e", "\u315f", "\u3160", "\u3161", "\u3162", "\u3163"]
    # except_jong = {}
    # for exception in exception_str:
    #     char_code = ord(exception)
    #     char_code -= COMPONENT_RANGE[0]
    #     jong = char_code % N_JONG
    #     except_jong[exception] = jong
    
    
    # for loaded_dataset in loader:
    #     style_ids, style_comp_ids, style_imgs, trg_ids, trg_comp_ids, _ = loaded_dataset
        
    #     style_ids       = style_ids.cuda()
    #     style_comp_ids  = style_comp_ids.cuda()
    #     style_imgs      = style_imgs.cuda()
    #     trg_ids         = trg_ids.cuda()
    #     # trg_comp_ids    = trg_comp_ids.cuda()
    #     with torch.no_grad():
    #         gen.encode_write(style_ids, style_comp_ids, style_imgs)
    #         # gen.encode_write(style_ids[6].unsqueeze(0), style_comp_ids[6].unsqueeze(0), style_imgs[6].unsqueeze(0))
        
    #     for idx, char in enumerate(all_hangul_chars):
    #         print(f"{char} \t {idx+1}/{len(all_hangul_chars)}")
            
    #         trg_comp_id = kor_decompose.decompose(char)
    #         trg_comp_id = torch.tensor(trg_comp_id, device='cuda')
    #         trg_comp_id = trg_comp_id.unsqueeze(0)
            
    #         # if any([trg_comp in style_comp_ids for trg_comp in trg_comp_id[0]]):
    #         try:
    #             gen_out = gen.read_decode(trg_ids[0].unsqueeze(0), trg_comp_id)
    #             gen_out = gen_out[0]
    #             utils.save_tensor_to_image(gen_out, f"/home/dev/dmfont/results/Jinbeop/{char}.png")
    #         except KeyError:
    #             jong = trg_comp_id.cpu().numpy()[0][-1]
    #             style_jong = style_comp_ids.cpu().numpy()[:, -1]
    #             is_target = len(np.where(style_jong == jong)[0]) != 0
    #             if is_target:
    #                 a=1
    #             else:
    #                 a=1
    #                 continue

if __name__ == '__main__':
    args, cfg = setup('custom handwriting model evaluation')

    # hdf5_paths = list(cfg['data_dir'].glob("*.hdf5"))
    hdf5_paths = ["/home/dev/dmfont/datasets/hdf5/JinbeopUnhae_ver2.hdf5"]
    save_dir = "/home/dev/dmfont/results/Jinbeop"
    
    os.makedirs(save_dir, exist_ok=True)
    
    main()
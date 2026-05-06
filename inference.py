
"""
Created on Mon Feb  9 09:43:56 2026

@author: Administrator
"""


from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from Corrosion_model import DeepLab, DeepLab_CBAM
from Segformer_ls import Segformer_ls
from metrics import SegmentationMetrics
import time
def get_time(f):
    def inner(*arg, **kwarg):
        s_time = time.time()
        res = f(*arg, **kwarg)
        e_time = time.time()
        print('耗时：{}秒'.format(e_time - s_time))
        return res
    return inner

class Predictor:
    def __init__(
        self,
        save_path: str,
        state: str = 'pred',
        model: str = 'DeepLab',
        mask_path: str = None,
        mask_dir: str = None,
        model_pt_path: str = None,
        crop_size: int = 512,
        max_patches: int = 8
    ):

        self.model_pt = model_pt_path
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.state = state.lower()
        self.mask_path = mask_path
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nc = 2
        self.crop_size = crop_size
        self.max_patches = max_patches
        model_name = str(model).lower()
        self.overlay_alpha = 0.32
        self.mask_fg_color = (248, 248, 248)
        if 'deeplab_cbam' in model_name:
            self.model = DeepLab_CBAM(num_classes=self.nc).to(self.device)
        elif 'deeplab' in model_name:
            self.model = DeepLab(num_classes=self.nc).to(self.device)
        elif 'segformer_ls' in model_name:
            self.model = Segformer_ls(
                encoder_name='mit-b2',
                classes=2
            ).to(self.device)
        else:
            self.model = smp.Segformer(
                encoder_name='mit_b2',
                classes=2
            ).to(self.device)

        self.state_dict = torch.load(self.model_pt, map_location=self.device)
        self.model.load_state_dict(self.state_dict)
        self.model.eval()
        self.metrics = SegmentationMetrics(num_classes=self.nc, ignore_index=255)
        self.current_img_path = None
        self.current_name = None
        self.current_mask_path = None

    def predict(self, img_path: str, mask_path: str = None):
        self.current_img_path = Path(img_path)
        self.current_name = self.current_img_path.stem

        if self.state == 'val':
            self.current_mask_path = self._resolve_mask_path(img_path, mask_path)
        else:
            self.current_mask_path = None

        self.img = self._preprocess(str(self.current_img_path))

        with torch.no_grad():
            self._sliding_window_inference(self.img)

        self._save()

        if self.state == 'val':
            one_metrics = SegmentationMetrics(num_classes=self.nc, ignore_index=255)
            one_metrics.update(self.preds.cpu(), self.mask.cpu())
            one_result = one_metrics.get_results()

            self.metrics.update(self.preds.cpu(), self.mask.cpu())

            print(f"[VAL] {self.current_img_path.name}: {one_result}")
            return one_result

        print(f"[PRED] {self.current_img_path.name} done.")
        return None
    
    @get_time
    def predict_folder(self, img_dir: str, exts=None):

        if exts is None:
            exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

        img_dir = Path(img_dir)


        img_list = sorted([
            p for p in img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        ])


        for idx, img_path in enumerate(img_list, start=1):
            self.predict(str(img_path))
        if self.state == 'val':
            print(self.metrics.get_results())

    def _resolve_mask_path(self, img_path: str, mask_path: str = None):

        if mask_path is not None:
            mask_file = Path(mask_path)
            return mask_file

        if self.mask_path is not None:
            mask_file = Path(self.mask_path)
            return mask_file

        if self.mask_dir is not None:
            img_stem = Path(img_path).stem
            mask_file = self.mask_dir / f'{img_stem}.txt'
            return mask_file


    def _preprocess(self, img_path: str):
        img_bgr = cv2.imread(img_path)

        self.img_raw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(self.img_raw.transpose((2, 0, 1))).float().div(255).unsqueeze(0).to(self.device)
        return img  # [1, C, H, W]

    def _save(self):
        assert self.state in ('val', 'pred')
    
        self._img_raw_save()
        self._mask_save()
        self._overlay_save()
    
        if self.state == 'val':
            self._mask_raw_save()
    def _save_file(self, suffix: str, ext: str = '.png'):
        return str(self.save_path / f'{self.current_name}_{suffix}{ext}')
    
    def _img_raw_save(self):
        cv2.imwrite(
            self._save_file('img_raw'),
            cv2.cvtColor(self.img_raw, cv2.COLOR_RGB2BGR)
        )

    def _mask_save(self):
        pred_mask = self.pred_label[0].cpu().numpy().astype(np.uint8)
    
        h, w = pred_mask.shape
        mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
        mask_vis[pred_mask == 1] = self.mask_fg_color
    
        cv2.imwrite(self._save_file('mask_pred'), mask_vis)
    def _overlay_save(self):
        img_bgr = cv2.cvtColor(self.img_raw, cv2.COLOR_RGB2BGR).copy().astype(np.float32)
        pred_mask = self.pred_label[0].cpu().numpy().astype(np.uint8)
    
        overlay = img_bgr.copy()
        white_layer = np.full_like(img_bgr, self.mask_fg_color, dtype=np.float32)
    
        mask_bool = (pred_mask == 1)
    
        overlay[mask_bool] = (
            (1.0 - self.overlay_alpha) * img_bgr[mask_bool]
            + self.overlay_alpha * white_layer[mask_bool]
        )
    
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        self.overlay = overlay
    
        cv2.imwrite(self._save_file('overlay'), self.overlay)
    def _mask_raw_save(self):
        h, w, _ = self.img_raw.shape
        label_list = []
    
        with open(self.current_mask_path, 'r', encoding='utf-8') as f:
            cont = f.read().split('\n')
            for line in cont:
                if len(line) == 0:
                    continue
                else:
                    l0 = line.split('  ')
                    pts = []
                    for i in range(1, len(l0), 2):
                        pts_x = np.array(eval(l0[i]) * w).astype(np.int32)
                        pts_y = np.array(eval(l0[i + 1]) * h).astype(np.int32)
                        pts.append(np.array([pts_x, pts_y]))
                    dic1 = {l0[0]: np.array(pts)}
                    label_list.append(dic1)
    
        mask = np.zeros((h, w), dtype=np.uint8)
        for pts_dic in label_list:
            pts = pts_dic['0']
            blended = cv2.fillPoly(mask, [pts], 255)
            mask = cv2.bitwise_or(mask, blended)
    
        self.mask = torch.from_numpy(mask).unsqueeze(0).div(255).long()
    
        mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
        mask_vis[mask == 255] = self.mask_fg_color
    
        cv2.imwrite(self._save_file('mask_raw'), mask_vis)
    def _get_positions(self, length: int):
        if length <= self.crop_size:
            return [0]

        positions = list(range(0, length - self.crop_size + 1, self.crop_size // 2))
        if positions[-1] != length - self.crop_size:
            positions.append(length - self.crop_size)
        return positions

    def _sliding_window_inference(self, img):
        crop_size = self.crop_size
        nc = self.nc
        device = self.device

        _, _, h, w = img.shape
        orig_h, orig_w = h, w
        pad_h = max(crop_size - h, 0)
        pad_w = max(crop_size - w, 0)
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)

        _, _, h_pad, w_pad = img.shape

        preds_map = torch.zeros((1, nc, h_pad, w_pad), device=device)
        count_map = torch.zeros((1, 1, h_pad, w_pad), device=device)

        patches = []
        coords = []

        y_positions = self._get_positions(h_pad)
        x_positions = self._get_positions(w_pad)

        for y1 in y_positions:
            for x1 in x_positions:
                y2 = y1 + crop_size
                x2 = x1 + crop_size

                patch = img[:, :, y1:y2, x1:x2]
                patches.append(patch)
                coords.append((y1, y2, x1, x2))

                if len(patches) == self.max_patches:
                    patch_batch = torch.cat(patches, dim=0)
                    preds_batch = self.model(patch_batch)

                    for i, (yy1, yy2, xx1, xx2) in enumerate(coords):
                        preds_map[:, :, yy1:yy2, xx1:xx2] += preds_batch[i:i + 1]
                        count_map[:, :, yy1:yy2, xx1:xx2] += 1

                    patches.clear()
                    coords.clear()

        if len(patches) > 0:
            patch_batch = torch.cat(patches, dim=0)
            preds_batch = self.model(patch_batch)

            for i, (yy1, yy2, xx1, xx2) in enumerate(coords):
                preds_map[:, :, yy1:yy2, xx1:xx2] += preds_batch[i:i + 1]
                count_map[:, :, yy1:yy2, xx1:xx2] += 1

        preds_map = preds_map / count_map.clamp_min(1.0)
        
        self.preds = preds_map[:, :, :orig_h, :orig_w]
        self.pred_prob = torch.softmax(self.preds, dim=1)   # [B, nc, H, W]
        self.pred_label = torch.argmax(self.pred_prob, dim=1)  # [B, H, W]


if __name__ == '__main__':
    save_path = r'Test_img'
    img_dir = r'D:val'
    mask_dir = r'val'
    model_pt_path =r'model.pt'

    predictor = Predictor(
        save_path=save_path,
        model='segformer_ls',
        state='val',  
        mask_dir=mask_dir,
        model_pt_path=model_pt_path
    )

    # single predict
    # predictor.predict(img_path=r'D:\Users_files\Test\DJI_0460.JPG')

    # folder predict
    predictor.predict_folder(img_dir=img_dir)
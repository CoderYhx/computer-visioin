import argparse

import os.path as path
import os

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
import torch.nn.functional as F
from models import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr
from skimage.metrics import structural_similarity


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-root', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = SRCNN().to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()


    imgs = []
    results = []
    img_root = args.image_root
    imgs = os.listdir(img_root)
    for img in imgs:
        img_file = path.join(img_root, img)

        result_item = [img]

        image = pil_image.open(img_file).convert('RGB')

        image_width = (image.width // args.scale) * args.scale
        image_height = (image.height // args.scale) * args.scale
        image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
        image = image.resize((image.width // args.scale, image.height // args.scale), resample=pil_image.BICUBIC)
        image = image.resize((image.width * args.scale, image.height * args.scale), resample=pil_image.BICUBIC)
        image.save(img_file.replace('.', '_bicubic_x{}.'.format(args.scale)))

        image = np.array(image).astype(np.float32)
        ycbcr = convert_rgb_to_ycbcr(image)

        y = ycbcr[..., 0]
        y /= 255.
        y = torch.from_numpy(y).to(device)
        y = y.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            preds = model(y).clamp(0.0, 1.0)

        print(y.shape)
        print(preds.shape)
        psnr = calc_psnr(y, preds)
        # result_item.append(round(psnr.cpu().item(), 4))
        result_item.append('PSNR: {:.2f}'.format(psnr))
        print('PSNR: {:.2f}'.format(psnr))

        y_np = (y.cpu().numpy().squeeze(0).squeeze(0)*255).astype(np.uint8)
        # y_np = (np.random.rand(510,510)*255).astype(np.uint8)
        preds_np = (preds.cpu().numpy().squeeze(0).squeeze(0)*255).astype(np.uint8)
        ssim = structural_similarity(y_np,preds_np)
        print('SSIM: {:.4f}'.format(ssim))
        # result_item.append(round(ssim, 4))
        result_item.append('SSIM: {:.4f}'.format(ssim))

        preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

        output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
        output = pil_image.fromarray(output)
        output.save(img_file.replace('.', '_srcnn_x{}.'.format(args.scale)))

        results.append(result_item)

    result_txt = "result.txt"

    with open(result_txt, "w") as f:
        for res in results:
            f.write(f"{res[0]}  {res[1]}  {res[2]} \n")




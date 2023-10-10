import numpy as np
import glob
import cv2

def _prepare_data(pred: np.ndarray, gt: np.ndarray) -> tuple:
    gt = gt > 128
    # im2double, mapminmax
    pred = pred / 255
    if pred.max() != pred.min():
        pred = (pred - pred.min()) / (pred.max() - pred.min())
    return pred, gt
def MAE_Loss(act, pred):
    diff = act - pred
    diff_abs = np.absolute(diff)
    mae = np.sum(diff_abs)/act.size
    return mae

testnumb=0
mae_s=0.0
test_image_dir = 'E:/data/CRACK dataset/CRACK500/testcrop/label/'
test_pred_dir='E:/saved_models/ablation experiment/ssim/crop/'
label_ext = '.png'
mask_name_list = glob.glob(test_image_dir+'*'+label_ext)
pred_name_list= glob.glob(test_pred_dir+'*'+label_ext)
for i in range(len(mask_name_list)):
    mask_path = mask_name_list[i]
    pred_path = pred_name_list[i]
            #print(pred_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    mask,pred=_prepare_data(pred,mask)
    mae_s+=MAE_Loss(mask,pred)
    testnumb+=1
print(mae_s/testnumb)
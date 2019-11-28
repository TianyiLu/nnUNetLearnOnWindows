#import torch
#a  = torch.cuda.is_available()
#print(a)


import nibabel as nib
import skimage.io as io
import numpy as np
img=nib.load('/nnUNet/nnunet/nnUNet_result/la_002.nii')
img_arr=img.get_data()

img_arr=np.squeeze(img_arr)
list_a = img_arr.tolist()

print(max(list_a))

io.imshow(img_arr[1])
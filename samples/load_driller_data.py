__author__ = 'shawnle'
__email__ = 'letrungson1@gmail.com'
__version__ = 'v1.5'

from pathlib import Path
import numpy as np
import os

import cv2
import skimage.io
import matplotlib.pyplot as plt

import transforms3d as tf
from transforms3d.quaternions import quat2mat, mat2quat
from transforms3d.axangles import axangle2mat, mat2axangle

class config():
  def __init__(self):
    self.PATH_ROOT = "D:\\SL\\test_tf2_data\\LINEMOD.tar\\"
    self.driller_data = self.PATH_ROOT + "driller\\driller\\data\\"

def parse_rots(rot_files):
  all_rots = np.zeros(shape=(len(rot_files), 3, 3), dtype=np.float32 )
  # print(all_rots.shape)
  for rot in rot_files:
    # print(rot)
    i=0
    with rot.open() as f:
      j=0
      for line in f:        
        line = line.split()

        if len(line) == 2 :
          continue  
        else: 
          all_rots[i,j,:] = line
          j=j+1
      
    i=i+1
    
  return all_rots

def parse_tras_by_len(PATH_ROOT, length):

  all_tras = np.zeros(shape=(length, 3, 1), dtype=np.float32 )
  for i in range(length):
    p0 = os.path.abspath(PATH_ROOT)
    # file_name ='./driller/data/tra' + str(i) + '.tra'
    file_name = os.path.join(p0, 'driller', 'driller', 'data', 'tra' + str(i) + '.tra')
    p = Path(file_name)

    with p.open() as f:
      j=0
      for line in f:        
        line = line.split()

        if len(line) == 2 :
          continue  
        else: 
          all_tras[i,j,:] = line
          j=j+1
    
  return all_tras

def parse_rots_by_len(DATA_ROOT, length):

  all_quats = np.zeros(shape=(length, 4), dtype=np.float32 )
  all_rots = np.zeros(shape=(length, 3, 3), dtype=np.float32 )
  all_axang = np.zeros(shape=(length, 3))
  print(all_rots.shape)
  for i in range(length):
    #file_name ='./driller/data/rot' + str(i) + '.rot'
	  #file_name = '\\driller\\data\\rot' + str(i) + '.rot'
    p0 = os.path.abspath(DATA_ROOT)
    file_name = os.path.join(p0, 'driller', 'driller', 'data', 'rot' + str(i) + '.rot')
    # print(file_name)
	
    p = Path(file_name)

    with p.open() as f:
      j=0
      for line in f:        
        line = line.split()

        if len(line) == 2 :
          continue  
        else: 
          all_rots[i,j,:] = line
          j=j+1

    all_quats[i,:] = mat2quat(all_rots[i,:,:])
    direc, angle = mat2axangle(all_rots[i,:,:])
    all_axang[i,:] = direc * angle

  return all_rots, all_quats, all_axang

def read_masks_and_rgbs(PATH_ROOT, length):

  def get_boundingbox(image):

    image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    r0 = image.shape[0]
    r1 = 0
    c0 = image.shape[1]
    c1 = 0

    y, x, z = np.where(image > 0)
    x_min_id = np.argmin(x)
    x_max_id = np.argmax(x)
    y_min_id = np.argmin(y)
    y_max_id = np.argmax(y)
    x_min = x[x_min_id]
    x_max = x[x_max_id]
    y_min = y[y_min_id]
    y_max = y[y_max_id]

    return (x_min, x_max, y_min, y_max)
  
  masks = []
  bbs = []
  rgbs = []
  for i in range(length):

    p0 = os.path.abspath(PATH_ROOT)    
    # file_name ='./LINEMOD/driller/mask/{:04d}'.format(i) + '.png'
    file_name = os.path.join(p0, 'LINEMOD', 'driller', 'mask', '{:04d}'.format(i) + '.png' )
    # print ('file name = ', file_name)
    # mask = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    mask = file_name
    file_name = os.path.join(p0, 'driller', 'driller', 'data', 'color{:d}.jpg'.format(i) )
    # rgb = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    rgb = file_name

    # print('cwd = ', os.getcwd())
    #full_path = os.path.join(os.getcwd(), 'LINEMOD', 'driller', 'mask', '{:04d}'.format(i) + '.png')
    # print('full path = ',full_path)
    # image = skimage.io.imread(full_path)
    # image = skimage.io.imread(file_name)

    # cv2.imshow("mask", image)
    # cv2.waitKey(1)
    # print(image.shape)
#     print(mask)

    bb = get_boundingbox(mask)

    rgbs.append(rgb)
    masks.append(mask) 
    bbs.append(bb)

  return masks, rgbs, bbs


def compute_centers(all_rots, all_tras, K, H=480, W=640):

  print("len rots", len(all_rots))

  origin = np.zeros((4,1))
  origin[-1] = 1.
  centers = []
  for i in range(len(all_rots)):

    T = np.concatenate((all_rots[i], all_tras[i]), axis=1)
    center = np.matmul(K, np.matmul(T, origin))
    center /= center[-1]

    assert 0 <= center[0] and center[0] < W and 0 <= center[1] and center[1] < H

    centers.append(center)

  return centers

def nonfill_rectangle(img, p0, p1, color, line_thick):

  assert type(p0) is tuple and type(p1) is tuple
  x0 = p0[0]
  y0 = p0[1]
  x1 = p1[0]
  y1 = p1[1]
  cv2.line(img, p0, (x1,y0), color, line_thick)
  cv2.line(img, p0, (x0,y1), color, line_thick)
  cv2.line(img, p1, (x1,y0), color, line_thick)
  cv2.line(img, p1, (x0,y1), color, line_thick)

  return img

def verify_centers_bbs(all_centers, all_masks, all_bbs):
  
  for i in range(10):
    c_id = np.random.randint(0, len(all_centers))
    c = all_centers[c_id]
    mask = cv2.imread(all_masks[c_id], cv2.IMREAD_UNCHANGED)
    bb = all_bbs[c_id]

    cv2.circle(mask, (c[0], c[1]), 3, (0,0,255), -1)
    mask = nonfill_rectangle(mask, (bb[0],bb[2]), (bb[1],bb[3]), (0,255,0), 1)
    cv2.imshow("center projection", mask)
    cv2.waitKey(0)
    cv2.imwrite("val_{}.jpg".format(i), mask)

def load_driller_data():
# if __name__ == '__main__':

  cfg = config()

  # p = Path('./driller/data')
  p = Path(cfg.driller_data)
  # print(p)

  rot_files = list(p.glob('rot*.rot'))
  # print (rot_files)
  # print (len(rot_files))

  tra_files = list(p.glob('tra*.tra'))
  # print (tra_files)
  # print (len(tra_files))

  # camera intrinsics
  fx=572.41140
  fy=573.57043
  px=325.26110
  py=242.04899
  K = np.identity(3)
  K[0,0] = fx
  K[1,1] = fy
  K[0,2] = px
  K[1,2] = py
  print("K=", K)
  # exit()
    
  all_rots, all_quats, all_axang = parse_rots_by_len(cfg.PATH_ROOT, len(rot_files))
  # print('results = ', all_rots[10,:,:])
  # print('some quats = ', all_quats[:10,:])
  # print('some axang = ', all_axang[:10,:])  

  all_tras = parse_tras_by_len(cfg.PATH_ROOT, len(rot_files))
  # print('results = ', all_tras[1,:,:])  

  all_masks, all_rgbs, all_bbs = read_masks_and_rgbs(cfg.PATH_ROOT, len(rot_files))

  one_image = cv2.imread(all_masks[0], cv2.IMREAD_UNCHANGED)
  H = one_image.shape[0]
  W = one_image.shape[1]
  all_centers = compute_centers(all_rots, all_tras, K, H, W)
  verify_centers_bbs(all_centers, all_masks, all_bbs)  

  return {'rots' : all_rots,
          'quats' : all_quats,
          'axang' : all_axang,
          'tras' : all_tras,
          'masks' : all_masks,
          'rgbs' : all_rgbs,
          'bbs' : all_bbs,
          'centers' : all_centers,
          'intrinsics' : K,
          'image_H' : H,
          'image_W' : W}
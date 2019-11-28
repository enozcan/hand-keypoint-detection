import torchvision
import torch
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import json
import numpy as np
from PIL import Image
from skimage import io, transform
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as F

#drive.mount('/drive')
#drive_base_path = "/drive/My Drive/"
root_dir = 'manual_train/'

'''
Method to draw annotated keypoints on the corresponding image.
A helper method to visualize the data, not necessary to the model.
'''
def show_hand_keypoints(image, hand_keypoints):
    """Show image with hand keypoints"""
    plt.figure(figsize = (30, 8.75))
    plt.imshow(image)
    plt.scatter(hand_keypoints[:, 0], hand_keypoints[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated|

'''
A trial method to draw calculated hand box on the corresponding image.
Used to test our hand box success, not necessary to the model.
'''
def draw_hand_box(image, box):
  """Show image with hand box"""
  fig,ax = plt.subplots(1)
  fig.set_size_inches(15, 10, forward=True)
  ax.imshow(image)
  rect = patches.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],linewidth=1,edgecolor='r',facecolor='none')
  ax.add_patch(rect)
  plt.show()

'''
Since the model expects the box of the interest object and our dataset does not 
contain hand box but instead the hand center, we calculate the approximate
hand box here.
'''
def get_hand_box(hand_keypoints):
  mins, maxs = np.amin(hand_keypoints, axis=0), np.amax(hand_keypoints, axis=0)
  x1, x2, y1, y2 = mins[0], maxs[0], mins[1], maxs[1]
  marginx = (x2 - x1) * 0.15 # 0.15 * box_width margin
  marginy = (y2 - y1) * 0.15 # 0.15 * box_height margin
  return(np.asarray([int(x1 - marginx), int(y1 - marginy) , int(x2 + marginx), int(y2 + marginy)]))

'''
To load dataset into the model, we need the images in each batch 
have the same size. This rescaler is passed to HandKeypointDataset.
'''
class Rescale(object):
    '''Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    '''

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, hand_keypoints, hand_box, label = sample['image'], sample['keypoints'], sample['boxes'], sample['labels']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        rescaled_image = transform.resize(image, (new_h, new_w)) # Rescale image

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        for i in range(len(hand_keypoints)):
          hand_keypoints[i] = hand_keypoints[i] * [new_h / h, new_w / w, 1] # Rescale hand keypoints as well
        hand_box = (hand_box * [new_h / h,  new_w / w, new_h / h, new_w / w]).astype(int) # Rescale hand box as well
        
       
        return {'image': rescaled_image, 'keypoints': hand_keypoints, 'boxes': hand_box, 'labels': label}

'''
When rescale object is used, only one dimention (height or width) becomes
equal to the defined size. In order to make all images have the same width and
height, this class performs random crop such that all the images have the same
width and height (in square format) before feeding the model. 

This cropper is also passed to HandKeypointDataset.
'''
class RandomCrop(object):
    '''Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    '''
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, hand_keypoints, hand_box, label = sample['image'], sample['keypoints'], sample['boxes'], sample['labels']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        # Decide cropping points concerning the hand box remains
        # in the new image
    
        random_width_interval = (np.maximum(0, hand_box[0][2] - new_w), np.minimum(hand_box[0][0], w - new_w));
        random_height_interval = (np.maximum(0, hand_box[0][3] - new_h), np.minimum(hand_box[0][1], h - new_h));
        
        top, left = 0, 0
        if(random_height_interval[1] != 0):
          top = np.random.randint(*random_height_interval) # new starting x coordinate
        if(random_width_interval[1] != 0):
          left = np.random.randint(*random_width_interval) # new starting y coordinate

        image = image[top: top + new_h, left: left + new_w]

        for i in range(len(hand_keypoints)):
          hand_keypoints[i] = hand_keypoints[i] - [left, top, 0]
        hand_box = hand_box - [left,  top,left, top]

        return {'image': image, 'keypoints': hand_keypoints, 'boxes': hand_box, 'labels': label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, hand_keypoints, hand_box, label = sample['image'], sample['keypoints'], sample['boxes'], sample['labels']
        # Swap color axis. 
        # The format is needed to feed the model.
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).cuda().type(torch.float32),
                'keypoints': torch.from_numpy(hand_keypoints).cuda().type(torch.float32),
                'boxes': torch.from_numpy(hand_box).cuda().type(torch.float32),
                'labels': label.cuda().type(torch.int64)}

'''
The function reads data from the directory and package the image name 
and other annotations. Output is like the following:

[ ( image_file_name, {boxes,labels,keyponts} ) ]
'''
def get_hand_keypoints_frame(root_dir):
    FRAME_SIZE = 1000; # Read this much data. Have to read all of them for training. 

    files = sorted([f for f in os.listdir(root_dir) if f.endswith('.json')])
    hand_keypoints_frame = []
    for f in files[:FRAME_SIZE]:
        with open(root_dir+f, 'r') as fid:
            dat = json.load(fid)

        # Each file contains 1 hand annotation, with 21 points in
        # 'hand_pts' of size 21x3, following this scheme:
        # https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md#hand-output-format
        # The 3rd column is 1 if valid.
        
        # Since we only try to detect hand keypoints rather than body keypoints etc.
        # we have only one label here.
        label = torch.Tensor([0]).view(1) 

        # Hand keypoints
        hand_keypoints = np.array(dat['hand_pts'])
        
        # Hand box is not present in the dataset. Obtain it from the declared method above.
        hand_box = get_hand_box(hand_keypoints)
        
        # Add image file name instead of the image itself.
        dot_idx = f.rfind(".")
        img_name = f[:dot_idx] + '.jpg'

        # Since this is the ground truth, following statement makes no sense.
        invalid = hand_keypoints[:,2]!=1

        target = {}
        target['boxes'] = hand_box
        target['labels'] = label
        target['keypoints'] = hand_keypoints

        hand_keypoints_frame.append((img_name, target))
    return hand_keypoints_frame

class HandKeypointsDataset(Dataset):
    """
    Hand Keypoints dataset defined in the doc. See the doc here:
    https://pytorch.org/docs/stable/torchvision/models.html
    
    * boxes (FloatTensor[N, 4]): 
    the ground-truth boxes in [x1, y1, x2, y2] format, with values between 0 and H and 0 and W

    * labels (Int64Tensor[N]): 
    the class label for each ground-truth box 

    * keypoints (FloatTensor[N, K, 3]): 
    the K keypoints location for each of the N instances, in the format [x, y, visibility],
     where visibility=0 means that the keypoint is not visible
    
    """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # hand_keypoints_frame = [ ( image_file_name, {boxes,labels,keyponts} ) ]
        self.hand_keypoints_frame = get_hand_keypoints_frame(root_dir) # Contains the training data
        
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.hand_keypoints_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.hand_keypoints_frame[idx][0])
        image = io.imread(img_name) # Read image 
        
        target = self.hand_keypoints_frame[idx][1]
        
        sample = {'image': image, 'keypoints': target['keypoints'], 'boxes': target['boxes'].reshape(1, -1), 'labels': torch.Tensor([0]).view(1)}
        if self.transform:
          sample = self.transform(sample)

        # sample = {'image': sample['image'], 'target': {'keypoints': sample['keypoints'],  'boxes': sample['boxes'], 'labels': sample['labels']}}
        return sample

#Draw rect on image

#root_dir = drive_base_path + 'hand_labels/manual_train/'
#hand_label_dataset = HandKeypointsDataset(root_dir, transform=transforms.Compose([Rescale(350),RandomCrop(349),ToTensor()]))

#item = hand_label_dataset.__getitem__(55)
#draw_hand_box(item['image'].permute(1, 2, 0), item['hand_box'])

#show_hand_keypoints(item['image'].permute(1, 2, 0), item['hand_keypoints'])

model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True, num_keypoints=17)
for child in model.children():
  for param in child.parameters():
    param.requires_grad = False # keep these network parts as they are.

# Define new predictor for 21 hand keypoints.
model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(512, 21) 
model.roi_heads.keypoint_predictor.requires_grad = True
for child in model.roi_heads.keypoint_predictor.children():
  for param in child.parameters():
    param.requires_grad = True

BATCH_SIZE = 10
hand_label_dataset = HandKeypointsDataset(root_dir, transform=transforms.Compose([Rescale(360),RandomCrop(350),ToTensor()]))
loader = DataLoader(hand_label_dataset, batch_size=BATCH_SIZE, shuffle=True)

'''
# Make sure the input format is true.

for i in range(len(hand_label_dataset)):
    sample = hand_label_dataset[i]
    print(i, sample['image'].size(), sample['keypoints'].size())
    if i == 3:
       break
for idx_batch, sample_batched in enumerate(loader):
    print(idx_batch, sample_batched['image'].size(),
          sample_batched['keypoints'].size(),
          sample_batched['boxes'].size(),
          sample_batched['labels'].size())
'''

model.cuda()
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005,momentum=0.9, weight_decay=0.0005)
epoch_loss = []

for idx_batch, sample_batched in enumerate(loader):
  #torch.split(tensor, split_size_or_sections, dim=0)
  #targets = [sample_batched['keypoints'], sample_batched['boxes'], sample_batched['labels']]
  #we need to convert targets to list of dicts

  targets = []
  for i in range(sample_batched['keypoints'].size(0)):
    temp = {'keypoints': sample_batched['keypoints'][i],
            'boxes': sample_batched['boxes'][i],
            'labels': sample_batched['labels'][i],
            }
    targets.append(temp)

  y = model(sample_batched['image'], targets)
  
  total_loss = 0
  for k, v in y.items():
    total_loss += v
    
  optimizer.zero_grad()
  total_loss.requires_grad = True
  total_loss.backward()
  epoch_loss.append(total_loss)
  optimizer.step()
  
  if (idx_batch+1)%100 == 0:
    print("Steps : {} Mean Loss: {}".format(idx_batch+1, torch.mean(torch.stack(epoch_loss))))

torch.mean(torch.stack(epoch_loss))
'''
# Alternative training loop
model.eval()
model.cuda()
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005,momentum=0.9, weight_decay=0.0005)
max_steps = 100
epoch_loss = []
for i in range(max_steps):
    idx = np.random.choice(10, size=BATCH_SIZE, replace=True)
    
    batch = []
    images = []

    # Create the batch randomly for trial.
    for j in range(BATCH_SIZE):
      try:
        inp = hand_label_dataset[idx[j]]
      except:
        j = j-1
      for k, v in inp.items():
        if k!="labels":
          inp[k] = v.cuda().type(torch.float32)
        else:
          inp[k] = v.cuda().type(torch.int64)
      
      inp["keypoints"] = inp["keypoints"].unsqueeze(0)
      images.append(inp["image"].cuda().float())
      batch.append(inp)
  
    #target['hand_box'] = target['hand_box'].float().to(device)
    #target['hand_keypoints'] = target['hand_keypoints'].float().to(device)
    y = model(images, batch)
    print(y)
    total_loss = 0
    for k,v in y.items():
      total_loss += v
      optimizer.zero_grad()

    total_loss = y["loss_objectness"]
    
    total_loss.backward()
    epoch_loss.append(total_loss)
    optimizer.step()
    if (i+1)%10 == 0:
      print("Steps : {} Mean Loss: {}".format(i+1, np.mean(epoch_loss)))
'''
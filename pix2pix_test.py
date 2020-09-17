# coding: utf-8

import torch # pytorch import
import torch.nn as nn # pytorch nn 모듈 import
import torchvision.datasets as dsets # torchvision datasets import
import torchvision.transforms as transforms # torchvision transforms import
import numpy as np # numpy 모듈 import
from PIL import Image # 이미지 파일을 불러오고 처리하기 위해 사용한다.
import os # 폴더 리스트 확인 위해 import
import itertools # iterator정의 위해 사용
import matplotlib.pyplot as plt # 그래프 및 그림 출력시 사용

def pil_loader(path): # 이미지를 PIL 형식으로 가져오기 위해 pil loader을 정의한다.
    with open(path, 'rb') as f: # path의 파일을 읽어온다.
        with Image.open(f) as img: # 이미지 파일을 Image로 읽는다.
            return img.convert("RGB") # PIL형식의 이미지를 반환한다.
        
class ImageDataSet(dsets.ImageFolder): # Image Folder에서 데이터 셋을 가져오는Dataset loader을 정의한다.
    def __init__(self, root, transforms): # 생성자 정의 root는 사진이 있는 폴더의 위치이다.
        self.root = root # 사진이 있는 폴더의 위치 저장
        self.transform = transform # transform 지정
        self.loader = pil_loader # pil loader를 통해 PIL을 이용할 것이다.

        self.dir = os.path.join(root, 'train')  # root/color폴더로 지정한다.
        self.image_paths = list(map(lambda x: os.path.join(self.dir, x), os.listdir(self.dir))) # dir폴더 내에 있는 모든 이미지 파일들을 리스트로 가져온다.
        
    def __len__(self): # 길이를 반환하는 함수 정의
        return len(self.image_paths) # 불러온 이미지의 개수를 반환해준다.
    
    def __getitem__(self, index): # 아이템을 받아오는 함수 정의
        # Load Image
        path = self.image_paths[index] # index위치의 이미지 파일을 path로 가져온다.
        image = self.loader(path) # pil loader를 이용해 path의 이미지 파일을 불러온다.
        input = image.convert('L') # 불러온 이미지를 흑백으로 만든다.
        color = image.convert('RGB') # 불러온 이미지를 RGB로 만든다.
        
        input = self.transform(input) # 흑백 이미지를 transform 해준다.
        color = self.transform(color) # 실제 이미지를 transform 해준다.

        return input, color # 입력과 color를 반환해준다.

    
transform_list = [transforms.ToTensor(), # 텐서로 바꿔준다.
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] # 채널별로 평균0.5, std 0.5로 normalize해준다.
transform = transforms.Compose(transform_list)
    
dset_test = ImageDataSet('color_val', transform) # resized_color_data 폴더에서 테스트용 데이터를 불러온다.
test_loader = torch.utils.data.DataLoader(dset_test,
                                           batch_size=1) # 한개씩 불러온다.

class G(nn.Module): # Generator Network를 정의한다.
    def __init__(self): # 생성자를 선언한다.
        super(G, self).__init__() # Model을 이용하기 위해 부모 클래스의 생성자를 호출해준다.
        # U-Net Skip-connection 모델 사용
        self.conv1 = nn.Conv2d(1, 64, 4, 2, 1) # input 1 output 64, kernel size 4, stride 2, padding 1 conv 1 layer
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1) # input 4 output 64, kernel size 4, stride 2, padding 1 conv 2 layer
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1) # input 64 output 256, kernel size 4, stride 2, padding 1 conv 3 layer
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1) # input 256 output 512, kernel size 4, stride 2, padding 1 conv 4 layer
        self.conv5 = nn.Conv2d(512, 512, 4, 2, 1) # input 512 output 512, kernel size 4, stride 2, padding 1 conv 5 layer
        self.conv6 = nn.Conv2d(512, 512, 4, 2, 1) # input 512 output 512, kernel size 4, stride 2, padding 1 conv 6 layer
        self.conv7 = nn.Conv2d(512, 512, 4, 2, 1) # input 512 output 512, kernel size 4, stride 2, padding 1 conv 7 layer
        self.conv8 = nn.Conv2d(512, 512, 4, 2, 1) # input 512 output 512, kernel size 4, stride 2, padding 1 conv 8 layer
        
        self.deconv8 = nn.ConvTranspose2d(512, 512, 4, 2, 1) # input 512 output 512, kernel size 4, stride 2, padding 1 deconv 8 layer
        self.deconv7 = nn.ConvTranspose2d(1024, 512, 4, 2, 1) # input 1024 output 512, kernel size 4, stride 2, padding 1 deconv 7 layer
        self.deconv6 = nn.ConvTranspose2d(1024, 512, 4, 2, 1) # input 1024 output 512, kernel size 4, stride 2, padding 1 deconv 6 layer
        self.deconv5 = nn.ConvTranspose2d(1024, 512, 4, 2, 1) # input 1024 output 512, kernel size 4, stride 2, padding 1 deconv 5 layer
        self.deconv4 = nn.ConvTranspose2d(1024, 256, 4, 2, 1) # input 1024 output 256, kernel size 4, stride 2, padding 1 deconv 4 layer
        self.deconv3 = nn.ConvTranspose2d(512, 128, 4, 2, 1) # input 512 output 128, kernel size 4, stride 2, padding 1 deconv 3 layer
        self.deconv2 = nn.ConvTranspose2d(256, 64, 4, 2, 1) # input 256 output 64, kernel size 4, stride 2, padding 1 deconv 2 layer
        self.deconv1 = nn.ConvTranspose2d(128, 3, 4, 2, 1) # input 128 output 3, kernel size 4, stride 2, padding 1 deconv 1 layer
        
        self.bn_64 = nn.BatchNorm2d(64) # 64 batch normalization 정의
        self.bn_128 = nn.BatchNorm2d(128) # 128 batch normalization 정의
        self.bn_256 = nn.BatchNorm2d(256) # 256 batch normalization 정의
        self.bn_512 = nn.BatchNorm2d(512) # 512 batch normalization 정의
        
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True) # LeakyReLU  a=0.2
        self.relu = nn.ReLU(inplace=True) # ReLU 
        
        self.dropout = nn.Dropout2d(0.5, inplace=True) # 0.5 비율 Dropout
        self.tanh = nn.Tanh() # Tanh
        
    def initialization_weights(self, mean, std): # 가중치 초기화 함수를 정의한다.
        for m in self._modules: # 모델의 모듈(레이어)마다 
            if isinstance(self._modules[m], nn.ConvTranspose2d) or isinstance(self._modules[m], nn.Conv2d): # 모듈이 Convolution이거나 Deconvolution layer이라면
                self._modules[m].weight.data.normal_(mean, std) # 그 모듈의 weight를 mean, std로 정규화 한다.
                self._modules[m].bias.data.zero_() # 그 모듈의 bias를 0으로 초기화한다.

    # forward method
    def forward(self, input): # forward 함수를 정의한다.
        en1 = self.conv1(input) # input(1,256,256) conv1 layer -> en1 (64,128,128)
        en2 = self.bn_128(self.conv2(self.leaky_relu(en1))) # en1 (64,128,128) -> leaky_relu -> conv2 layer -> 128 batch_norm -> en2 (128,64,64)
        en3 = self.bn_256(self.conv3(self.leaky_relu(en2)))# en2(128,64,64) -> leaky_relu -> conv3 layer -> 256 batch_norm -> en3 (256,32,32)
        en4 = self.bn_512(self.conv4(self.leaky_relu(en3))) # en3(256,32,32) -> leaky_relu -> conv4 layer -> 512 batch_norm -> en4 (512,16,16)
        en5 = self.bn_512(self.conv5(self.leaky_relu(en4))) # en4(512,16,16) -> leaky_relu -> conv5 layer -> 512 batch_norm -> en5 (512,8,8)
        en6 = self.bn_512(self.conv6(self.leaky_relu(en5))) # en5(512,8,8) -> leaky_relu -> conv6 layer -> 512 batch_norm -> en6 (512,4,4)
        en7 = self.bn_512(self.conv7(self.leaky_relu(en6))) # en6(512,4,4) -> leaky_relu -> conv7 layer -> 512 batch_norm -> en7 (512,2,2)
        en8 = self.conv8(self.leaky_relu(en7)) # en7(512,2,2) -> leaky_relu -> conv7 layer -> en8 (512,1,1)
        
        de8 = self.dropout(self.bn_512(self.deconv8(self.relu(en8)))) # en8 (512,1,1) -> leaky_relu -> deconv8 layer -> 512 batch_norm -> 0.5 dropout -> de8 (512,2,2)
        de8 = torch.cat((de8,en7), 1) # de8, en7 skip-connection
        de7 = self.dropout(self.bn_512(self.deconv7(self.relu(de8))))  # de8 (1024,2,2) -> leaky_relu -> deconv7 layer -> 512 batch_norm -> 0.5 dropout -> de7 (512,4,4)
        de7 = torch.cat((de7,en6), 1) # de7, en6 skip-connection
        de6 = self.dropout(self.bn_512(self.deconv6(self.relu(de7)))) # de7 (1024,4,4) -> leaky_relu -> deconv6 layer -> 512 batch_norm -> 0.5 dropout -> de6 (512,8,8)
        de6 = torch.cat((de6,en5), 1) # de6, en5 skip-connection
        de5 = self.bn_512(self.deconv5(self.relu(de6))) # de6 (1024,8,8) -> leaky_relu -> deconv5 layer -> 512 batch_norm -> de5 (512,16,16)
        de5 = torch.cat((de5, en4), 1) # de5, en4 skip-connection
        de4 = self.bn_256(self.deconv4(self.relu(de5))) # de5 (1024,16,16) -> leaky_relu -> deconv4 layer -> 256 batch_norm -> de4 (256,32,32)
        de4 = torch.cat((de4, en3), 1) # de4, en3 skip-connection
        de3 = self.bn_128(self.deconv3(self.relu(de4))) # de4 (512,32,32) -> leaky_relu -> deconv3 layer -> 128 batch_norm -> de3 (128,64,64)
        de3 = torch.cat((de3, en2), 1) # de3, en2 skip-connection
        de2 = self.bn_64(self.deconv2(self.relu(de3))) # de3 (256,64,64) -> leaky_relu -> deconv2 layer -> 64 batch_norm -> de2 (64,128,128)
        de2 = torch.cat((de2, en1), 1) # de2, en1 skip-connection
        de1 = self.deconv1(self.relu(de2)) # de2 (128,128,128) -> leaky_relu -> deconv1 layer -> de1 (3,256,256)
        o = self.tanh(de1) # de1 (3,256,256) -> tanh -> o (3, 256, 256)

        return o # o 반환
    
gen = Generator() # Generator Network를 선언한다.
gen.load_state_dict(torch.load('./save_G.pkl')) # 저장된 모델 파일에서 파라미터들을 불러온다.
gen.eval() # 테스트 모드로 바꾼다.

for step, (input, target) in enumerate(test_loader): # train_loader에서 batch단위로 데이터를 가져온다. 
    colorized = gen(input).data.numpy().reshape(3,256,256).transpose(1, 2, 0)# 흑백이미지를 학습된 generator에 넣어 컬러이미지를 생성하고 출력할 수 있도록 reshape후 256,256,3의 shape로 만든다.
    target = target.numpy().reshape(3,256,256).transpose(1, 2, 0) # 실제 이미지를 pyplot에서 출력할 수 있도록 reshape해준 후 256,256,3이 되도록 transpose 해준다..
    fig, ax = plt.subplots(1, 3, figsize=(16, 16)) # 1x3그리드로 subplot해준다.
    for i in range(3):
        ax[i].get_xaxis().set_visible(False) # x축이 보이지 않게 한다.
        ax[i].get_yaxis().set_visible(False) # y축이 보이지 않게 한다.
        
    ax[0].cla() # i,0 그리드의 사진을 클리어한다.
    #pyplot의 imshow는 (256,256,3) 형태의 사진만 출력이 가능하다.
    grayimage = input.reshape(256,256) # 흑백 이미지(1,256,256)을 (256,256)으로 만들어 준다.
    ax[0].imshow((grayimage + 1) / 2 , cmap='gray') # 0번째 그리드에 입력 이미지(흑백 이미지)를 출력해준다.
    ax[1].cla() # i,1 그리드의 사진을 클리어한다.
    ax[1].imshow((colorized + 1) / 2) # (3,256,256)의 사진을 (256,256,3)으로 만들어 1번째 그리드에 generator가 생성한 이미지를 출력한다.
    ax[2].cla() # i,2 그리드의 사진을 클리어한다.
    ax[2].imshow((target + 1) / 2) # (3,256,256)의 사진을 (256,256,3)으로 만들어 2번째 그리드에 실제 이미지를 출력한다.
    plt.show()# 출력해준다.


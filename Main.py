import os
import numpy as np
from FCM import FCM
from Model_3D_Trans_DenseUnet import Model_3D_Trans_DenseUnet
from Model_CNN import Model_CNN
from Model_DenseNet import Model_DenseNet
from Model_MDDNet_ASPP import Model_MDDNet_ASPP
from Model_MobileNet import Model_MobileNet
from Model_RAN import Model_RAN
from Model_Trans_Unet import Model_Trans_Unet
from Model_Trans_Unet_plus_plus import Model_Trans_Unet_plus_plus
from Model_UNET import Model_Unet
from Model_Unet3Plus import Model_Unet3plus
from Plot_results import *

# Read Dataset
an = 0
if an == 1:
    Dataset_path = './Dataset/Dataset/Data/'
    Path = os.listdir(Dataset_path)
    Orignal_Images = []
    Target = []
    for i in range(len(Path)):
        Folder = Dataset_path + Path[i]
        Fold_path = os.listdir(Folder)
        for j in range(len(Fold_path)):
            Classes = Folder + '/' + Fold_path[j]
            Class_path = os.listdir(Classes)
            for k in range(len(Class_path)):
                img_path = Classes + '/' + Class_path[k]
                image = cv.imread(img_path)
                image = cv.resize(image, (256, 256))
                cv.imwrite('./Original_images/Image_%04d.png' % (i + 1), image)

                Class_name = Classes.split('/')
                name = Class_name[5]
                if name == 'adenocarcinoma':
                    tar = 1
                elif name == 'large.cell.carcinoma':
                    tar = 2
                elif name == 'squamous.cell.carcinoma':
                    tar = 3
                elif name == 'normal':
                    tar = 0
                Orignal_Images.append(image)
                Target.append(tar)
    Uni = np.unique(Target)
    uni = np.asarray(Uni)
    Tar = np.zeros((len(Target), len(uni))).astype('int')
    for j in range(len(uni)):
        ind = np.where(Target == uni[j])
        Tar[ind, j] = 1
    np.save('Images.npy', Orignal_Images)
    np.save('Target.npy', Tar)

# Ground_Truth
an = 0
if an == 1:
    Images = np.load('Images.npy', allow_pickle=True)
    Seg_Img = []
    for j in range(len(Images)):
        print(j)
        image = Images[j]
        if len(image.shape) == 3:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        alpha = 1.5  # Contrast control (1.0-3.0)
        beta = 0  # Brightness control (0-100)
        med = cv.medianBlur(image, 3)  # Median Filter
        contra = cv.convertScaleAbs(med, alpha=alpha, beta=beta)  # Contrast Enhancement
        cluster = FCM(contra, image_bit=2, n_clusters=5, m=2, epsilon=0.8, max_iter=5)
        cluster.form_clusters()
        result = cluster.result.astype('uint8')
        uniq = np.unique(result)
        uniq = uniq[2:]
        lenUniq = [len(np.where(uniq[i] == result)[0]) for i in range(len(uniq))]
        index = np.argsort(lenUniq)
        img = np.zeros(image.shape, dtype=np.uint8)
        img[result == uniq[index[0]]] = 255
        kernel = np.ones((3, 3), np.uint8)
        img_erosion = cv.erode(img, kernel, iterations=1)
        img_dilate = cv.dilate(img_erosion, kernel, iterations=1)
        Seg_Img.append(img_erosion)
    np.save('Ground_Truth.npy', Seg_Img)


# Segmentation
an = 0
if an == 1:
    Data_path = './Original_images/'
    Data = np.load('Images.npy', allow_pickle=True)  # Load the Data
    Target = np.load('Ground_truth.npy', allow_pickle=True)  # Load the ground truth
    Unet = Model_Unet(Data_path)
    Unet3plus = Model_Unet3plus(Data, Target)
    Trans_Unet = Model_Trans_Unet(Data, Target)
    Trans_Unet_plus_plus = Model_Trans_Unet_plus_plus(Data, Target)
    Proposed = Model_3D_Trans_DenseUnet(Data_path)
    Seg = [Unet, Unet3plus, Trans_Unet, Trans_Unet_plus_plus, Proposed]
    np.save('Segmented_image.npy', Proposed)
    np.save('Seg_img.npy', Seg)


# Classification
an = 0
if an == 1:
    EVAL = []
    Feat = np.load('Segmented_image.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    Batchsize = [4, 8, 16, 32, 48, 64]
    for Bs in range(len(Batchsize)):
        learnperc = round(Feat.shape[0] * 0.75)
        Train_Data = Feat[:learnperc, :]
        Train_Target = Target[:learnperc, :]
        Test_Data = Feat[learnperc:, :]
        Test_Target = Target[learnperc:, :]
        Eval = np.zeros((5, 14))
        Eval[0, :], pred1 = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target, Batchsize[Bs])
        Eval[1, :], pred2 = Model_DenseNet(Train_Data, Train_Target, Test_Data, Test_Target, Batchsize[Bs])
        Eval[2, :], pred3 = Model_RAN(Train_Data, Train_Target, Test_Data, Test_Target, Batchsize[Bs])
        Eval[3, :], pred4 = Model_MobileNet(Train_Data, Train_Target, Test_Data, Test_Target, Batchsize[Bs])
        Eval[4, :], pred5 = Model_MDDNet_ASPP(Train_Data, Train_Target, Test_Data, Test_Target, Batchsize[Bs])
        EVAL.append(Eval)
    np.save('Eval_all_fold.npy', EVAL)  # Save the Eval all


PLot_ROC()
plot_results()
plot_results_Batch_size()
Images_Sample()
Image_segment()

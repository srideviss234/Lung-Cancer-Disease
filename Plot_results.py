import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from itertools import cycle
import cv2 as cv


def PLot_ROC():
    lw = 2
    cls = ['LSTM', 'RNN', 'ResNet', 'DenseNet', 'MDDNet-ASPP']
    Actual = np.load('Target.npy', allow_pickle=True).astype('int')
    colors = cycle(["blue", "crimson", "gold", "lime", "black"])
    for i, color in zip(range(len(cls)), colors):  # For all classifiers
        Predicted = np.load('Y_Score.npy', allow_pickle=True)[0][i]
        false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
        plt.plot(
            false_positive_rate1,
            true_positive_rate1,
            color=color,
            lw=lw,
            label=cls[i], )
    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    path1 = "./Results/ROC.png"
    plt.savefig(path1)
    plt.show()




def plot_results():
    Eval_all = np.load('Eval_all_seg.npy', allow_pickle=True)

    Terms = ['Dice Coefficient', 'Jaccard', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV',
             'FDR', 'F1-Score', 'MCC']

    value_all = Eval_all[0, :]

    stats = np.zeros((value_all[0].shape[1] - 4, value_all.shape[0] + 4, 5))
    for i in range(4, value_all[0].shape[1] - 9):
        for j in range(value_all.shape[0] + 4):
            if j < value_all.shape[0]:
                stats[i, j, 0] = np.max(value_all[j][:, i])
                stats[i, j, 1] = np.min(value_all[j][:, i])
                stats[i, j, 2] = np.mean(value_all[j][:, i])
                stats[i, j, 3] = np.median(value_all[j][:, i])
                stats[i, j, 4] = np.std(value_all[j][:, i])
        X = np.arange(stats.shape[2])
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.bar(X + 0.00, stats[i, 0, :], color='#ff5b00', edgecolor='k', width=0.10, label="Unet")  # r
        ax.bar(X + 0.10, stats[i, 1, :], color='#08ff08', edgecolor='k', width=0.10, label="Unet3+")  # g
        ax.bar(X + 0.20, stats[i, 2, :], color='#3d7afd', edgecolor='k', width=0.10, label="TransUnet")  # b
        ax.bar(X + 0.30, stats[i, 3, :], color='#ff0789', edgecolor='k', width=0.10, label="Trans-Unet++")  # m
        ax.bar(X + 0.40, stats[i, 4, :], color='k', edgecolor='k', width=0.10, label="3D-TDUnet++")  # k
        plt.xticks(X + 0.20, ('BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD'))
        plt.xlabel('Statisticsal Analysis')
        plt.ylabel(Terms[i - 4])
        plt.ylim([0, 1])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
        path1 = "./Results/Segmentation_%s_met.png" % (Terms[i - 4])
        plt.savefig(path1)
        plt.show()


def plot_results_Batch_size():
    eval1 = np.load('Eval_all_fold.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Terms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Classifier = ['TERMS', 'LSTM', 'RNN', 'ResNet', 'DenseNet', 'MDDNet-ASPP']

    value1 = eval1[0, 4, :, 4:]
    Table = PrettyTable()
    Table.add_column(Classifier[0], Terms)
    for j in range(len(Classifier) - 1):
        Table.add_column(Classifier[j + 1], value1[j, :])
    print('-------------------------------------------------- 48 batch size - Dataset', 0 + 1, 'Classifier Comparison',
          '--------------------------------------------------')
    print(Table)

    learnper = [1, 2, 3, 4, 5]
    for j in range(len(Graph_Terms)):
        Graph = np.zeros((eval1.shape[1], eval1.shape[2]))
        for k in range(eval1.shape[1]):
            for l in range(eval1.shape[2]):
                if j == 9:
                    Graph[k, l] = eval1[0, k, l, Graph_Terms[j] + 4]
                else:
                    Graph[k, l] = eval1[0, k, l, Graph_Terms[j] + 4]

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        X = np.arange(6)
        ax.bar(X + 0.00, Graph[:, 0], color='#ff0490', edgecolor='k', hatch='o', width=0.10, label="LSTM")
        ax.bar(X + 0.10, Graph[:, 1], color='#0aff02', edgecolor='k', hatch='o', width=0.10, label="RNN")
        ax.bar(X + 0.20, Graph[:, 2], color='#ad03de', edgecolor='k', hatch='o', width=0.10, label="ResNet")
        ax.bar(X + 0.30, Graph[:, 3], color='#fb7d07', edgecolor='k', hatch='o', width=0.10, label="DenseNet")
        ax.bar(X + 0.40, Graph[:, 4], color='k', edgecolor='w', hatch='oo', width=0.10, label="MDDNet-ASPP")
        plt.xticks(X + 0.10, ('4', '8', '16', '32', '48', '64'))
        plt.xlabel('Batch size')
        plt.ylabel(Terms[Graph_Terms[j]])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
        path1 = "./Results/%s_bar_batch_size.png" % (Terms[Graph_Terms[j]])
        plt.savefig(path1)
        plt.show()


def Images_Sample():
    Original = np.load('Images.npy', allow_pickle=True)
    for i in range(5, 6):
        Orig_1 = Original[i]
        Orig_2 = Original[i + 1]
        Orig_3 = Original[i + 2]
        Orig_4 = Original[i + 3]
        Orig_5 = Original[i + 4]
        Orig_6 = Original[i + 5]
        plt.suptitle('Sample Images from Dataset', fontsize=25)
        plt.subplot(2, 3, 1).axis('off')
        plt.imshow(Orig_1)
        plt.subplot(2, 3, 2).axis('off')
        plt.imshow(Orig_2)
        plt.subplot(2, 3, 3).axis('off')
        plt.imshow(Orig_3)
        plt.subplot(2, 3, 4).axis('off')
        plt.imshow(Orig_4)
        plt.subplot(2, 3, 5).axis('off')
        plt.imshow(Orig_5)
        plt.subplot(2, 3, 6).axis('off')
        plt.imshow(Orig_6)
        plt.show()


def Image_segment():
    Original = np.load('Images.npy', allow_pickle=True)
    segmented = np.load('Seg_Img.npy', allow_pickle=True)
    Image = [39, 49, 66, 89, 103]
    for i in range(len(Image)):
        Orig = Original[Image[i]]
        Seg_1 = segmented[Image[i]]
        for j in range(1):
            print(i, j)
            Orig_1 = Seg_1[j]
            Orig_2 = Seg_1[j + 1]
            Orig_3 = Seg_1[j + 2]
            Orig_4 = Seg_1[j + 3]
            Orig_5 = Seg_1[j + 4]
            plt.suptitle('Segmented Images from Dataset', fontsize=20)
            plt.subplot(2, 3, 1).axis('off')
            plt.imshow(Orig)
            plt.title('Orignal', fontsize=10)
            plt.subplot(2, 3, 2).axis('off')

            plt.imshow(Orig_1)
            plt.title('Unet', fontsize=10)
            plt.subplot(2, 3, 3).axis('off')

            plt.imshow(Orig_2)
            plt.title('Unet3+', fontsize=10)
            plt.subplot(2, 3, 4).axis('off')

            plt.imshow(Orig_3)
            plt.title('TransUnet ', fontsize=10)
            plt.subplot(2, 3, 5).axis('off')

            plt.imshow(Orig_4)
            plt.title('Trans-Unet++', fontsize=10)
            plt.subplot(2, 3, 6).axis('off')

            plt.imshow(Orig_5)
            plt.title('3D Trans-DenseUnet++', fontsize=10)
            plt.show()

            cv.imwrite('./Results/Image_results/Original_image_' + str(i + 1) + '.png', Orig)
            cv.imwrite('./Results/Image_results/segm_img_Unet_' + str(i + 1) + '.png', Orig_1)
            cv.imwrite('./Results/Image_results/segm_img_ResUnet_' + str(i + 1) + '.png', Orig_2)
            cv.imwrite('./Results/Image_results/segm_img_Unet3+_' + str(i + 1) + '.png', Orig_3)
            cv.imwrite('./Results/Image_results/segm_img_ATDUnet_' + str(i + 1) + '.png', Orig_4)
            cv.imwrite('./Results/Image_results/segm_img_PROPOSED_' + str(i + 1) + '.png', Orig_5)


if __name__ == '__main__':
    PLot_ROC()
    plot_results()
    plot_results_Batch_size()
    Images_Sample()
    Image_segment()

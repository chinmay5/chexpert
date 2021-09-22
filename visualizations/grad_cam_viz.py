import os.path

import torch
from dotmap import DotMap
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, multilabel_confusion_matrix
import numpy as np
from tqdm import tqdm

import cv2

from dataset.dataloader_factory import get_dataloader
from environment_setup import PROJECT_ROOT_DIR, read_config
from models.model_factory import create_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
               'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
               'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

class HeatmapGenerator():

    def __init__(self, model):

        self.model = model
        self.model.eval()
        self.weights = list(self.model.features.parameters())[-2]

    def generate(self, single_image, pathImageFile, img_base_path, transCrop=320):

        with torch.no_grad():
            l = self.model(single_image)
            output = self.model.features(single_image)
            label = class_names[torch.max(l,1)[1]]
            #---- Generate heatmap
            heatmap = None
            for i in range(0, len(self.weights)):
                map = output[0, i, :, :]
                if i == 0:
                    heatmap = self.weights[i] * map
                else:
                    heatmap += self.weights[i] * map
                npHeatmap = heatmap.cpu().data.numpy()

        imgOriginal = cv2.imread(os.path.join(img_base_path, pathImageFile), 1)
        imgOriginal = cv2.resize(imgOriginal, (transCrop, transCrop))

        cam = npHeatmap / np.max(npHeatmap)
        cam = cv2.resize(cam, (transCrop, transCrop))
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)

        img = cv2.addWeighted(imgOriginal,1,heatmap,0.35,0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.title(label)
        plt.imshow(img)
        plt.plot()
        plt.axis('off')
        plt.show()


def get_optimal_thresh(outGT, outPRED):
    false_pos_rate, true_pos_rate, thresholds = roc_curve(outGT, outPRED)
    optimal_idx = np.argmax(true_pos_rate - false_pos_rate)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


def get_threshold_all(dataGT, dataPRED, classCount):
    optimal_thresh = []

    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()

    for i in range(classCount):
        try:
            optimal_thresh.append(get_optimal_thresh(datanpGT[:, i], datanpPRED[:, i]))
        except ValueError:
            pass
    return optimal_thresh


def get_optimal_threshold(model, dataLoaderTest):
    outGT = torch.FloatTensor().to(DEVICE)
    outPRED = torch.FloatTensor().to(DEVICE)
    model.eval()

    with torch.no_grad():
        for i, (input, target, _) in tqdm(enumerate(dataLoaderTest)):
            input = input.to(DEVICE)
            target = target.cuda()
            outGT = torch.cat((outGT, target), 0).to(DEVICE)
            out = model(input)

            outPRED = torch.cat((outPRED, out), 0)
    # TODO: This value should be obtained from the Validation Dataset. DO FIX THIS!!!
    threhold_list = get_threshold_all(dataGT=outGT, dataPRED=outPRED, classCount=14)
    print(f"Obtained threshold is:- \n {threhold_list}")
    # Find prediction to the dataframe applying threshold
    threshold_tensor = torch.tensor(threhold_list).reshape(1, -1).to(DEVICE)
    threshholded_pred = torch.where(outPRED < threshold_tensor, 0, 1)
    # Print confusion Matrix
    cm = multilabel_confusion_matrix(y_true=outGT.cpu().numpy(), y_pred=threshholded_pred.cpu().numpy())
    return cm


def boot_strap():
    args = DotMap()
    parser = read_config()
    args.model_type = parser['visualization'].get('model_type')
    args.batch_size = parser['visualization'].getint('batch_size')
    args.num_workers = parser['visualization'].getint('num_workers')
    model_number = parser['visualization'].getint('model_number')
    img_base_path = os.path.join(PROJECT_ROOT_DIR, parser['visualization'].get('img_base_path'))
    nn_class_count = len(class_names)
    data_loader = get_dataloader(args=args, visualization=True)
    model = create_model(args.model_type).to(DEVICE)
    args.class_count = nn_class_count
    args.save_dir = os.path.join(PROJECT_ROOT_DIR, 'results', 'best_models', args.model_type)
    modelCheckpoint = torch.load(os.path.join(args.save_dir, f"model{model_number}.pth"))
    model.load_state_dict(modelCheckpoint['state_dict'])
    # print(get_optimal_threshold(model=model, dataLoaderTest=data_loader['test']))
    hm = HeatmapGenerator(model=model)
    # TODO: Fix this. Should be the `test` split
    single_image, _, image_path = next(iter(data_loader['test']))
    image_path = image_path[0]

    hm.generate(single_image=single_image[0].unsqueeze(0).to(DEVICE), pathImageFile=image_path, img_base_path=img_base_path)


if __name__ == '__main__':
    boot_strap()

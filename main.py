import glob
import os
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

import wandb

wandb.init("labor5")

table = wandb.Table(columns=["id", "image_and_boxes"])


IMAGE_SIZE = (384, 512)

class ListDataset(Dataset):
    def __init__(self, list_path, img_size=IMAGE_SIZE):
        self.img_files = [list_path + img for img in glob.glob1(list_path, "*.png")]
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in
                            self.img_files]
        self.img_shape = img_size
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = Image.open(img_path).convert('RGB')

        input_img = self.transform(img)

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        labels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
        # Fill matrix
        filled_labels = np.zeros((50, 5))
        if labels is not None:
            filled_labels[range(len(labels))[:50]] = labels[:50]
        filled_labels = torch.from_numpy(filled_labels)

        return img_path, input_img, filled_labels

    def __len__(self):
        return len(self.img_files)


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates, átlós sarokpontok számítása mindkét bbox-ra
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle - ez világos
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area - ha kisebb mint nulla, akkor nincs közös részük, ezért min=0

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)  # azért kell a szam a végén hogy ne intként ossza el?

    return iou


import math


def get_anchor_ious(w, h, anchors):
    # Get shape of gt box
    gt_box = torch.FloatTensor(np.array([0, 0, w, h])).unsqueeze(0)
    # Get shape of anchor box
    anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))
    # Calculate iou between gt and anchor shapes
    return bbox_iou(gt_box, anchor_shapes)


def build_targets(pred_boxes, pred_conf, pred_classes, target, anchors, num_anchors, num_classes, grid_size_y,
                  grid_size_x, ignore_thres, img_dim):
    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    nGx = grid_size_x
    nGy = grid_size_y

    # Masks: mask is one for the best bounding box
    # Conf mask is one for BBs, where the confidence is enforced to match target
    mask = torch.zeros(nB, nA, nGy, nGx)
    conf_mask = torch.ones(nB, nA, nGy, nGx)

    # Target values for x,y,w,h and confidence and class
    tx = torch.zeros(nB, nA, nGy, nGx)
    ty = torch.zeros(nB, nA, nGy, nGx)
    tw = torch.zeros(nB, nA, nGy, nGx)
    th = torch.zeros(nB, nA, nGy, nGx)
    tconf = torch.ByteTensor(nB, nA, nGy, nGx).fill_(
        0)  # nullával tölt fel egy uint8as tensort, miért fontos, itt a tipus?
    tcls = torch.ByteTensor(nB, nA, nGy, nGx).fill_(0)

    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(target.shape[1]):
            if target[b, t].sum() == 0:
                continue
            nGT += 1

            # Convert to position relative to box
            t_class = target[b, t, 0].long()
            gx = target[b, t, 1] * nGx
            gy = target[b, t, 2] * nGy
            gw = target[b, t, 3] * nGx
            gh = target[b, t, 4] * nGy

            # Get grid box indices
            gi = int(gx)
            gj = int(gy)

            # Get IoU values between target and anchors
            anch_ious = get_anchor_ious(gw, gh, anchors)

            # tehát itt kiütjük az összes olyan boxot ami ugyanazt az objectet találta meg és utana
            # csak azt állitjuk egybe amelyik a legjobb match

            # Where the overlap is larger than threshold set conf_mask to zero (ignore)
            conf_mask[b, anch_ious > ignore_thres, gj, gi] = 0

            # Find the best matching anchor box
            best_n = np.argmax(anch_ious)

            # Create ground truth box
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)

            # Get the best prediction
            pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)
            pred_class = torch.argmax(pred_classes[b, best_n, gj, gi])

            # Masks
            mask[b, best_n, gj, gi] = 1
            conf_mask[b, best_n, gj, gi] = 1

            # Coordinates
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj

            # Width and height
            tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
            th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)

            # One-hot encoding of label
            tconf[b, best_n, gj, gi] = 1
            tcls[b, best_n, gj, gi] = t_class

            # Calculate iou between ground truth and best matching prediction
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
            score = pred_conf[b, best_n, gj, gi]
            if iou > 0.5 and score > 0.5 and t_class == pred_class:
                nCorrect += 1

    return nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes  # bx by bw bh p0
        self.image_dim = img_dim  # hany csatornas a kep
        self.ignore_thres = 0.5  # mi alatt nullázza ki a találatod?
        self.lambda_coord = 1  # ez mi?

        self.mse_loss = nn.MSELoss(reduction='mean')  # Coordinate loss
        self.bce_loss = nn.BCELoss(reduction='mean')  # Confidence loss
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')  # Class loss

    def forward(self, x, targets=None):
        nA = self.num_anchors  # anchorok száma
        nB = x.size(0)
        nGy = x.size(2)  # grid dimenzió?
        nGx = x.size(3)
        stride = self.image_dim[0] / nGy  # hány lépésre kell felosztani a képet konvhoz

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        # Reshape x: [batchSize x x numAnchors*(5+numClass) nGy x nGx ] -> [batchSize x numAnchors x nGy x nGx x (5+numClass)]
        prediction = x.view(nB, nA, self.bbox_attrs, nGy, nGx).permute(0, 1, 3, 4,
                                                                       2).contiguous()  # contiguous memben is atirja
        #

        # Get outputs
        # ... = ellipszis: kihagyja az összes előző dimenziót, majd (5 + numclassből az adott indexűt szedi ki)
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_class = prediction[..., 5:]  # Class az összes classra adott prediction

        # Calculate offsets for each grid ngx x ngy- os grid 0 1 2 3 soronként és oszloponként
        grid_x = torch.arange(nGx).repeat(nGy, 1).view([1, 1, nGy, nGx]).type(FloatTensor)
        grid_y = torch.arange(nGy).repeat(nGx, 1).t().view([1, 1, nGy, nGx]).type(FloatTensor)

        scaled_anchors = FloatTensor(
            [(a_w / stride, a_h / stride) for a_w, a_h in self.anchors])  # lenormáljuk az anchorokat stride-al
        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.detach() + grid_x
        pred_boxes[..., 1] = y.detach() + grid_y
        pred_boxes[..., 2] = torch.exp(w.detach()) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.detach()) * anchor_h

        # Training
        if targets is not None:

            if x.is_cuda:
                self.mse_loss = self.mse_loss.cuda()
                self.bce_loss = self.bce_loss.cuda()
                self.ce_loss = self.ce_loss.cuda()

            nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(
                pred_boxes=pred_boxes.cpu().detach(),
                pred_conf=pred_conf.cpu().detach(),
                pred_classes=pred_class.cpu().detach(),
                target=targets.cpu().detach(),
                anchors=scaled_anchors.cpu().detach(),
                num_anchors=nA,
                num_classes=self.num_classes,
                grid_size_y=nGy,
                grid_size_x=nGx,
                ignore_thres=self.ignore_thres,
                img_dim=self.image_dim,
            )

            nProposals = int(
                (pred_conf > 0.5).sum().item())  # amikből object lehet, tehát megtalálta a layer 0.5 feletti conf-el
            recall = float(nCorrect / nGT) if nGT else 1  # nemtom
            precision = float(nCorrect / nProposals)

            # Handle masks
            mask = mask.type(ByteTensor).bool()
            conf_mask = conf_mask.type(ByteTensor).bool()

            # Handle target variables
            tx = tx.type(FloatTensor)
            ty = ty.type(FloatTensor)
            tw = tw.type(FloatTensor)
            th = th.type(FloatTensor)
            tconf = tconf.type(FloatTensor)
            tcls = tcls.type(LongTensor)

            # Get conf mask where gt and where there is no gt
            conf_mask_true = mask
            conf_mask_false = conf_mask ^ mask

            # Mask outputs to ignore non-existing objects
            loss_x = self.mse_loss(x[mask], tx[mask])
            loss_y = self.mse_loss(y[mask], ty[mask])
            loss_w = self.mse_loss(w[mask], tw[mask])
            loss_h = self.mse_loss(h[mask], th[mask])
            loss_conf = 10 * self.bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false]) + self.bce_loss(
                pred_conf[conf_mask_true], tconf[conf_mask_true])
            loss_cls = self.ce_loss(pred_class[mask], tcls[mask])
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            return (
                loss,
                loss_x.item(),
                loss_y.item(),
                loss_w.item(),
                loss_h.item(),
                loss_conf.item(),
                loss_cls.item(),
                recall,
                precision,
            )

        else:
            # If not in training phase return predictions
            output = torch.cat(
                (
                    pred_boxes.view(nB, -1, 4) * stride,
                    pred_conf.view(nB, -1, 1),
                    pred_class.view(nB, -1, self.num_classes),
                ),
                -1,
            )
            return output


import torch
import torch.nn as nn


# konvolucio inplanes bemeno es planes kimeno csatornaval
class Conv(nn.Module):
    def __init__(self, inplanes, planes, size=3, stride=1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=size, padding=size // 2, stride=stride)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        return self.bn(torch.relu(self.conv(x)))


class YOLO(nn.Module):
    def __init__(self, planes, nClass, anchors):
        super(YOLO, self).__init__()
        # nB az hogy hány darab anchorunk van?
        nB = len(anchors)

        # minden egyes anchorra megadjuk, a kimenetet, ez lesz a kimeneti tensor
        nOut = nB * (5 + nClass)

        self.c1 = Conv(3, planes)
        self.d1 = Conv(planes, planes * 2, stride=2)
        self.c2 = Conv(planes * 2, planes * 2)
        self.d2 = Conv(planes * 2, planes * 4, stride=2)
        self.c3 = Conv(planes * 4, planes * 4)
        self.d3 = Conv(planes * 4, planes * 8, stride=2)
        self.c4 = Conv(planes * 8, planes * 8)
        self.d4 = Conv(planes * 8, planes * 16, stride=2)
        self.c5 = Conv(planes * 16, planes * 16)
        self.d5 = Conv(planes * 16, planes * 32, stride=2)
        self.c6 = Conv(planes * 32, planes * 32)
        self.c7 = Conv(planes * 32, planes * 32)

        self.classifier = nn.Conv2d(planes * 32, nOut, kernel_size=1)

        # utolso param a képdimenzio, meghatarozzuk, hogy mekkora hibaval talalta el a boxokat
        self.loss = YOLOLayer(anchors, nClass, (384, 512))

    def forward(self, x, targets=None):
        x = self.d1(self.c1(x))
        x = self.d2(self.c2(x))
        x = self.d3(self.c3(x))
        x = self.d4(self.c4(x))
        x = self.d5(self.c5(x))
        x = self.classifier(self.c7(self.c6(x)))

        return self.loss(x, targets)


import matplotlib
import matplotlib.pyplot as plt
from IPython.display import HTML, display


# ez gondolom valami fancy megjelenites
def progress(value, max=100):
    return HTML("""
        <progress
            value='{value}'
            max='{max}',
            style='width: 100%'
        >
            {value}
        </progress>
    """.format(value=value, max=max))


anchors = [[16, 8], [23, 103], [28, 23], [56, 47], [96, 123], [157, 248]]


def train():
    global anchors
    # Makes multiple runs comparable - pseudo random generator
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # betoltjuk a trainset-et
    dataloader = torch.utils.data.DataLoader(
        ListDataset(r"D:\Datasets\YOLOLABOR5\ROBO_Finetune\Finetune/train/", img_size=(384, 512)), batch_size=32,
        shuffle=True)

    # 8 csatornánk van és négy osztályt szeretnénk meghatározni
    model = YOLO(8, 4, anchors).cuda()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)

    bestLoss = 999999999

    numEpoch = 20

    for epoch in range(numEpoch):

        # 5 tulajdonság + 4 class miatt
        losses = np.zeros(9)

        avg_loss = 0.0

        # bar = display(progress(0, len(dataloader)), display_id=True)

        for i, (_, imgs, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
            imgs = imgs.cuda()
            targets = targets.cuda().requires_grad_(False)

            optimizer.zero_grad()

            loss = model(imgs, targets)

            loss[0].backward()
            optimizer.step()

            # szepen kiirjuk a losst parameterenkent
            losses += torch.Tensor(loss).cpu().detach().numpy()
            avg_loss += loss[0].item()
            # bar.update(progress(i + 1, len(dataloader)))

        avg_loss /= len(dataloader)

        if loss[0].item() < bestLoss:
            print("Best model saved")
            bestLoss = loss[0].item()
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }, "ckpt.tar")

        # szepen kiirjuk a losst parameterenkent
        print(
            "[Epoch %d/%d][Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
            % (
                epoch + 1,
                20,
                losses[1] / float(len(dataloader)),
                losses[2] / float(len(dataloader)),
                losses[3] / float(len(dataloader)),
                losses[4] / float(len(dataloader)),
                losses[5] / float(len(dataloader)),
                losses[6] / float(len(dataloader)),
                losses[0] / float(len(dataloader)),
                losses[7] / float(len(dataloader)),
                losses[8] / float(len(dataloader)),
            )
        )
    return model


def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    box_corner = prediction.new(prediction.shape)  # létrehozunk egy ugyanolyan dimenziójú tenort arrayt matrixot
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :,
                                                3] / 2  # meghatározzuk a két átlós koordinátáját a doboznak a bejövő paramokból
    prediction[:, :, :4] = box_corner[:, :, :4]  # visszarakjuk a prediction matrixban az atalakitott koordinatakat

    output = [None for _ in range(
        len(prediction))]  # csinálunk egy tömböt, ami olyan hosszú, mint a beérkező prediction-ök none-al tölti fel, futo valtozot nem hasznaljuk
    for image_i, image_pred in enumerate(prediction):  # i futo valtozo, pred adott elem

        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()  # a zarojel utan mi a visszateresi ertek?
        image_pred = image_pred[conf_mask]

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence, 1 azt jelenti, hogy melyik dimenziót legyen kilőve, amugy keepdim, megtartjuk a dimenziót
        class_conf, class_pred = torch.max(image_pred[:, 5:], 1, keepdim=True)

        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()  # unique kiszedi a többször benne levő részeket
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:

            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]

            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]

            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):  # == while true, kivéve ha a detection class ures

                # Get detection with highest confidence and save as max detection
                max_detections.append(
                    detections_class[0].unsqueeze(0))  # csinal a detectionoknek egy sorvektort [[0 1 2 3] ...]

                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = (
                max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))
            )

    return output


class COCOBox:
    pass


class YOLOBox:
    def __init__(self, t, mode='YOLO'):
        if mode == 'YOLO':
            x, y, w, h = t
            x = x*IMAGE_SIZE[1]
            y = y*IMAGE_SIZE[0]
            w = w*IMAGE_SIZE[0]
            h = h*IMAGE_SIZE[1]

            self.xmin = x - w/2
            self.xmax = x + w/2
            self.ymin = y - h/2
            self.ymax = y + h/2

        elif mode == 'COCO':
            self.xmin = t[0]
            self.ymin = t[1]
            self.xmax = t[2]
            self.ymax = t[3]
        else:
            print('Invalid model parameter')


def evaluation(model):
    from matplotlib.pyplot import imshow
    import matplotlib.pyplot as plt
    import cv2

    plt.rcParams["axes.grid"] = False

    dataloader = torch.utils.data.DataLoader(
        ListDataset(r"D:\Datasets\YOLOLABOR5\ROBO_Finetune\Finetune/train/", img_size=(384, 512)), batch_size=16,
        shuffle=True)

    model.eval()

    images = []
    boxes = []
    classes = []
    scores = []

    for i, (paths, imgs, targets) in enumerate(dataloader):
        imgs = imgs.cuda()

        with torch.no_grad():
            detections = model(imgs)
            detections = non_max_suppression(detections, 4)

        for path, detection in zip(paths, detections):
            img = np.array(Image.open(path).convert('RGB'))
            img = cv2.cvtColor(img, cv2.COLOR_YUV2RGB)
            img = cv2.resize(img, (512, 384))

            images.append(img)

            temp_boxes = []
            temp_classes = []
            temp_scores = []

            if detection is not None:
                unique_labels = detection[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                bbox_colors = [(255, 0, 0), (255, 0, 255), (0, 0, 255), (255, 255, 0)]
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                    # Rescale coordinates to original dimensions
                    '''y1 = (y1) * 1.25
                    x1 = (x1) * 1.25
                    y2 = (y2) * 1.25
                    x2 = (x2) * 1.25'''

                    color = bbox_colors[int(cls_pred)]
                    # Create a Rectangle patch

                    minX = min(x1, x2)
                    maxX = max(x1, x2)
                    minY = min(y1, y2)
                    maxY = max(y1, y2)

                    temp_boxes.append(YOLOBox((minX, maxX, minY, maxY, cls_pred), mode='COCO'))
                    temp_classes.append(cls_pred)
                    temp_scores.append(conf)

                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            boxes.append(temp_boxes)
            classes.append(temp_classes)
            scores.append(temp_scores)

            # plt.figure()
            # imshow(img)

        # plt.show()
        break

    return imgs, boxes, classes, scores


def from_pretrained(file):
    return torch.load(file)


# this is the order in which my classes will be displayed
display_ids = {"car": 0, "truck": 1, "person": 2, "traffic light": 3, "stop sign": 4,
               "bus": 5, "bicycle": 6, "motorbike": 7, "parking meter": 8, "bench": 9,
               "fire hydrant": 10, "aeroplane": 11, "boat": 12, "train": 13}
# this is a revese map of the integer class id to the string class label
# class_id_to_label = { int(v) : k for k, v in display_ids.items()}

class_id_to_label = {
    0: "grass",
    1: "football",
    3: "robot",
    4: "goalpost"
}

idx = 0


def bounding_boxes(raw_image, v_boxes, v_clsids, v_scores, log_width=1024, log_height=1024):
    global idx
    # load raw input photo
    all_boxes = []
    # plot each bounding box for this image
    for b_i, box in enumerate(v_boxes):
        # get coordinates and labels
        box_data = {
            "position": {
                "minX": int(box.xmin.item()),
                "maxX": int(box.xmax.item()),
                "minY": int(box.ymin.item()),
                "maxY": int(box.ymax.item())
            },
            "class_id": int(v_clsids[b_i].item()),
            # optionally caption each box with its class and score
            "box_caption": "%s (%.3f)" % (v_clsids[b_i], v_scores[b_i] if v_scores is not None else 1),
            "domain": "pixel",
            # "scores": {"score": v_scores[b_i].item()}
            "scores": {"score": 1}
        }
        all_boxes.append(box_data)

    # log to wandb: raw image, predictions, and dictionary of class labels for each class id
    box_image = wandb.Image(raw_image,
                            boxes={"predictions": {"box_data": all_boxes, "class_labels": class_id_to_label}})

    # TODO
    # ITT FOLYTASSAD!! Ebben a debug ban az all boxes valamiert ures, nem kerul bele semmi erdemi adat a ground truth okbol, also nincs a groud truth okban kiszurva, ha valami nem bb, ugye ez az az eset amikor csupa 0 a sor

    table.add_data(idx, box_image)
    idx += 1

    return box_image


def visualize_wandb(imgs, boxes, classes, scoress):
    if scoress is not None:
        for img, bbs, clss, scores in zip(imgs, boxes, classes, scoress):
            bounding_boxes(img, bbs, clss, scores)
    else:
        for img, bbs, clss in zip(imgs, boxes, classes):
            bounding_boxes(img, bbs, clss, None)

def some_ground_truth_for_visualization(number=10):
    dataset = ListDataset(r'D:\Datasets\YOLOLABOR5\ROBO_Finetune\Finetune\test/')
    dataloader = DataLoader(dataset, batch_size=1)

    idx = 0

    imgs = []
    bboxes = []
    classes = []
    scores = None

    for idx, (path, img, boxes) in enumerate(dataloader):
        imgs.append(img)
        temp_classes = []
        temp_boxes = []
        for i in range(50):
            temp_classes.append(boxes[0, i, 0])
            if boxes[0, i, 1] == 0 and boxes[0, i, 2] == 0 and boxes[0, i, 3] == 0 and boxes[0, i, 4] == 0:
                break
            temp_boxes.append(YOLOBox((boxes[0, i, 1], boxes[0, i, 2], boxes[0, i, 3], boxes[0, i, 4])))
        bboxes.append(temp_boxes)
        classes.append(temp_classes)
        if idx == number:
            break

    return imgs, bboxes, classes, None # mert ebben ground_truth-okkal terunk vissza

if __name__ == '__main__':
    # model = train()
    ckpt = from_pretrained("ckpt.tar")
    model = YOLO(8, 4, anchors).cuda()
    model.load_state_dict(ckpt["model"])
    # imgs, boxes, classes, scores = evaluation(model)
    # raw_image, v_boxes, v_labels, v_scores

    imgs, boxes, classes, scores = some_ground_truth_for_visualization()

    visualize_wandb(imgs, boxes, classes, scores)
    wandb.log({"labor5table": table})
    wandb.finish()

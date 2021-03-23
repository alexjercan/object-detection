"""
Implementation of Yolo Loss Function similar to the one in Yolov3 paper,
the difference from what I can tell is I use CrossEntropy for the classes
instead of BinaryCrossEntropy.
"""
import random
import torch
import torch.nn as nn

from util.general import intersection_over_union


class LossFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, targets, anchors):
        device = targets[0].device
        class_loss = torch.zeros(1, device=device)
        box_loss = torch.zeros(1, device=device)
        object_loss = torch.zeros(1, device=device)
        no_object_loss = torch.zeros(1, device=device)

        for p, t, a in zip(predictions, targets, anchors):
            obj = t[..., 0] == 1
            noobj = t[..., 0] == 0

            # ======================= #
            #   FOR NO OBJECT LOSS    #
            # ======================= #

            no_object_loss += self.bce((p[..., 0:1][noobj]),
                                       (t[..., 0:1][noobj]),)

            # ==================== #
            #   FOR OBJECT LOSS    #
            # ==================== #

            a = a.reshape(1, 3, 1, 1, 2)
            box_preds = torch.cat(
                [self.sigmoid(p[..., 1:3]), torch.exp(p[..., 3:5]) * a], dim=-1)
            ious = intersection_over_union(box_preds[obj],
                                           t[..., 1:5][obj]).detach()
            object_loss += self.mse(self.sigmoid(p[..., 0:1][obj]),
                                    ious * t[..., 0:1][obj])

            # ======================== #
            #   FOR BOX COORDINATES    #
            # ======================== #

            p[..., 1:3] = self.sigmoid(p[..., 1:3])
            t[..., 3:5] = torch.log((1e-16 + t[..., 3:5] / a))
            box_loss += self.mse(p[..., 1:5][obj], t[..., 1:5][obj])

            # ================== #
            #   FOR CLASS LOSS   #
            # ================== #

            class_loss += self.entropy((p[..., 5:][obj]),
                                       (t[..., 5][obj].long()),)

        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )

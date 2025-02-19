from typing import Any

import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pathlib import Path
import torch
import pandas as pd
import torch.nn as nn
from torch import nn
import torch.nn.functional as F

Tensor = torch.Tensor
BatchType = tuple[Tensor, dict[str, Any]]

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradient = None
        self.activation = None
        
        self.model.eval()
        self.handlers = self.register_hooks()
        
    def register_hooks(self):
        handlers = []
        
        def backward_hook(module, grad_input, grad_output):
            self.gradient = grad_output[0]
            
        def forward_hook(module, input, output):
            self.activation = output
            
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                handlers.append(module.register_forward_hook(forward_hook))
                handlers.append(module.register_backward_hook(backward_hook))
                
        return handlers
    
    def remove_hooks(self):
        for handler in self.handlers:
            handler.remove()
    
    def __call__(self, input_tensor, target_class):
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        target = output[:, target_class]

        target.backward()
        
        gradients = self.gradient.mean(dim=-1, keepdim=True)
        weights = F.relu(gradients)
        
        activations = self.activation.squeeze()
        grad_cam = torch.sum(activations * weights, dim=0)
        
        grad_cam = F.relu(grad_cam)
        grad_cam = grad_cam / torch.max(grad_cam)
        
        return output, grad_cam


class BaseModule(pl.LightningModule):
    def __init__(self, attribute=False, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.attribute = attribute
        self.test_data_name = "UNKNOWN"
        self.labels = ["negative", "positive"]

        self.inference_meta = []
        self.inference_attr = []

    def training_step(self, batch: BatchType, batch_idx: int) -> Tensor:
        x, meta = batch

        z = self.forward(x)
        loss = nn.functional.cross_entropy(z, meta["y"])
        self.log("train_loss", loss)

        return loss
    

    def validation_step(self, batch: BatchType, batch_idx: int) -> Tensor:
        self.val_test_step(batch, batch_idx)
        

    def test_step(self, batch: BatchType, batch_idx: int) -> Tensor:
        self.val_test_step(batch, batch_idx)


    def val_test_step(self, batch: BatchType, batch_idx: int) -> Tensor:
        torch.set_grad_enabled(True)

        x, meta = batch

        if not self.attribute:
            z = self.forward(x)
            y_hat = torch.argmax(z, dim=1)
        else:
            gradcam = GradCAM(self, target_layer="layer4")
            z = []
            grad_cam = []
            for i in range(0, x.shape[0]):
                z_, grad_cam_ = gradcam(x[i].unsqueeze(0), 1)
                grad_cam_ = grad_cam_.mean(dim=0).unsqueeze(0).unsqueeze(0)
                grad_cam_ = F.interpolate(grad_cam_, size=8192, mode="linear", align_corners=False)
                grad_cam_ = grad_cam_.flatten() / torch.max(grad_cam_)
                grad_cam.append(grad_cam_.detach().cpu())
                z.append(z_.squeeze(0))
            grad_cam = torch.stack(grad_cam)
            gradcam.remove_hooks()

            z  = torch.stack(z)
            y_hat = torch.argmax(z, dim=1)

            self.inference_attr.append(grad_cam)

        meta["y_hat"] = y_hat
        for key in meta:
            if isinstance(meta[key], torch.Tensor):
                meta[key] = meta[key].detach().cpu().numpy()
        self.inference_meta.append(meta)


    def log_predictions(self, group: str):
        meta = pd.concat([pd.DataFrame(m) for m in self.inference_meta], ignore_index=True)

        if self.attribute:
            attr = torch.cat(self.inference_attrf)

        # 0. Save predictions to file
        meta.to_csv(f"inference_{group[:60]}.csv", index=False)

        # 1. Accuracy
        self.log(f"{group} accuracy", (meta["y_hat"] == meta["y"]).mean())

        # 2. Predicted label distribution
        p_yhat_label = []
        for li, label in enumerate(self.labels):
            p_yhat_label.append((meta["y_hat"] == li).mean())
            self.log(f"{group}  p(ŷ={label})", p_yhat_label[-1])

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.pie(p_yhat_label, labels=self.labels, autopct='%1.1f%%', shadow=True, startangle=90)
        ax.axis('equal')
        self.logger.experiment.log_figure(figure=fig, figure_name=f"{group} p(ŷ)", step=self.global_step)
        plt.close(fig)

        # 3. Confusion matrix
        self.logger.experiment.log_confusion_matrix(
            meta["y"],
            meta["y_hat"],
            labels=self.labels,
            title=f"Confusion Matrix - dataset {group}",
            file_name=f"{group}-confusion-matrix.json",
        )

        # 4. Attribution by class, lil github-style heatmap
        if self.attribute:
            fig, ax = plt.subplots(figsize=(15, 7))
            colors_set2 = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f"]
            
            vmin,vmax = 0,0
            for label, color in zip(self.labels, colors_set2):
                median_attr = torch.median(attr, dim=0).values
                q1 = torch.quantile(attr, 0.25, dim=0)
                q3 = torch.quantile(attr, 0.75, dim=0)
                vmin = min(vmin, q1.min().item())
                vmax = max(vmax, q3.max().item())

                ax.plot(median_attr, c=color, label=label)
                ax.fill_between(attr.shape[-1], q1, q3, alpha=0.2, color=color)
            ax.legend()
            ax.set_ylim(vmin, vmax)
            self.logger.experiment.log_figure(figure=fig, figure_name=f"{group} attribution by class", step=self.global_step)
            plt.close(fig)

        self.inference_meta = []
        self.inference_attr = []


    def on_validation_epoch_end(self):
        self.log_predictions("val")

    def on_test_epoch_end(self):
        self.log_predictions(self.test_data_name)

    def predict_step(self, batch: BatchType, batch_idx) -> Tensor:
        x, meta = batch
        return self.forward(x), meta

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer


class MLPClassifier(BaseModule):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.flatten(x, start_dim=1)  # Flatten input to (batch_size, input_size)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ResNetClassifier(BaseModule):
    def __init__(self, input_size: int, num_classes: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        pkw = 1
        for kw in [16, 32, 64]:
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(kw, kw, kernel_size=3, padding=1),
                    nn.BatchNorm1d(kw),
                    nn.ReLU(),
                    nn.Conv1d(kw, kw, kernel_size=3, padding=1),
                    nn.BatchNorm1d(kw),
                    nn.ReLU(),
                    nn.Conv1d(kw, kw, kernel_size=3, padding=1),
                    nn.BatchNorm1d(kw),
                )
            )
            self.downsamples.append(nn.Conv1d(pkw, kw, kernel_size=1))
            pkw = kw
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        x = nn.functional.avg_pool1d(x, kernel_size=2)

        for downsample, block in zip(self.downsamples, self.blocks):
            x = downsample(x)
            out = block(x)
            x = out + x
            x = nn.functional.relu(x)
            x = nn.functional.max_pool1d(x, kernel_size=2)

        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


class SpecClassiFier(BaseModule):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(80_000, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = nn.functional.relu(self.bn1(x))
        x = self.pool(x)
        x = self.conv2(x)
        x = nn.functional.relu(self.bn2(x))
        x = self.pool(x)
        x = self.conv3(x)
        x = nn.functional.relu(self.bn3(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def conv3(in_channel, out_channel, stride=1, padding=1, groups=1):
    return nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=stride, padding=padding, bias=False, dilation=padding, groups=groups)


def conv1(in_channel, out_channel, stride=1, padding=0):
    return nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=stride, padding=padding, bias=False)


def bcnorm(channel):
    return nn.BatchNorm1d(channel)


class Bottleneck(nn.Module):
    expansion = 1.5

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1(in_channel, in_channel)
        self.bn1 = bcnorm(in_channel)
        self.conv2 = conv3(in_channel, in_channel, stride)
        self.bn2 = bcnorm(in_channel)
        self.conv3 = conv1(in_channel, out_channel)
        self.bn3 = bcnorm(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SquiggleNet(BaseModule):
    def __init__(self, block, layers, num_classes, attribute=False):
        super(SquiggleNet, self).__init__(attribute=attribute)
        self.chan1 = 20

        # first block
        self.conv1 = nn.Conv1d(1, 20, 19, padding=5, stride=3)
        self.bn1 = bcnorm(self.chan1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2, padding=1, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(67, num_classes)
        self.attribute = attribute

        self.layer1 = self._make_layer(block, 20, layers[0])
        self.layer2 = self._make_layer(block, 30, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 45, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 67, layers[3], stride=2)
        # self.layer5 = self._make_layer(block, 100, layers[4], stride=2)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.chan1 != channels:
            downsample = nn.Sequential(
                conv1(self.chan1, channels, stride),
                bcnorm(channels),
            )

        layers = []
        layers.append(block(self.chan1, channels, stride, downsample))
        if stride != 1 or self.chan1 != channels:
            self.chan1 = channels
        for _ in range(1, blocks):
            layers.append(block(self.chan1, channels))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # 32 8192
        x = x.unsqueeze(1)  # 32 1 8192
        x = self.conv1(x)  # 32 20 2731
        x = self.bn1(x)  # 32 20 2731
        x = self.relu(x)  # 32 20 2731
        x = self.pool(x)  # 32 20 1366

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.layer5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

import numpy as np
import torch
import math


def pytorch_switch(tensor_image):
    return tensor_image.permute(1, 2, 0)


def to_pytorch(tensor_image, device=None):
    if torch.is_tensor(tensor_image):
        x = tensor_image
    else:
        x = torch.as_tensor(tensor_image, dtype=torch.float32)

    if x.ndim == 3:
        if x.shape[0] == 3:
            pass
        elif x.shape[-1] == 3:
            x = x.permute(2, 0, 1)
        else:
            raise ValueError(f"Expected CHW or HWC image, got shape {tuple(x.shape)}")
    elif x.ndim == 4:
        if x.shape[1] == 3:
            pass
        elif x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Expected NCHW or NHWC batch, got shape {tuple(x.shape)}")
    else:
        raise ValueError(f"Expected image or image batch, got shape {tuple(x.shape)}")

    if device is not None:
        x = x.to(device=device, dtype=torch.float32, non_blocking=True)
    return x.contiguous()


class _LossBase:
    def __init__(self, model, true, unormalize=False, to_pytorch=False):
        self.model = model
        self.true = true
        self.unormalize = unormalize
        self.to_pytorch = to_pytorch

    def _prepare(self, img):
        return img * 255. if self.unormalize else img

    def _predict(self, img):
        img_ = self._prepare(img)
        if self.to_pytorch:
            device = getattr(self.model, "device", None)
            if device is None and hasattr(self.model, "model"):
                device = getattr(self.model.model, "device", None)
            img_ = to_pytorch(img_, device=device)
            if img_.ndim == 3:
                img_ = img_[None, :]
            return self.model.predict(img_)

        img_ = np.asarray(img_)
        if img_.ndim == 3:
            img_ = np.expand_dims(img_, axis=0)
        return self.model.predict(img_)

    def get_label(self, img):
        preds = self._predict(img)
        if torch.is_tensor(preds):
            preds = preds.detach()
            if preds.ndim == 1:
                preds = preds.unsqueeze(0)
            return int(torch.argmax(preds[0]).item())
        return int(np.argmax(np.asarray(preds).reshape(-1)))


class UnTargeted(_LossBase):
    def __init__(self, model, true, unormalize=False, to_pytorch=False):
        super().__init__(model, true, unormalize=unormalize, to_pytorch=to_pytorch)

    def __call__(self, img):
        if self.unormalize:
            img_ = img * 255.

        else:
            img_ = img

        if self.to_pytorch:
            device = getattr(self.model, "device", None)
            if device is None and hasattr(self.model, "model"):
                device = getattr(self.model.model, "device", None)
            img_ = to_pytorch(img_, device=device)
            img_ = img_[None, :]
            preds = self.model.predict(img_).flatten()
            y = int(torch.argmax(preds))
            preds = preds.detach().cpu().tolist()
        else:
            preds = self.model.predict(np.expand_dims(img_, axis=0)).flatten()
            y = int(np.argmax(preds))

        is_adversarial = True if y != self.true else False

        f_true = math.log(math.exp(preds[self.true]) + 1e-30)
        preds[self.true] = -math.inf

        f_other = math.log(math.exp(max(preds)) + 1e-30)
        return [is_adversarial, float(f_true - f_other)]

    def batch(self, imgs):
        preds = self._predict(imgs)
        if torch.is_tensor(preds):
            preds = preds.detach()
            if preds.ndim == 1:
                preds = preds.unsqueeze(0)
            labels = torch.argmax(preds, dim=1)
            masked = preds.clone()
            masked[:, self.true] = -torch.inf
            losses = preds[:, self.true] - torch.max(masked, dim=1).values
            return (
                (labels != self.true).cpu().tolist(),
                losses.cpu().tolist(),
            )

        preds = np.asarray(preds)
        if preds.ndim == 1:
            preds = preds[None, :]
        labels = np.argmax(preds, axis=1)
        masked = preds.copy()
        masked[:, self.true] = -np.inf
        losses = preds[:, self.true] - np.max(masked, axis=1)
        return (
            (labels != self.true).tolist(),
            losses.astype(float).tolist(),
        )


class Targeted(_LossBase):
    def __init__(self, model, true, target, unormalize=False, to_pytorch=False):
        super().__init__(model, true, unormalize=unormalize, to_pytorch=to_pytorch)
        self.target = target

    def __call__(self, img):
        if self.unormalize:
            img_ = img * 255.

        else:
            img_ = img

        if self.to_pytorch:
            device = getattr(self.model, "device", None)
            if device is None and hasattr(self.model, "model"):
                device = getattr(self.model.model, "device", None)
            img_ = to_pytorch(img_, device=device)
            img_ = img_[None, :]
            preds = self.model.predict(img_).flatten()
            y = int(torch.argmax(preds))
            preds = preds.detach().cpu().tolist()
        else:
            preds = self.model.predict(np.expand_dims(img_, axis=0)).flatten()
            y = int(np.argmax(preds))

        is_adversarial = True if y == self.target else False
        f_target = preds[self.target]

        f_other = math.log(sum(math.exp(pi) for pi in preds))
        return [is_adversarial, f_other - f_target]

    def batch(self, imgs):
        preds = self._predict(imgs)
        if torch.is_tensor(preds):
            preds = preds.detach()
            if preds.ndim == 1:
                preds = preds.unsqueeze(0)
            labels = torch.argmax(preds, dim=1)
            losses = torch.logsumexp(preds, dim=1) - preds[:, self.target]
            return (
                (labels == self.target).cpu().tolist(),
                losses.cpu().tolist(),
            )

        preds = np.asarray(preds)
        if preds.ndim == 1:
            preds = preds[None, :]
        labels = np.argmax(preds, axis=1)
        losses = np.logaddexp.reduce(preds, axis=1) - preds[:, self.target]
        return (
            (labels == self.target).tolist(),
            losses.astype(float).tolist(),
        )

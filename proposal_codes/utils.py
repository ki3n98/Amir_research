import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

def denorm(img_chw: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    # img_chw: (3, H, W), float
    x = img_chw.detach().cpu()
    x = x * std + mean
    return x.clamp(0, 1)


def show_img_with_boxes(img_chw, target, label_names=None, title=None):
    """
    target: dict with keys:
      - 'boxes': Tensor[N,4] in (x1,y1,x2,y2)
      - 'labels': Tensor[N]
    """
    img = denorm(img_chw)
    img_hwc = img.permute(1, 2, 0).numpy()  # HWC for matplotlib

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img_hwc)
    ax.axis("off")
    if title:
        ax.set_title(title)

    boxes = target["boxes"].detach().cpu()
    labels = target["labels"].detach().cpu()

    for box, lab in zip(boxes, labels):
        x1, y1, x2, y2 = box.tolist()
        w, h = (x2 - x1), (y2 - y1)

        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)

        lab_int = int(lab.item())
        text = label_names.get(lab_int, str(lab_int)) if label_names else str(lab_int)
        ax.text(x1, y1 - 3, text, color="lime", fontsize=10, va="bottom",
                bbox=dict(facecolor="black", alpha=0.4, pad=2, edgecolor="none"))

    plt.show()



def to_device(obj):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return [to_device(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_device(v) for k, v in obj.items()}
    return obj

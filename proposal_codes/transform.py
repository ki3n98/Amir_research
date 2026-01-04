import albumentations as A


IMG_SIZE = (300, 300)
class_labels = ['background', 'drinking', 'eating', 'mobile use', 'smoking']

transforms_with_boxes = A.Compose([
    A.RandomCrop(width=IMG_SIZE[0], height=IMG_SIZE[1], p=1.0),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Normalize(),
    A.pytorch.ToTensorV2(),
],  bbox_params=A.BboxParams(format='pascal_voc', 
                        label_fields=["class_labels"],
                        clip=True,
                        min_visibility=0.1
                        ))

val_transforms = A.Compose([
    A.RandomCrop(width=IMG_SIZE[0], height=IMG_SIZE[1], p=1.0),
    A.Normalize(),
    A.pytorch.ToTensorV2(),
],  bbox_params=A.BboxParams(format='pascal_voc', 
                        label_fields=["class_labels"],
                        clip=True,
                        min_visibility=0.1
                        ))


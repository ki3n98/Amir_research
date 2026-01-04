import os
import cv2
import torch
from torchvision import tv_tensors

class CustomDataset():
    def __init__(self, image_dir, annotations, annotations_dir, transform=None):
        self.image_dir = image_dir
        self.annotations = annotations
        self.annotations_dir = annotations_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    

    def __get_img__(self, img_path, idx):
        """
        Load image at img_path return RGB img.
        """
        image = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)
        if image is None:
            return None
        
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    

    def __get_annotation__(self, annotation_path, width, height):
        """
        Load annotation file and return list of label and boxes
        """
        boxes = []
        labels = []
        # Open annotation file
        with open(annotation_path, "r") as f:
            for line in f:
                # Split data
                data = line.strip().split()
                class_index = int(data[0]) 
                x_center = float(data[1]) * width
                y_center = float(data[2]) * height
                box_width = float(data[3]) * width
                box_height = float(data[4]) * height

                # Convert to absolute coordinates [xmin, ymin, xmax, ymax]
                xmin = x_center - box_width / 2
                ymin = y_center - box_height / 2
                xmax = x_center + box_width / 2
                ymax = y_center + box_height / 2

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(class_index + 1)) # +1 for pretrain __background__ class

        return boxes, labels


    def __getitem__(self, idx):
        """
        Get img and target at idx. 
        """
        # Get annotation and img_path
        annotation = self.annotations[idx]
        img_path = os.path.join(self.image_dir, annotation[:-4] + '.jpg')


        # Load image
        tries = 0
        while tries < len(self.annotations):
            annotation = self.annotations[idx]
            img_path = os.path.join(self.image_dir, annotation[:-4] + ".jpg")

            image = self.__get_img__(img_path, idx)
            if image is not None:
                break

            print(f"Warning: Image {img_path} not found. Skipping.")
            idx = (idx + 1) % len(self.annotations)
            tries += 1
        # image = self.__get_img__(img_path, idx)

        height, width, _ = image.shape

        # Parse the annotation file
        annotation_file = os.path.join(self.annotations_dir, annotation)
        boxes, labels = self.__get_annotation__(annotation_file, width, height)

        # Apply transformations
        if self.transform:
            transform = self.transform(
                image=image, 
                bboxes=boxes, 
                class_labels=labels
                )
            
            image = transform["image"]
            boxes = transform["bboxes"]
            labels = transform["class_labels"]
        

        labels = torch.tensor(labels, dtype=torch.int64)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        target = {"boxes": boxes, "labels": labels}

        return image, target

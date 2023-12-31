"""
#TODO:
-> Preprocessing
-> Fix Random caption
-> Fix variable length of list of labels
"""
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import functional
from PIL import Image
import itertools
import random
import yaml
import os
from ast import literal_eval
import json

default_config_file = "/home/tbaweja/wproj/dataloader/dataloader_config.yaml"

class COCODataset(torch.utils.data.Dataset):

    def __init__(self, config_file: str = default_config_file, mode: str = "train"):

        with open(config_file, "rb") as f:
            self.config = yaml.load(f, Loader = yaml.FullLoader)
        
        self.mode = mode
        random.seed(self.config["seed"])
        metadata_dir = self.config["metadata_path"]
        metadata_file_path = os.path.join(metadata_dir, self.config["metadata_file"])
        data_version = self.config["data_version"]
        self.min_classes = self.config["min_classes"]
        self.metadata_df = pd.read_csv(metadata_file_path)
        self._preprocess_metadata()
        self.data_dir = self.config["data_path"]
        
        self.images_dir = os.path.join(self.data_dir, f"{self.mode}{data_version}")
        self.res_height = self.config["img_height"]
        self.res_width = self.config["img_width"]
        self.caption_mode = self.config["caption_mode"]
        self.metadata_df = self._limit_dataset()
    
    def _preprocess_metadata(self):

        def convert_to_iter(str_rep: str):
            try:
                return literal_eval(str_rep)
            except ValueError:
                elements = str_rep.strip("{}").split(", ")
                return set(elements)
        
        self.metadata_df["image_captions"] = self.metadata_df["image_captions"].apply(convert_to_iter)
        self.metadata_df["labels"] = self.metadata_df["labels"].apply(convert_to_iter)
    
    def _limit_dataset(self):
        metadata_subset = self.metadata_df[self.metadata_df["labels"].apply(lambda x: len(x) >= self.min_classes)]
        return metadata_subset
    
    def convert_index(self, index: int):
        try:
            row = self.metadata_df.loc[index].to_dict()
            return row["id"]
        except KeyError:
            print(f"Index {index} not found")
            return None

    
    def load_image_info(self, image_idx: int):
        image_subset_df = self.metadata_df[self.metadata_df["id"] == image_idx]
        image_info_dict = image_subset_df.iloc[0].to_dict()
        return image_info_dict

    
    def load_image(self, image_info_dict: dict):        
        image_filename = image_info_dict["file_name"]
        file_path = os.path.join(self.images_dir, image_filename)
        img = Image.open(file_path)
        return img
    
    def __len__(self):
        return len(self.metadata_df)
    
    def load_caption(self, image_info_dict: dict):
        captions = image_info_dict["image_captions"]        
        captions = sorted(captions, key = lambda x: len(x))
        if self.caption_mode == "random":
            caption = random.choice(captions)
        elif self.caption_mode == "detailed":
            caption = captions[-1]
        return caption
    
    def load_label_catgories(self, image_info_dict: dict):
        labels = list(image_info_dict["labels"])
        return labels[:self.min_classes]
    
    def crop_img(self, img):
        height, width = img.size
        if height < self.res_height or width < self.res_width:
            raise AssertionError("Incompatible image")
        
        resized_img = functional.resize(img, (self.res_height, self.res_width))
        resized_arr = np.array(resized_img)
        return resized_arr
    
    def preprocess_image(self, img_arr: np.ndarray):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized_image_array = (img_arr / 255.0 - mean) / std
        return normalized_image_array
    
    def __getitem__(self, idx):
        train_img_idx = self.convert_index(idx)
        image_info = self.load_image_info(train_img_idx)
        img = self.load_image(image_info)
        img_arr = self.crop_img(img)
        norm_img = self.preprocess_image(img_arr)
        tensor_img = torch.tensor(norm_img).permute(2, 0, 1)
        caption = self.load_caption(image_info)
        labels = self.load_label_catgories(image_info)

        return {
            "image": tensor_img,
            "caption": caption,
            "labels": labels
        }


if __name__ == "__main__":
    dataset = COCODataset()
    img_dict = dataset[0]
    print(img_dict)
    
        
    

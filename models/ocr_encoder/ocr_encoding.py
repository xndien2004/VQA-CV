import torch
from torch import nn
import os
from typing import List
from typing import List, Dict,Any
import numpy as np
import scipy.spatial.distance as distance
import random

def load_ocr_features(image_id: str, ocr_features_path: str) -> Dict[str, Any]:
    feature_ocr = np.load(ocr_features_path, allow_pickle=True).item()
    return feature_ocr.get(image_id, None)

class Vision_Encode_Ocr_Feature(nn.Module):
    def __init__(self, config: Dict):
        super(Vision_Encode_Ocr_Feature,self).__init__()
        self.sort_type = config.sort_type
        self.ocr_path = config.ocr_path
        self.scene_text_threshold = config.scene_text_threshold
        self.max_scene_text = config.max_scene_text
        self.global_ocr_features = None
        if os.path.isfile(self.ocr_path):
            self.global_ocr_features = np.load(self.ocr_path, allow_pickle=True).item()
         
    def forward(self, images: List[int]) -> List[Dict[str, Any]]:
        ocr_info = [self.load_ocr_features(image_id) for image_id in images]
        return ocr_info
    
    def pad_array(self, array: np.ndarray, max_len: int, value):
        if max_len == 0:
            array= np.zeros((0, array.shape[-1]))
        else:
            pad_value_array = np.zeros((max_len-array.shape[0], array.shape[-1])).fill(value)
            array = np.concatenate([array, pad_value_array], axis=0)
        return array

    def pad_tensor(self, tensor: torch.Tensor, max_len: int, value):
        if max_len == 0:
            tensor = torch.zeros((0, tensor.shape[-1]))
        else:
            pad_value_tensor = torch.zeros((max_len-tensor.shape[0], tensor.shape[-1])).fill_(value)
            tensor = torch.cat([tensor, pad_value_tensor], dim=0)
        return tensor

    def pad_list(self, list: List, max_len: int, value):
        pad_value_list = [value] * (max_len - len(list))
        list.extend(pad_value_list)
        return list

    def get_size_ocr(self, image_id: int): 
        if self.global_ocr_features is not None:
            features = self.global_ocr_features.get(str(image_id), None)
            if features is None:
                return torch.tensor([1,1,1,1])
            w,h=features['weight'],features['height']
            return torch.tensor([w,h,w,h])
        else:
            return torch.tensor([1,1,1,1])
    
    def convert_to_polygon(self, bbox, text):
        x1, y1, x2, y2 = bbox
        return [f'{text}',[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]

    def sorting_bounding_box(self, data):
        points=[self.convert_to_polygon(data['boxes'][i],data['texts'][i]) for i in range(len(data['boxes']))]
        points = list(map(lambda x:[x[0],x[1][0],x[1][2]],points))
        points_sum = list(map(lambda x: [x[0],x[1],sum(x[1]),x[2][1]],points))
        x_y_cordinate = list(map(lambda x: x[1],points_sum))
        final_sorted_list = []
        while True:
            try:
                new_sorted_text = []
                initial_value_A  = [i for i in sorted(enumerate(points_sum), key=lambda x:x[1][2])][0]
                threshold_value = abs(initial_value_A[1][1][1] - initial_value_A[1][3])
                threshold_value = (threshold_value/2) + 500
                del points_sum[initial_value_A[0]]
                del x_y_cordinate[initial_value_A[0]]
                # print(threshold_value)
                A = [initial_value_A[1][1]]
                K = list(map(lambda x:[x,abs(x[1]-initial_value_A[1][1][1])],x_y_cordinate))
                K = [[count,i]for count,i in enumerate(K)]
                K = [i for i in K if i[1][1] <= threshold_value]
                sorted_K = list(map(lambda x:[x[0],x[1][0]],sorted(K,key=lambda x:x[1][1])))
                B = []
                points_index = []
                for tmp_K in sorted_K:
                    points_index.append(tmp_K[0])
                    B.append(tmp_K[1])
                dist = distance.cdist(A,B)[0]
                d_index = [i for i in sorted(zip(dist,points_index), key=lambda x:x[0])]
                new_sorted_text.append(initial_value_A[1][0])

                index = []
                for j in d_index:
                    new_sorted_text.append(points_sum[j[1]][0])
                    index.append(j[1])
                for n in sorted(index, reverse=True):
                    del points_sum[n]
                    del x_y_cordinate[n]
                final_sorted_list.append(new_sorted_text)
            except Exception as e:
                # print(e)
                break

        combined_list = [item for sublist in final_sorted_list for item in sublist]
        new_index= [data['texts'].index(item) for item in combined_list]
        return combined_list, new_index
    
    def load_ocr_features(self, image_id: int) -> Dict[str, Any]:
        if self.global_ocr_features is not None:
            features = self.global_ocr_features.get(str(image_id), None)
        else:
            features = load_ocr_features(str(image_id), self.ocr_path)

        if features is None:
            return None

        features = {k: v for k, v in features.items()}

        for key, feature in features.items():
            if isinstance(feature, np.ndarray):
                features[key] = torch.tensor(feature)

        if self.sort_type == 'random':
            random_indices = list(range(len(features['scores'])))
            random.shuffle(random_indices)
            features['det_features'] = features['det_features'][random_indices]
            features['rec_features'] = features['rec_features'][random_indices]
            features['boxes'] = features['boxes'][random_indices]
            features['texts'] = [features['texts'][idx] for idx in random_indices]

        if self.sort_type=='score':
            features['scores']=torch.tensor(features['scores'])
            selected_indices = torch.where(features['scores'] > self.scene_text_threshold)[0]
            sorted_indices = torch.argsort(features['scores'][selected_indices], descending=True)
            new_ids = selected_indices[sorted_indices].tolist()
            features['det_features'] = features['det_features'][new_ids]
            features['rec_features'] = features['rec_features'][new_ids]
            features['boxes']=features['boxes'][new_ids]
            features['texts'] = [features['texts'][idx] for idx in new_ids]

        if self.sort_type=='top-left bottom-right':
            if len(features['texts'])>1:
                features["boxes"]=features["boxes"]*self.get_size_ocr(image_id)
                features['texts'], new_ids=self.sorting_bounding_box(features)
                features['det_features'] = features['det_features'][new_ids]
                features['rec_features'] = features['rec_features'][new_ids]
                features['boxes']=features['boxes'][new_ids]
                features["boxes"]=features["boxes"]/self.get_size_ocr(image_id)

        if self.sort_type is not None and self.sort_type not in ['random','score', 'top-left bottom-right']:
            raise ValueError("Invalid sort_type. Must be either 'score' or 'top-left bottom-right' or None ")

        if len(features['det_features'])>= self.max_scene_text:
            features['det_features']=features['det_features'][:self.max_scene_text]
            features['rec_features']=features['rec_features'][:self.max_scene_text]
            features['boxes']=features['boxes'][:self.max_scene_text]
        else:
            features['det_features'] = self.pad_tensor(features['det_features'], self.max_scene_text, 0.)
            features['rec_features'] = self.pad_tensor(features['rec_features'], self.max_scene_text, 0.)
            features['boxes'] = self.pad_tensor(features['boxes'], self.max_scene_text, 0.)

        ocr_info={
                "det_features": features["det_features"].float().detach().cpu(),
                "rec_features": features["rec_features"].float().detach().cpu(),
                "boxes": features["boxes"].float().detach().cpu(),
                'height': features['height'],
                'width': features['weight'],
                }
        return ocr_info

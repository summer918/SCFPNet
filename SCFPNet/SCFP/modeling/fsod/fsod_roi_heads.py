import inspect
import logging
import random
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
import random
from detectron2.config import configurable
from detectron2.layers import ShapeSpec, nonzero_tuple
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from .SCFP import make_stage
from functools import partial
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.roi_heads import ROIHeads
from .fsod_fast_rcnn import FsodFastRCNNOutputLayers
import cv2
import time
from detectron2.structures import Boxes, Instances
import math
import random

from timm.models.layers import to_2tuple, trunc_normal_

ROI_HEADS_REGISTRY = Registry("ROI_HEADS")
ROI_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""

logger = logging.getLogger(__name__)


def build_roi_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)


@ROI_HEADS_REGISTRY.register()
class FsodRes5ROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg)

        # fmt: off
        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales = (1.0 / input_shape[self.in_features[0]].stride,)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        self.mask_on = cfg.MODEL.MASK_ON
        self.freeze_roi_feature_extractor = cfg.MODEL.ROI_HEADS.FREEZE_ROI_FEATURE_EXTRACTOR
        self.only_train_norm = cfg.MODEL.ROI_HEADS.ONLY_TRAIN_NORM
       
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_one = nn.Conv2d(512, 64, 1)
        self.relu = nn.ReLU()
        self.conv_two = nn.Conv2d(64, 512, 1)
        self.sigmoid = nn.Sigmoid()
        assert not cfg.MODEL.KEYPOINT_ON
        assert len(self.in_features) == 1

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.apply(self._init_weights)
        self.branch_embed4, self.patch_embed4, self.block4, self.norm4, out_channels,self.attention_matrix = self._build_res5_block(
            cfg)

        print("是否冻结roi！", self.freeze_roi_feature_extractor)
        if self.freeze_roi_feature_extractor:
            print("冻结roi feature")
            self._freeze_roi_feature_extractor()
        if self.only_train_norm:
            self._only_train_norm()

        self.box_predictor = FsodFastRCNNOutputLayers(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1)
        )

        if isinstance(self.branch_embed4, nn.Parameter):
            self.branch_embed4.normal_(mean=0.0, std=0.02)
        elif isinstance(self.branch_embed4, nn.Embedding):
            self.branch_embed4.weight.data.normal_(mean=0.0, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Parameter):
            m.normal_(mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            m.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _freeze_roi_feature_extractor(self):
        print("冻结roi_feature")
        self.branch_embed4.eval()
        for param in self.branch_embed4.parameters():
            param.requires_grad = False

        self.patch_embed4.eval()
        for param in self.patch_embed4.parameters():
            param.requires_grad = False

        self.block4.eval()
        for param in self.block4.parameters():
            param.requires_grad = False

        self.norm4.eval()
        for param in self.norm4.parameters():
            param.requires_grad = False

    def _only_train_norm(self):
        for name, param in self.branch_embed4.named_parameters():
            if 'norm' not in name:
                param.requires_grad = False

        for name, param in self.patch_embed4.named_parameters():
            if 'norm' not in name:
                param.requires_grad = False

        for name, param in self.block4.named_parameters():
            if 'norm' not in name:
                param.requires_grad = False

        for name, param in self.norm4.named_parameters():
            if 'norm' not in name:
                param.requires_grad = False

    def _build_res5_block(self, cfg):
        backbone_type = cfg.MODEL.BACKBONE.TYPE
        if backbone_type == "pvt_v2_b2_li":
            branch_embed = nn.Embedding(2, 512)
            patch_embed, block, norm,attention_matrix  = make_stage(
                i=3,
                img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 320, 512],
                num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3],
                sr_ratios=[8, 4, 2, 1], num_stages=3, linear=True, pretrained=None
            )
            out_channels = 512
        elif backbone_type == "pvt_v2_b5":
            branch_embed = nn.Embedding(2, 512)
            patch_embed, block, norm = make_stage(
                i=3,
                img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 320, 512],
                num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3],
                sr_ratios=[8, 4, 2, 1], num_stages=3, linear=False, pretrained=None
            )
            out_channels = 512
        elif backbone_type == "pvt_v2_b4":
            branch_embed = nn.Embedding(2, 512)
            patch_embed, block, norm = make_stage(
                i=3,
                img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 320, 512],
                num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3],
                sr_ratios=[8, 4, 2, 1], num_stages=3, linear=False, pretrained=None
            )
            out_channels = 512
        elif backbone_type == "pvt_v2_b3":
            branch_embed = nn.Embedding(2, 512)
            patch_embed, block, norm = make_stage(
                i=3,
                img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 320, 512],
                num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3],
                sr_ratios=[8, 4, 2, 1], num_stages=3, linear=False, pretrained=None
            )
            out_channels = 512
        elif backbone_type == "pvt_v2_b2":
            branch_embed = nn.Embedding(2, 512)
            patch_embed, block, norm = make_stage(
                i=3,
                img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 320, 512],
                num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3],
                sr_ratios=[8, 4, 2, 1], num_stages=3, linear=False, pretrained=None
            )
            out_channels = 512
        elif backbone_type == "pvt_v2_b1":
            branch_embed = nn.Embedding(2, 512)
            patch_embed, block, norm = make_stage(
                i=3,
                img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 320, 512],
                num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2],
                sr_ratios=[8, 4, 2, 1], num_stages=3, linear=False, pretrained=None
            )
            out_channels = 512
        elif backbone_type == "pvt_v2_b0":
            branch_embed = nn.Embedding(2, 256)
            patch_embed, block, norm = make_stage(
                i=3,
                img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[32, 64, 160, 256],
                num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2],
                sr_ratios=[8, 4, 2, 1], num_stages=3, linear=False, pretrained=None
            )
            out_channels = 256
        else:
            print("do not support backbone type ", backbone_type)
            return None

        return branch_embed, patch_embed, block, norm, out_channels,attention_matrix

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        B = x.shape[0]
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x

    def featuremap_2_heatmap(self,feature_map):
        assert isinstance(feature_map, torch.Tensor)

        # 1*256*200*256 # feat的维度要求，四维
        feature_map = feature_map.detach()

        # 1*256*200*256->1*200*256
        heatmap = feature_map[:, 0, :, :] * 0
        for c in range(feature_map.shape[1]):
            heatmap += feature_map[:, c, :, :]
        heatmap = heatmap.cpu().numpy()
        heatmap = np.mean(heatmap, axis=0)

        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        return heatmap
    def show_k_feature_map(self,feature_map, k,id):
        print(feature_map.shape)
        
        feature_map = feature_map.squeeze(0)
        feature_map = feature_map.cpu().numpy()
        for index, feature_map_i in enumerate(feature_map):
            print(feature_map_i.shape)
            feature_map_i = np.array(feature_map_i * 255, dtype=np.uint8)
            feature_map_i = cv2.resize(feature_map_i, (224, 224), interpolation=cv2.INTER_NEAREST)
            if k == index + 1:
                feature_map_i=(feature_map_i/256).astype(np.uint8)
                print(feature_map_i.shape)
#                 feature_map_i = cv2.applyColorMap(feature_map_i, cv2.COLORMAP_JET)
                cv2.imwrite("{}_{}.jpg".format(id,str(index + 1)), feature_map_i)

    def _shared_roi_transform_mutual(self,box_features, support_box_features):
        x = box_features
        y = support_box_features

        B_x, _, H_x, W_x, = x.shape
        B_y, _, H_y, W_y = y.shape
 
        x, H_x, W_x = self.patch_embed4(x)
        y, H_y, W_y = self.patch_embed4(y)
        x_branch_embed = torch.zeros(x.shape[:-1], dtype=torch.long).cuda()
        x = x + self.branch_embed4(x_branch_embed)

        y_branch_embed = torch.ones(y.shape[:-1], dtype=torch.long).cuda()
        y = y + self.branch_embed4(y_branch_embed)
        # i=0
        # print(x.shape,y.shape)
        for blk in self.block4:
            # if i==2:
                x, y = blk(x, H_x, W_x, y, H_y, W_y)
            # else:
            #     x, y = blk(x, H_x, W_x, y, H_y, W_y, query_image, 3)
            # i+=1

        x = self.norm4(x)
        x = x.reshape(B_x, H_x, W_x, -1).permute(0, 3, 1, 2).contiguous()
        y = self.norm4(y)
        y = y.reshape(B_y, H_y, W_y, -1).permute(0, 3, 1, 2).contiguous()
        

#         y1=self.avg_pool(y)
 
#         y_tmp=self.sigmoid(y1)
 
#         x=x*y_tmp 
#         y_tmp = self.avg_pool(y)
#         y_tmp = self.conv_one(y_tmp)
       
#         y_tmp = self.conv_two(self.relu(y_tmp))
      
#         x = x * self.sigmoid(y_tmp)
#         x3=img*
        x, y = self.attention_matrix(x, y)
        x = x.reshape(B_x, H_x, W_x, -1).permute(0, 3, 1, 2).contiguous()
        y = y.reshape(B_y, H_y, W_y, -1).permute(0, 3, 1, 2).contiguous()
         
#         if id==15: 
#             y_ = y.repeat(B_x, 1, 1, 1)
#             y_=y_.flatten(2).transpose(1, 2).contiguous()
#             x_=x.flatten(2).transpose(1, 2).contiguous()
# #             print(y_.shape,x_.shape)
#             y_ = y_ / (torch.linalg.norm(y_, dim=-1, keepdim=True))  # 方差归一化，即除以各自的模
#             x_ = x_ / (torch.linalg.norm(x_, dim=-1, keepdim=True))
#             atten = x_.transpose(-2, -1) @ y_
#             atten=atten.softmax(dim=-1)
#             for i in range(atten.shape[0]):

#                 value,index=torch.max(atten[i],dim=-1)
#                 print(value,index)
#         if id==12:
# #             y_tmp=y_tmp[0]
#             for i in range(x.shape[0]):
# #                 print(x.shape,x[0].shape,y_tmp.shape)
#                 re=x[i]
#                 featuremap=re.unsqueeze(dim=0)
#             #         print(x.shape,img.shape)
#                 heatmap = self.featuremap_2_heatmap(featuremap)
#                     # 200*256->512*640
#                 heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的
#                     # 大小调整为与原始图像相同
#                 heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
#                     # 512*640*3
#                 heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原
#                     # 始图像
#             #         print(heatmap.shape,img.shape)
#                 superimposed_img = heatmap * 0.7 + 0.3 * img  # 热力图强度因子，修改参数，
#                     # 得到合适的热力图
#                 cv2.imwrite( f'羊{i}热力图.jpg',superimposed_img)  # 将图像保存
#             exit(0)
        return x, y

    def roi_pooling(self, features, boxes):
        box_features = self.pooler(
            [features[f] for f in self.in_features], boxes
        )

        return box_features  # feature_pooled

    def forward(self, images, features, support_box_features, proposals, targets=None):
        """
        See :meth:`ROIHeads.forward`.
        """

        del images
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets
        proposal_boxes = [x.proposal_boxes for x in proposals]
#         print(proposal_boxes.shape)
        box_features = self.roi_pooling(features, proposal_boxes)
#         print(box_features.shape)
        #         support_box_features1=support_box_features[0]
        # support_box_features1 = support_box_features.sum(dim=0).unsqueeze(0)
        # support_box_features=support_box_features1+self.new_s(support_box_features1)
        support_box_features = support_box_features.mean(0, True)
        box_features, support_box_features = self._shared_roi_transform_mutual(box_features, support_box_features)

        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features, support_box_features)

        return pred_class_logits, pred_proposal_deltas, proposals

    @torch.no_grad()
    def eval_with_support(self, images, query_features_dict, support_proposals_dict, support_box_features_dict):
        """
        See :meth:`ROIHeads.forward`.
        """
        # del images

        full_proposals_ls = []
        full_scores_ls = []
        full_bboxes_ls = []
        full_cls_ls = []
        cnt = 0
        
        for cls_id, proposals in support_proposals_dict.items():
            print(cls_id,len(proposals),support_box_features_dict[cls_id].shape )
             
            support_box_features = support_box_features_dict[cls_id].mean(0, True)

            proposals_ls = [Instances.cat([proposals[0], ])]
            full_proposals_ls.append(proposals[0])
            # [100,4]
            proposal_boxes = [x.proposal_boxes for x in proposals_ls]
            # torch.Size([100, 49, 512])
            box_features = self.roi_pooling(query_features_dict[cls_id], proposal_boxes)
            
            box_features, support_box_features = self._shared_roi_transform_mutual(images,cls_id,box_features, support_box_features)

            pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features, support_box_features)
            full_scores_ls.append(pred_class_logits)
            full_bboxes_ls.append(pred_proposal_deltas)
            full_cls_ls.append(torch.full_like(pred_class_logits[:, 0].unsqueeze(-1), cls_id).to(torch.int8))
            del box_features
            del support_box_features

            cnt += 1
        
        class_logits = torch.cat(full_scores_ls, dim=0)
        proposal_deltas = torch.cat(full_bboxes_ls, dim=0)
        pred_cls = torch.cat(full_cls_ls, dim=0)  # .unsqueeze(-1)

        predictions = class_logits, proposal_deltas
        proposals = [Instances.cat(full_proposals_ls)]
        pred_instances, _ = self.box_predictor.inference(pred_cls, predictions, proposals)
        pred_instances = self.forward_with_given_boxes(query_features_dict, pred_instances)
        
        return pred_instances, {}

    def forward_with_given_boxes(self, features, instances):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        if self.mask_on:
            features = [features[f] for f in self.in_features]
            x = self._shared_roi_transform(features, [x.pred_boxes for x in instances])
            return self.mask_head(x, instances)
        else:
            return instances
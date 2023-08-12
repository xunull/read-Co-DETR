import warnings

import torch
import torch.nn as nn

from mmdet.core import bbox2result
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
from mmdet.models.detectors.base import BaseDetector


@DETECTORS.register_module()
class CoDETR(BaseDetector):
    def __init__(self,
                 backbone,
                 neck=None,
                 # 四个头
                 query_head=None,
                 rpn_head=None,
                 roi_head=[None],
                 bbox_head=[None],

                 train_cfg=[None, None],
                 test_cfg=[None, None],
                 pretrained=[None, None],
                 init_cfg=None,
                 with_pos_coord=True,
                 with_attn_mask=True,
                 eval_module='detr',
                 eval_index=0):

        super(CoDETR, self).__init__(init_cfg)
        self.with_pos_coord = with_pos_coord
        self.with_attn_mask = with_attn_mask
        # Module for evaluation, ['detr', 'one-stage', 'two-stage']
        self.eval_module = eval_module
        # Module index for evaluation
        self.eval_index = eval_index
        self.backbone = build_backbone(backbone)

        head_idx = 0

        if neck is not None:
            self.neck = build_neck(neck)

        # CoDeformDETRHead
        if query_head is not None:
            query_head.update(
                train_cfg=train_cfg[head_idx] if (train_cfg is not None and train_cfg[head_idx] is not None) else None)
            query_head.update(test_cfg=test_cfg[head_idx])
            self.query_head = build_head(query_head)  # CoDeformDETRHead
            self.query_head.init_weights()
            head_idx += 1

        # RPNHead
        if rpn_head is not None:
            rpn_train_cfg = train_cfg[head_idx].rpn if (
                    train_cfg is not None and train_cfg[head_idx] is not None) else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg[head_idx].rpn)
            self.rpn_head = build_head(rpn_head_)
            self.rpn_head.init_weights()

        # CoStandardRoIHead
        self.roi_head = nn.ModuleList()
        for i in range(len(roi_head)):
            if roi_head[i]:
                rcnn_train_cfg = train_cfg[i + head_idx].rcnn if (
                        train_cfg and train_cfg[i + head_idx] is not None) else None
                roi_head[i].update(train_cfg=rcnn_train_cfg)
                roi_head[i].update(test_cfg=test_cfg[i + head_idx].rcnn)
                self.roi_head.append(build_head(roi_head[i]))
                self.roi_head[-1].init_weights()

        # CoATSSHead
        self.bbox_head = nn.ModuleList()
        for i in range(len(bbox_head)):
            if bbox_head[i]:
                bbox_head[i].update(train_cfg=train_cfg[i + head_idx + len(self.roi_head)] if (
                        train_cfg and train_cfg[i + head_idx + len(self.roi_head)] is not None) else None)
                bbox_head[i].update(test_cfg=test_cfg[i + head_idx + len(self.roi_head)])
                self.bbox_head.append(build_head(bbox_head[i]))
                self.bbox_head[-1].init_weights()

        self.head_idx = head_idx
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    # 是否有rpn，这个rpn应该是针对encoder
    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_query_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'query_head') and self.query_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None and len(self.roi_head) > 0

    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self, 'roi_head') and self.roi_head[0].with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self, 'roi_head') and self.roi_head is not None and len(self.roi_head) > 0)
                or (hasattr(self, 'bbox_head') and self.bbox_head is not None and len(self.bbox_head) > 0))

    # 抽取出特征
    def extract_feat(self, img, img_metas=None):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    # 用于计算网络计算量
    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # img [bs,3,h,w] 图像的尺寸
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape

        if not self.with_attn_mask:  # remove attn mask for LSJ
            for i in range(len(img_metas)):
                input_img_h, input_img_w = img_metas[i]['batch_input_shape']
                img_metas[i]['img_shape'] = [input_img_h, input_img_w, 3]

        # 提取特征 (经过backbone, 以及->token的变换)
        x = self.extract_feat(img, img_metas)

        losses = dict()

        # update loss的方法, 会更新key，key加上idx，同时有权重的处理
        def upd_loss(losses, idx, weight=1):
            new_losses = dict()
            for k, v in losses.items():
                new_k = '{}{}'.format(k, idx)
                if isinstance(v, list) or isinstance(v, tuple):
                    new_losses[new_k] = [i * weight for i in v]
                else:
                    new_losses[new_k] = v * weight
            return new_losses

        # CoDeformDETRHead 这个就是正常的DETR的transformer的过程
        # DETR encoder and decoder forward
        if self.with_query_head:
            # bbox_losses 是decoder最后一层的loss，encoder的loss，以及decoder其他层的loss
            # x是encoder的输出变换回特征图的形式
            bbox_losses, x = self.query_head.forward_train(x, img_metas,
                                                           gt_bboxes, gt_labels, gt_bboxes_ignore)
            losses.update(bbox_losses)

        # RPN forward and loss
        if self.with_rpn:
            # {'nms_pre': 4000, 'max_per_img': 1000, 'nms': {'type': 'nms', 'iou_threshold': 0.7}, 'min_bbox_size': 0}
            proposal_cfg = self.train_cfg[self.head_idx].get('rpn_proposal', self.test_cfg[self.head_idx].rpn)

            # RPNHead(
            #   (loss_cls): CrossEntropyLoss(avg_non_ignore=False)
            #   (loss_bbox): L1Loss()
            #   (rpn_conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            #   (rpn_cls): Conv2d(256, 9, kernel_size=(1, 1), stride=(1, 1))
            #   (rpn_reg): Conv2d(256, 36, kernel_size=(1, 1), stride=(1, 1))
            # )
            # init_cfg={'type': 'Normal', 'layer': 'Conv2d', 'std': 0.01}

            # proposal_list list [1000,5] ,list为bs， 1000个nms之后的结果，5是四个坐标值，以及前景的分数值
            # 返回的是rpn阶段的loss, 以及rpn后给出的proposals
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,  # encoder输出的特征图
                img_metas,
                gt_bboxes,
                gt_labels=None,  # rpn只需要box即可，无需类别
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)

            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        # 自定义的正样本选择
        positive_coords = []

        # 上面是RPN的处理，RPN处理之后，就会交给RoI，这里是FasterRCNN的流程
        # 当然还可以加入多个头，共用这一个rpn的结果

        # roi_head CoStandardRoIHead
        for i in range(len(self.roi_head)):
            # loss_cls, acc, loss_bbox, pos_coords
            roi_losses = self.roi_head[i].forward_train(x, img_metas,
                                                        proposal_list, # proposal_list 是rpn的结果
                                                        gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks,
                                                        **kwargs)

            # Customized Positive Queries Generation
            if self.with_pos_coord:
                # 加入到自定义的正样本选择
                positive_coords.append(roi_losses.pop('pos_coords'))
            else:
                # 不需要，那么就将pos_coords 删除
                if 'pos_coords' in roi_losses.keys():
                    tmp = roi_losses.pop('pos_coords')
            # upd_loss 会在loss的key后面补上idx
            roi_losses = upd_loss(roi_losses, idx=i)
            losses.update(roi_losses)

        # CoATSSHead
        # ATSS 单阶段检测，不需要RPN
        for i in range(len(self.bbox_head)):
            # 四个内容 loss_cls, loss_bbox, loss_centerness, pos_coords
            bbox_losses = self.bbox_head[i].forward_train(x, img_metas,
                                                          gt_bboxes, gt_labels, gt_bboxes_ignore)

            # Customized Positive Queries Generation, 与上面的方法类似
            if self.with_pos_coord:
                pos_coords = bbox_losses.pop('pos_coords')
                positive_coords.append(pos_coords)
            else:
                if 'pos_coords' in bbox_losses.keys():
                    tmp = bbox_losses.pop('pos_coords')

            bbox_losses = upd_loss(bbox_losses, idx=i + len(self.roi_head))
            losses.update(bbox_losses)

        # Customized Positive Queries Generation
        if self.with_pos_coord and len(positive_coords) > 0:
            # 每个辅助头都会生成一组
            for i in range(len(positive_coords)):
                # self.query_head 这个还是CoDeformDETRHead，还是那个transformer
                # 上面也调用过一个了，这里调用的是 forward_train_aux 方法，不再是forward_train了
                bbox_losses = self.query_head.forward_train_aux(x, img_metas, gt_bboxes,
                                                                gt_labels, gt_bboxes_ignore,
                                                                positive_coords[i], i)

                bbox_losses = upd_loss(bbox_losses, idx=i)

                losses.update(bbox_losses)

        return losses

    def simple_test_roi_head(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        if not self.with_attn_mask:  # remove attn mask for LSJ
            for i in range(len(img_metas)):
                input_img_h, input_img_w = img_metas[i]['batch_input_shape']
                img_metas[i]['img_shape'] = [input_img_h, input_img_w, 3]

        x = self.extract_feat(img, img_metas)
        if self.with_query_head:
            results = self.query_head.forward(x, img_metas)
            x = results[-2]
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head[self.eval_index].simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def simple_test_query_head(self, img, img_metas, proposals=None, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        index = 0
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        if not self.with_attn_mask:  # remove attn mask for LSJ
            for i in range(len(img_metas)):
                input_img_h, input_img_w = img_metas[i]['batch_input_shape']
                img_metas[i]['img_shape'] = [input_img_h, input_img_w, 3]

        x = self.extract_feat(img, img_metas)
        results_list = self.query_head.simple_test(
            x, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.query_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def simple_test_bbox_head(self, img, img_metas, proposals=None, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        if not self.with_attn_mask:  # remove attn mask for LSJ
            for i in range(len(img_metas)):
                input_img_h, input_img_w = img_metas[i]['batch_input_shape']
                img_metas[i]['img_shape'] = [input_img_h, input_img_w, 3]

        x = self.extract_feat(img, img_metas)
        if self.with_query_head:
            results = self.query_head.forward(x, img_metas)
            x = results[-2]
        results_list = self.bbox_head[self.eval_index].simple_test(
            x, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head[self.eval_index].num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.eval_module in ['detr', 'one-stage', 'two-stage']
        if self.with_bbox and self.eval_module == 'one-stage':
            return self.simple_test_query_head(img, img_metas, proposals, rescale)
        if self.with_roi_head and self.eval_module == 'two-stage':
            return self.simple_test_roi_head(img, img_metas, proposals, rescale)
        return self.simple_test_query_head(img, img_metas, proposals, rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.query_head, 'aug_test'), \
            f'{self.query_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.query_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.query_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas, with_nms=True):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.query_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape

        if len(outs) == 2:
            # add dummy score_factor
            outs = (*outs, None)
        # TODO Can we change to `get_bboxes` when `onnx_export` fail
        det_bboxes, det_labels = self.query_head.onnx_export(
            *outs, img_metas, with_nms=with_nms)

        return det_bboxes, det_labels

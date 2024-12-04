# The new config inherits a base config to highlight the necessary modification
_base_ = '../cascade_rcnn/cascade-mask-rcnn_r101_fpn_1x_coco.py'

# Modify dataset related settings
data_root = ''

train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_root=data_root,
        ann_file='train_new_p.json',
        data_prefix=dict(img='data/')))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='val_new_p.json',
        data_prefix=dict(img='data/')))

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='test_new_p.json',
        data_prefix=dict(img='data/')))

# Modify metric related settings
val_evaluator = dict(ann_file='val_new_p.json')
test_evaluator = dict(ann_file='test_new_p.json')

# We can use the pre-trained Mask RCNN model to obtain higher performance
# load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

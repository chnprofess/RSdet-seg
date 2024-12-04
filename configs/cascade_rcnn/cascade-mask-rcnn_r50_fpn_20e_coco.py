_base_ = [
    '../_base_/models/cascade-mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance_aug.py',
    '../_base_/schedules/schedule_20e_rs.py', '../_base_/default_runtime.py'
]

import logging
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../../")))

from deploy.py_infer.src.data_process.preprocess.transforms.layout_transforms import (
    image_norm,
    image_transpose,
    letterbox,
)
from mindocr.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from mindocr.data.transforms import create_transforms, run_transforms

_logger = logging.getLogger("mindocr")


class Preprocessor(object):
    def __init__(self, task="det", algo="DB", **kwargs):
        # algo = algo.lower()
        if task == "det":
            limit_side_len = kwargs.get("det_limit_side_len", 736)
            limit_type = kwargs.get("det_limit_type", "min")

            pipeline = [
                {"DecodeImage": {"img_mode": "RGB", "keep_ori": True, "to_float32": False}},
                # {'DetResize':
                #     {'target_size': [732, 1280],
                #      'keep_ratio': False,
                #      'target_limit_side': None, #target_limit_side, # TODO: add to arg
                #      'limit_type': None, #limit_type,
                #      'padding': False,
                #      'force_divisable': True,
                #      'divisor': 32
                #     }},
                {
                    "DetResize": {
                        "target_size": None,  # [ 1152, 2048 ]
                        "keep_ratio": True,
                        "limit_side_len": limit_side_len,
                        "limit_type": limit_type,
                        "padding": False,
                        "force_divisable": True,
                    }
                },
                {
                    "NormalizeImage": {
                        "bgr_to_rgb": False,
                        "is_hwc": True,
                        "mean": IMAGENET_DEFAULT_MEAN,
                        "std": IMAGENET_DEFAULT_STD,
                    }
                },
                {"ToCHWImage": None},
            ]

            if algo == "DB_PPOCRv3":
                pipeline = [
                    {"DecodeImage": {"img_mode": "RGB", "keep_ori": True, "to_float32": False}},
                    {"DetResize": {"limit_type": "min", "limit_side_len": 736}},
                    {
                        "NormalizeImage": {
                            "bgr_to_rgb": True,
                            "is_hwc": True,
                            "mean": IMAGENET_DEFAULT_MEAN,
                            "std": IMAGENET_DEFAULT_STD,
                        }
                    },
                    {"ToCHWImage": None},
                ]

            _logger.info(f"Pick optimal preprocess hyper-params for det algo {algo}:\n {pipeline[1]}")
            # TODO: modify the base pipeline for non-DBNet network if needed

        elif task == "rec":
            # defalut value if not claim in optim_hparam
            DEFAULT_PADDING = True
            DEFAULT_KEEP_RATIO = True
            # TODO: norm before padding is more reasonable but the previous models (trained before 2023.05.26)
            #  is based on norm in the end.
            DEFAULT_NORM_BEFORE_PAD = False

            # register optimal hparam for each model
            optimal_hparam = {
                # 'CRNN': dict(target_height=32, target_width=100, padding=True, keep_ratio=True, norm_before_pad=True),
                "CRNN": dict(target_height=32, target_width=100, padding=False, keep_ratio=False),
                "CRNN_CH": dict(target_height=32, taget_width=320, padding=True, keep_ratio=True),
                "RARE": dict(target_height=32, target_width=100, padding=False, keep_ratio=False),
                "RARE_CH": dict(target_height=32, target_width=320, padding=True, keep_ratio=True),
                "SVTR": dict(target_height=64, target_width=256, padding=False, keep_ratio=False),
                "SVTR_PPOCRv3_CH": dict(target_height=48, target_width=320, padding=True, keep_ratio=True),
            }

            # get hparam by combining default value, optimal value, and arg parser value. Prior: optimal value ->
            # parser value -> default value
            parsed_img_shape = kwargs.get("rec_image_shape", "3, 32, 320").split(",")
            parsed_height, parsed_width = int(parsed_img_shape[1]), int(parsed_img_shape[2])
            if algo in optimal_hparam:
                target_height = optimal_hparam[algo]["target_height"]
            else:
                target_height = parsed_height

            norm_before_pad = optimal_hparam[algo].get("norm_before_pad", DEFAULT_NORM_BEFORE_PAD)

            # TODO: update max_wh_ratio for each batch
            # max_wh_ratio = parsed_width /  float(parsed_height)
            # batch_num = kwargs.get('rec_batch_num', 1)
            batch_mode = kwargs.get("rec_batch_mode", False)  # and (batch_num > 1)
            if not batch_mode:
                # For single infer, the optimal choice is to resize the image to target height while keeping
                # aspect ratio, no padding. limit the max width.
                padding = False
                keep_ratio = True
                target_width = None
            else:
                # parse optimal hparam
                if algo in optimal_hparam:
                    padding = optimal_hparam[algo].get("padding", DEFAULT_PADDING)
                    keep_ratio = optimal_hparam[algo].get("keep_ratio", DEFAULT_KEEP_RATIO)
                    target_width = optimal_hparam[algo].get("target_width", parsed_width)
                else:
                    padding = DEFAULT_PADDING
                    keep_ratio = DEFAULT_KEEP_RATIO
                    target_width = parsed_width

            if (target_height != parsed_height) or (target_width != parsed_width):
                _logger.warning(
                    f"`rec_image_shape` {parsed_img_shape[1:]} dose not meet the network input requirement or "
                    f"is not optimal, which should be [{target_height}, {target_width}] under batch mode = {batch_mode}"
                )

            _logger.info(
                f"Pick optimal preprocess hyper-params for rec algo {algo}:\n"
                + "\n".join(
                    [
                        f"{k}:\t{str(v)}"
                        for k, v in dict(
                            target_height=target_height,
                            target_width=target_width,
                            padding=padding,
                            keep_ratio=keep_ratio,
                            norm_before_pad=norm_before_pad,
                        ).items()
                    ]
                )
            )

            pipeline = [
                {"DecodeImage": {"img_mode": "RGB", "keep_ori": True, "to_float32": False}},
                {
                    "RecResizeNormForInfer": {
                        "target_height": target_height,
                        "target_width": target_width,  # 100,
                        "keep_ratio": keep_ratio,
                        "padding": padding,
                        "norm_before_pad": norm_before_pad,
                        # 'interpolation': cv2.INTER_CUBIC
                    }
                },
                # {'NormalizeImage':
                #     {'bgr_to_rgb': False,
                #    'is_hwc': True,
                #    'mean': [127.0, 127.0, 127.0],
                #    'std': [127.0, 127.0, 127.0]}},
                {"ToCHWImage": None},
            ]
        elif task == "ser":
            class_path = kwargs.get("ser_class_dict_path", "mindocr/utils/dict/class_list_xfun.txt")
            pipeline = [
                {"DecodeImage": {"img_mode": "RGB", "infer_mode": True, "to_float32": False}},
                {
                    "VQATokenLabelEncode": {
                        "contains_re": False,
                        "infer_mode": True,
                        "algorithm": "LayoutXLM",
                        "class_path": class_path,
                        "order_method": "tb-yx",
                    }
                },
                {"VQATokenPad": {"max_seq_len": 512, "infer_mode": True, "return_attention_mask": True}},
                {"VQASerTokenChunk": {"infer_mode": True, "max_seq_len": 512}},
                {"LayoutResize": {"infer_mode": True, "size": [224, 224]}},
                {
                    "NormalizeImage": {
                        "infer_mode": True,
                        "bgr_to_rgb": False,
                        "is_hwc": True,
                        "mean": "imagenet",
                        "std": "imagenet",
                    }
                },
                {"ToCHWImage": None},
            ]
        elif task == "layout":
            if algo == "LAYOUTLMV3":
                pipeline = [
                    letterbox(scaleup=True, model_name="layoutlmv3"),
                    {
                        "NormalizeImage": {
                            "infer_mode": True,
                            "bgr_to_rgb": True,
                            "is_hwc": True,
                            "mean": [127.5, 127.5, 127.5],
                            "std": [127.5, 127.5, 127.5],
                        }
                    },
                    {"ToCHWImage": None},
                    {"ImageStridePad": {"stride": 32}},
                ]
            elif algo == "YOLOv8":
                pipeline = [
                    letterbox(scaleup=False),
                    image_norm(scale=255.0),
                    image_transpose(bgr2rgb=True, hwc2chw=True),
                ]
            else:
                raise ValueError(f"No preprocess config defined for {algo}. Please check the algorithm name.")
        elif task == "table":
            table_max_len = kwargs.get("table_max_len", 480)
            pipeline = [
                {"DecodeImage": {"img_mode": "RGB", "keep_ori": True, "to_float32": False}},
                {"ResizeTableImage": {"max_len": table_max_len}},
                {"PaddingTableImage": {"size": [table_max_len, table_max_len]}},
                {
                    "TableImageNorm": {
                        "std": [0.5, 0.5, 0.5],
                        "mean": [0.5, 0.5, 0.5],
                        "scale": "1./255.",
                        "order": "hwc",
                    }
                },
                {"ToCHWImage": None},
            ]

        elif task == "cls":
            pipeline = [
                {"DecodeImage": {"img_mode": "BGR", "to_float32": False}},
                {"Rotate90IfVertical": {"threshold": 2.0, "direction": "counterclockwise"}},
                {"RecResizeImg": {"image_shape": [48, 192], "padding": False}},
                {
                    "NormalizeImage": {
                        "bgr_to_rgb": True,
                        "is_hwc": True,
                        "mean": [127.0, 127.0, 127.0],
                        "std": [127.0, 127.0, 127.0],
                    }
                },
                {"ToCHWImage": None},
            ]

        self.pipeline = pipeline
        self.transforms = create_transforms(pipeline)

    # TODO: allow multiple image inputs and preprocess them with multi-thread
    def __call__(self, img_or_path):
        """
        Return:
            dict, preprocessed data containing keys:
                - image: np.array, transfomred image
                - image_ori: np.array, original image
                - shape: list of [ori_h, ori_w, scale_h, scale_w]
                and other keys added in transform pipeline.
        """
        if isinstance(img_or_path, str):
            data = {"img_path": img_or_path}
            output = run_transforms(data, self.transforms)
        elif isinstance(img_or_path, dict):
            output = run_transforms(img_or_path, self.transforms)
        else:
            data = {"image": img_or_path}
            data["image_ori"] = img_or_path.copy()  # TODO
            data["image_shape"] = img_or_path.shape
            output = run_transforms(data, self.transforms[1:])

        return output

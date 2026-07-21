from lightx2v_train.utils.registry import build_trainer

from .bernini_dmd import BerniniDmdTrainer, BerniniR2VDmdTrainer
from .dmd import DmdTrainer, LTX2T2AVArDmdTrainer, LTX2T2AVDmdTrainer, LingBotVideoDmdTrainer, VideoArDmdTrainer, VideoDmdTrainer
from .dopsd import DopsdTrainer
from .flow import FlowMatchingTrainer, LTX2T2AVFlowTrainer
from .tf import LTX2T2AVTeacherForcingTrainer, TFTrainer
from .wan22_dmd import Wan22A14BDmdTrainer

ARDmdTrainer = VideoArDmdTrainer

__all__ = [
    "build_trainer",
    "ARDmdTrainer",
    "BerniniDmdTrainer",
    "BerniniR2VDmdTrainer",
    "DmdTrainer",
    "FlowMatchingTrainer",
    "LTX2T2AVArDmdTrainer",
    "LTX2T2AVDmdTrainer",
    "LTX2T2AVFlowTrainer",
    "LTX2T2AVTeacherForcingTrainer",
    "LingBotVideoDmdTrainer",
    "TFTrainer",
    "VideoArDmdTrainer",
    "VideoDmdTrainer",
    "Wan22A14BDmdTrainer",
    "DopsdTrainer",
]

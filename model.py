import torch.nn as nn
from .networks import build_component_encoder
from .networks import build_component_decoder


class MaskGuidedPortraitEditing:
    def __init__(self, norm_type: str = 'instance'):
        self.norm_type = norm_type

        self.net_face_encoder = None
        self.net_hair_encoder = None
        self.net_mouth_encoder = None
        self.net_left_eye_encoder = None
        self.net_right_eye_encoder = None

        self.net_face_decoder = None
        self.net_hair_decoder = None
        self.net_mouth_decoder = None
        self.net_left_eye_decoder = None
        self.net_right_eye_decoder = None

        self.build_models()

    def build_models(self):
        # Local Embedding Sub-Network # Encoder
        self.net_face_encoder = build_component_encoder(input_shape=(3, 256, 256), norm_type=self.norm_type)
        self.net_hair_encoder = build_component_encoder(input_shape=(3, 256, 256), norm_type=self.norm_type)
        self.net_mouth_encoder = build_component_encoder(input_shape=(3, 80, 144), norm_type=self.norm_type)
        self.net_left_eye_encoder = build_component_encoder(input_shape=(3, 32, 40), norm_type=self.norm_type)
        self.net_right_eye_encoder = build_component_encoder(input_shape=(3, 32, 40), norm_type=self.norm_type)

        # Local Embedding Sub-Network # Decoder
        self.net_face_decoder = build_component_decoder(input_shape=(3, 256, 256), norm_type=self.norm_type)
        self.net_hair_decoder = build_component_decoder(input_shape=(3, 256, 256), norm_type=self.norm_type)
        self.net_mouth_decoder = build_component_decoder(input_shape=(3, 80, 144), norm_type=self.norm_type)
        self.net_left_eye_decoder = build_component_decoder(input_shape=(3, 32, 40), norm_type=self.norm_type)
        self.net_right_eye_decoder = build_component_decoder(input_shape=(3, 32, 40), norm_type=self.norm_type)

    def __str__(self):
        return "Mask-Guided Portrait Editing w/ cGAN"

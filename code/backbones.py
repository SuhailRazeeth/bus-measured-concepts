"""Frozen backbones. Each returns (model, preprocess(PIL)->tensor, embed(batch)->features, text_fn or None)."""
import torch

# Concept prompts (BI-RADS lexicon). Each concept = (suspicious phrase, benign-looking phrase).
CONCEPT_PROMPTS = {
    "shape_irregular": ("a breast mass with irregular shape", "a breast mass with oval or round shape"),
    "margin_noncirc": ("a breast mass with indistinct, angular or microlobulated margins",
                       "a breast mass with circumscribed smooth margins"),
    "margin_spiculated": ("a breast mass with spiculated margins", "a breast mass with smooth margins"),
    "malignant": ("a malignant breast tumor", "a benign breast lesion"),
}
TEMPLATES = {"MG": "a mammogram showing {}.", "US": "a breast ultrasound image showing {}."}


def load_backbone(name: str, device: str):
    if name == "biomedclip":
        import open_clip

        model, preprocess = open_clip.create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        tok = open_clip.get_tokenizer("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        model = model.to(device).eval()

        def embed(x):
            return model.encode_image(x)

        def text(strings):
            return model.encode_text(tok(strings, context_length=256).to(device))

        return model, preprocess, embed, text

    if name == "rad_dino":
        from transformers import AutoImageProcessor, AutoModel

        proc = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
        model = AutoModel.from_pretrained("microsoft/rad-dino").to(device).eval()

        def preprocess(img):
            return proc(images=img, return_tensors="pt")["pixel_values"][0]

        def embed(x):
            return model(pixel_values=x).pooler_output

        return model, preprocess, embed, None

    import timm

    timm_name = {"dinov2_b": "vit_base_patch14_dinov2.lvd142m", "resnet50_in": "resnet50.a1_in1k"}[name]
    kw = {"img_size": 224} if name == "dinov2_b" else {}  # DINOv2 defaults to 518 px; resample pos-embed to 224
    model = timm.create_model(timm_name, pretrained=True, num_classes=0, **kw).to(device).eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    if name == "dinov2_b":
        cfg.update(input_size=(3, 224, 224), crop_pct=1.0)
    preprocess = timm.data.create_transform(**cfg)
    return model, preprocess, (lambda x: model(x)), None


ALL = ["biomedclip", "rad_dino", "dinov2_b", "resnet50_in"]

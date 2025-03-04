from glide_text2im.gaussian_diffusion import get_named_beta_schedule
from glide_text2im.respace import SpacedDiffusion, space_timesteps
from glide_text2im.text2im_model import (
    InpaintText2ImUNet,
    SuperResInpaintText2ImUnet,
    SuperResText2ImUNet,
    Text2ImUNet,
)
from glide_text2im.tokenizer.bpe import get_encoder
from typing import Union


def model_and_diffusion_defaults():
    return dict(
        image_size=64,
        num_channels=192,
        num_res_blocks=3,
        channel_mult="",
        num_heads=1,
        num_head_channels=64,
        num_heads_upsample=-1,
        attention_resolutions="32,16,8",
        dropout=0.1,
        text_ctx=128,
        xf_width=512,
        xf_layers=16,
        xf_heads=8,
        xf_final_ln=True,
        xf_padding=True,
        diffusion_steps=1000,
        noise_schedule="squaredcos_cap_v2",
        timestep_respacing="",
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_fp16=True,
        cache_text_emb=False,
        inpaint=False,
        super_res=False,
    )


def model_and_diffusion_defaults_upsampler():
    result = model_and_diffusion_defaults()
    result.update(
        dict(
            image_size=256,
            num_res_blocks=2,
            noise_schedule="linear",
            super_res=True,
        )
    )
    return result


def create_model_and_diffusion(
    image_size: int,
    num_channels: int,
    num_res_blocks: int,
    channel_mult: Union[str, tuple],
    num_heads: int,
    num_head_channels: int,
    num_heads_upsample: int,
    attention_resolutions: str,
    dropout: float,
    text_ctx: int,
    xf_width: int,
    xf_layers: int,
    xf_heads: int,
    xf_final_ln: bool,
    xf_padding: bool,
    diffusion_steps: int,
    noise_schedule: str,
    timestep_respacing: str,
    use_scale_shift_norm: bool,
    resblock_updown: bool,
    use_fp16: bool,
    cache_text_emb: bool,
    inpaint: bool,
    super_res: bool,
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        text_ctx=text_ctx,
        xf_width=xf_width,
        xf_layers=xf_layers,
        xf_heads=xf_heads,
        xf_final_ln=xf_final_ln,
        xf_padding=xf_padding,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        cache_text_emb=cache_text_emb,
        inpaint=inpaint,
        super_res=super_res,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        noise_schedule=noise_schedule,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def create_model(
    image_size: int,                    # e.g. 64
    num_channels: int,                  # e.g. 192
    num_res_blocks: int,                # e.g. 3
    channel_mult: Union[str, tuple],    # e.g. "", (1, 2, 3, 4)
    attention_resolutions: str,         # e.g. "32,16,8"
    num_heads: int,                     # e.g. 1
    num_head_channels: int,             # e.g. 64
    num_heads_upsample: int,            # e.g. -1
    use_scale_shift_norm: bool,
    dropout: float,                     # e.g. 0.1
    text_ctx: int,                      # e.g. 128
    xf_width: int,                      # e.g. 512
    xf_layers: int,                     # e.g. 16
    xf_heads: int,                      # e.g. 8
    xf_final_ln: bool,
    xf_padding: bool,
    resblock_updown: bool,
    use_fp16: bool,
    cache_text_emb: bool,
    inpaint: bool,
    super_res: bool,
):
    if channel_mult == "":
        if image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))
        assert 2 ** (len(channel_mult) + 2) == image_size

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    if inpaint and super_res:
        model_cls = SuperResInpaintText2ImUnet
    elif inpaint:
        model_cls = InpaintText2ImUNet
    elif super_res:
        model_cls = SuperResText2ImUNet
    else:
        model_cls = Text2ImUNet
    return model_cls(
        text_ctx=text_ctx,
        xf_width=xf_width,
        xf_layers=xf_layers,
        xf_heads=xf_heads,
        xf_final_ln=xf_final_ln,
        tokenizer=get_encoder(),
        xf_padding=xf_padding,
        in_channels=3,
        model_channels=num_channels,
        out_channels=6,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        cache_text_emb=cache_text_emb,
    )


def create_gaussian_diffusion(
    steps:int,                  # e.g. 1000
    noise_schedule:str,         # e.g. "squaredcos_cap_v2"
    timestep_respacing:str,     # e.g. "200"
):
    betas = get_named_beta_schedule(noise_schedule, steps)
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
    )

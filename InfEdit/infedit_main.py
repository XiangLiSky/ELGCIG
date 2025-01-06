from diffusers import LCMScheduler
from pipeline_ead import EditPipeline
import os
import torch
from PIL import Image
import torch.nn.functional as nnf
from typing import Optional, Union, Tuple, List, Callable, Dict
import abc
import numpy as np
import math
import ptp_utils
import seq_aligner
import utils

LOW_RESOURCE = False
MAX_NUM_WORDS = 77

is_colab = utils.is_google_colab()
colab_instruction = "" if is_colab else """
Colab Instuction"""

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id_or_path = "SimianLuo/LCM_Dreamshaper_v7"
device_print = "GPU 🔥" if torch.cuda.is_available() else "CPU 🥶"
device = "cuda" if torch.cuda.is_available() else "cpu"

if is_colab:
    scheduler = LCMScheduler.from_config(model_id_or_path, subfolder="scheduler")
    pipe = EditPipeline.from_pretrained(model_id_or_path, scheduler=scheduler, torch_dtype=torch_dtype)
else:
    scheduler = LCMScheduler.from_config(model_id_or_path, use_auth_token=os.environ.get("USER_TOKEN"),
                                         subfolder="scheduler")
    pipe = EditPipeline.from_pretrained(model_id_or_path, use_auth_token=os.environ.get("USER_TOKEN"),
                                        scheduler=scheduler, torch_dtype=torch_dtype)

tokenizer = pipe.tokenizer
encoder = pipe.text_encoder

if torch.cuda.is_available():
    pipe = pipe.to("cuda")


class InfEditPipe:

    def __init__(
            self,
            image_path: Optional[str] = "",
            source_prompt: Optional[str] = "",
            target_prompt: Optional[str] = "",
            seed: Optional[int] = 42,
            width: Optional[int] = 512,
            height: Optional[int] = 512,
            denoise: Optional[bool] = False,
            strength: Optional[float] = 0.7,
            num_inference_steps: Optional[int] = 25,
            guidance_s: Optional[float] = 1.0,
            guidance_t: Optional[float] = 1.5,
            cross_replace_steps: Optional[float] = 0.65,
            self_replace_steps: Optional[float] = 0.7,
            thresh_e: Optional[float] = 0.65,
            thresh_m: Optional[float] = 0.65,
            local: Optional[str] = "",
            mutual: Optional[str] = "",
            positive_prompt: Optional[str] = "",
            negative_prompt: Optional[str] = ""
    ):
        self.image_path = image_path
        self.img = Image.open(image_path)
        self.source_prompt = source_prompt
        self.target_prompt = target_prompt
        self.seed = seed
        self.width = width
        self.height = height
        self.denoise = denoise
        self.strength = strength
        self.num_inference_steps = num_inference_steps
        self.guidance_s = guidance_s
        self.guidance_t = guidance_t
        self.cross_replace_steps = cross_replace_steps
        self.self_replace_steps = self_replace_steps
        self.thresh_e = thresh_e
        self.thresh_m = thresh_m
        self.mutual = mutual
        self.local = local
        self.positive_prompt = positive_prompt
        self.negative_prompt = negative_prompt

    def inf_edit(self):
        # 从params字典中提取所有参数
        torch.manual_seed(self.seed)
        ratio = min(self.height / self.img.height, self.width / self.img.width)
        self.img = self.img.resize((int(self.img.width * ratio), int(self.img.height * ratio)))
        if not self.denoise:
            self.strength = 1
        num_denoise_num = math.trunc(self.num_inference_steps * self.strength)
        num_start = self.num_inference_steps - num_denoise_num

        local_blend = LocalBlend(self.thresh_e, self.thresh_m, save_inter=False)
        controller = AttentionRefine(
            [self.source_prompt, self.target_prompt],
            [[self.local, self.mutual]],
            self.num_inference_steps,
            num_start,
            cross_replace_steps=self.cross_replace_steps,
            self_replace_steps=self.self_replace_steps,
            local_blend=local_blend
        )
        ptp_utils.register_attention_control(pipe, controller)

        results = pipe(prompt=self.target_prompt,
                       source_prompt=self.source_prompt,
                       positive_prompt=self.positive_prompt,
                       negative_prompt=self.negative_prompt,
                       image=self.img,
                       num_inference_steps=self.num_inference_steps,
                       eta=1,
                       strength=self.strength,
                       guidance_scale=self.guidance_t,
                       source_guidance_scale=self.guidance_s,
                       denoise_model=self.denoise,
                       callback=controller.step_callback
                       )
        # if hasattr(results, 'images') and len(results.images) > 0:
        #     result_img = results.images[0]  # 获取第一张图像
        #     result_img.save("/home/user/Pictures/edited_image.jpg")  # 保存图像
        #     print("Image processing complete and saved.")
        # else:
        #     print("No images were generated.")

        return results


class LocalBlend:

    def get_mask(self, x_t, maps, word_idx, thresh, i):
        maps = maps * word_idx.reshape(1, 1, 1, 1, -1)
        maps = (maps[:, :, :, :, 1:self.len - 1]).mean(0, keepdim=True)
        maps = (maps).max(-1)[0]
        maps = nnf.interpolate(maps, size=(x_t.shape[2:]))
        maps = maps / maps.max(2, keepdim=True)[0].max(3, keepdim=True)[0]
        mask = maps > thresh
        return mask

    def save_image(self, mask, i, caption):
        image = mask[0, 0, :, :]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.cpu().numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        if not os.path.exists(f"inter/{caption}"):
            os.mkdir(f"inter/{caption}")
        ptp_utils.save_images(image, f"inter/{caption}/{i}.jpg")

    def __call__(self, i, x_s, x_t, x_m, attention_store, alpha_prod, temperature=0.15, use_xm=False):
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        h, w = x_t.shape[2], x_t.shape[3]
        h, w = ((h + 1) // 2 + 1) // 2, ((w + 1) // 2 + 1) // 2
        maps = [
            item.reshape(2, -1, 1, h // int((h * w / item.shape[-2]) ** 0.5), w // int((h * w / item.shape[-2]) ** 0.5),
                         MAX_NUM_WORDS) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps_s = maps[0, :]
        maps_m = maps[1, :]
        thresh_e = temperature / alpha_prod ** (0.5)
        if thresh_e < self.thresh_e:
            thresh_e = self.thresh_e
        thresh_m = self.thresh_m
        mask_e = self.get_mask(x_t, maps_m, self.alpha_e, thresh_e, i)
        mask_m = self.get_mask(x_t, maps_s, (self.alpha_m - self.alpha_me), thresh_m, i)
        mask_me = self.get_mask(x_t, maps_m, self.alpha_me, self.thresh_e, i)
        if self.save_inter:
            self.save_image(mask_e, i, "mask_e")
            self.save_image(mask_m, i, "mask_m")
            self.save_image(mask_me, i, "mask_me")

        if self.alpha_e.sum() == 0:
            x_t_out = x_t
        else:
            x_t_out = torch.where(mask_e, x_t, x_m)
        x_t_out = torch.where(mask_m, x_s, x_t_out)
        if use_xm:
            x_t_out = torch.where(mask_me, x_m, x_t_out)

        return x_m, x_t_out

    def __init__(self, thresh_e=0.3, thresh_m=0.3, save_inter=False):
        self.thresh_e = thresh_e
        self.thresh_m = thresh_m
        self.save_inter = save_inter

    def set_map(self, ms, alpha, alpha_e, alpha_m, len):
        self.m = ms
        self.alpha = alpha
        self.alpha_e = alpha_e
        self.alpha_m = alpha_m
        alpha_me = alpha_e.to(torch.bool) & alpha_m.to(torch.bool)
        self.alpha_me = alpha_me.to(torch.float)
        self.len = len


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers // 2 + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class EmptyControl(AttentionControl):

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn

    def self_attn_forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        b = q.shape[0] // num_heads
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        return out


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class AttentionControlEdit(AttentionStore, abc.ABC):

    def step_callback(self, i, t, x_s, x_t, x_m, alpha_prod):
        if (self.local_blend is not None) and (i > 0):
            use_xm = (self.cur_step + self.start_steps + 1 == self.num_steps)
            x_m, x_t = self.local_blend(i, x_s, x_t, x_m, self.attention_store, alpha_prod, use_xm=use_xm)
        return x_m, x_t

    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        b = q.shape[0] // num_heads

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        return out

    def self_attn_forward(self, q, k, v, num_heads):
        if q.shape[0] // num_heads == 3:
            if (self.self_replace_steps <= ((self.cur_step + self.start_steps + 1) * 1.0 / self.num_steps)):
                q = torch.cat([q[:num_heads * 2], q[num_heads:num_heads * 2]])
                k = torch.cat([k[:num_heads * 2], k[:num_heads]])
                v = torch.cat([v[:num_heads * 2], v[:num_heads]])
            else:
                q = torch.cat([q[:num_heads], q[:num_heads], q[:num_heads]])
                k = torch.cat([k[:num_heads], k[:num_heads], k[:num_heads]])
                v = torch.cat([v[:num_heads * 2], v[:num_heads]])
            return q, k, v
        else:
            qu, qc = q.chunk(2)
            ku, kc = k.chunk(2)
            vu, vc = v.chunk(2)
            if (self.self_replace_steps <= ((self.cur_step + self.start_steps + 1) * 1.0 / self.num_steps)):
                qu = torch.cat([qu[:num_heads * 2], qu[num_heads:num_heads * 2]])
                qc = torch.cat([qc[:num_heads * 2], qc[num_heads:num_heads * 2]])
                ku = torch.cat([ku[:num_heads * 2], ku[:num_heads]])
                kc = torch.cat([kc[:num_heads * 2], kc[:num_heads]])
                vu = torch.cat([vu[:num_heads * 2], vu[:num_heads]])
                vc = torch.cat([vc[:num_heads * 2], vc[:num_heads]])
            else:
                qu = torch.cat([qu[:num_heads], qu[:num_heads], qu[:num_heads]])
                qc = torch.cat([qc[:num_heads], qc[:num_heads], qc[:num_heads]])
                ku = torch.cat([ku[:num_heads], ku[:num_heads], ku[:num_heads]])
                kc = torch.cat([kc[:num_heads], kc[:num_heads], kc[:num_heads]])
                vu = torch.cat([vu[:num_heads * 2], vu[:num_heads]])
                vc = torch.cat([vc[:num_heads * 2], vc[:num_heads]])

            return torch.cat([qu, qc], dim=0), torch.cat([ku, kc], dim=0), torch.cat([vu, vc], dim=0)

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        if is_cross:
            h = attn.shape[0] // self.batch_size
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce, attn_masa = attn[0], attn[1], attn[2]
            attn_replace_new = self.replace_cross_attention(attn_masa, attn_repalce)
            attn_base_store = self.replace_cross_attention(attn_base, attn_repalce)
            if (self.cross_replace_steps >= ((self.cur_step + self.start_steps + 1) * 1.0 / self.num_steps)):
                attn[1] = attn_base_store
            attn_store = torch.cat([attn_base_store, attn_replace_new])
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
            attn_store = attn_store.reshape(2 * h, *attn_store.shape[2:])
            super(AttentionControlEdit, self).forward(attn_store, is_cross, place_in_unet)
        return attn

    def __init__(self, prompts, num_steps: int, start_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts) + 1
        self.self_replace_steps = self_replace_steps
        self.cross_replace_steps = cross_replace_steps
        self.num_steps = num_steps
        self.start_steps = start_steps
        self.local_blend = local_blend


class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device).to(torch_dtype)


class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_masa, att_replace):
        attn_masa_replace = attn_masa[:, :, self.mapper].squeeze()
        attn_replace = attn_masa_replace * self.alphas + \
                       att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(self, prompts, prompt_specifiers, num_steps: int, start_steps: int, cross_replace_steps: float,
                 self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, start_steps, cross_replace_steps, self_replace_steps,
                                              local_blend)
        self.mapper, alphas, ms, alpha_e, alpha_m = seq_aligner.get_refinement_mapper(prompts, prompt_specifiers,
                                                                                      tokenizer, encoder, device)
        self.mapper, alphas, ms = self.mapper.to(device), alphas.to(device).to(torch_dtype), ms.to(device).to(
            torch_dtype)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])
        self.ms = ms.reshape(ms.shape[0], 1, 1, ms.shape[1])
        ms = ms.to(device)
        alpha_e = alpha_e.to(device)
        alpha_m = alpha_m.to(device)
        t_len = len(tokenizer(prompts[1])["input_ids"])
        self.local_blend.set_map(ms, alphas, alpha_e, alpha_m, t_len)


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float], Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch_dtype)
    for word in word_select:
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values
    return equalizer


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    editor = InfEditPipe(
        image_path="/home/user/Experiment/Lance/datasets/hardImageNet/val/n04162706/ILSVRC2012_val_00005254.JPEG",
        source_prompt="A photo of a woman sitting in the back seat of a car with her hand on the steering wheel.",
        target_prompt="A photo of a woman sitting in the back seat of a vintage car with her hand on the steering wheel.")

    editor.inf_edit()


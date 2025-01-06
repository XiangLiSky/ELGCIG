# Copyright 2023 the LANCE team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
import os
import warnings
import time
import json
import functools
import argparse
from typing import Dict, Optional
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger

torch.backends.cuda.matmul.allow_tf32 = True
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDIMScheduler
from lance.utils.misc_utils import *
from lance.utils.edit_utils import *
from InfEdit.pipeline_ead import EditPipeline
from diffusers import LCMScheduler


class ImageEditor:
    def __init__(
            self,
            args: argparse.Namespace,
            device: torch.device,
            self_replace_steps_range: Optional[list] = [0.4, 0.5, 0.6, 0.7, 0.8],
            cross_replace_steps: Optional[dict] = {"default_": 0.8},
            similarity_metric: Optional[ClipSimilarity] = ClipSimilarity(),
            text_similarity_threshold: Optional[float] = 0.5,
            verbose: Optional[bool] = False,
            edit_word_weight: Optional[float] = 2.0,
            clip_img_thresh: Optional[float] = 0.7,
            clip_thresh: Optional[float] = 0.2,
            clip_dir_thresh: Optional[float] = 0.2,
    ):
        """
        Initialize image editor
        Args:
            args: Command line arguments from argparse
            device: Device to run model on
            self_replace_steps_range: Range of self replace steps to use. Defaults to [0.4, 0.5, 0.6, 0.7, 0.8].
            cross_replace_steps: Dictionary mapping image names to cross replace steps. Defaults to {"default_": 0.8}.
            similarity_metric: Similarity metric to use. Defaults to ClipSimilarity().
            text_similarity_threshold: Similarity threshold between . Defaults to 0.7
            verbose: Logging verbosity. Defaults to False.
            edit_word_weight: Edit word weight. Defaults to 2.0.
            clip_img_thresh: Image similarity threshold. Defaults to 0.7.
            clip_thresh: Text similarity threshold. Defaults to 0.2.
            clip_dir_thresh: Directional similarity threshold. Defaults to 0.2.
        """
        self.args = args
        self.device = device
        self.self_replace_steps_range = self_replace_steps_range
        self.cross_replace_steps = cross_replace_steps
        self.clip_similarity = similarity_metric
        self.text_similarity_threshold = text_similarity_threshold
        self.verbose = verbose
        self.edit_word_weight = edit_word_weight
        self.clip_img_thresh = clip_img_thresh
        self.clip_thresh = clip_thresh
        self.clip_dir_thresh = clip_dir_thresh

        model_id_or_path = "SimianLuo/LCM_Dreamshaper_v7"

        if is_colab:
            scheduler = LCMScheduler.from_config(model_id_or_path, subfolder="scheduler")
            self.model = EditPipeline.from_pretrained(model_id_or_path, scheduler=scheduler, torch_dtype=torch_dtype)
        else:
            scheduler = LCMScheduler.from_config(model_id_or_path, use_auth_token=os.environ.get("USER_TOKEN"),
                                                 subfolder="scheduler")
            self.model = EditPipeline.from_pretrained(model_id_or_path, use_auth_token=os.environ.get("USER_TOKEN"),
                                                      scheduler=scheduler, torch_dtype=torch_dtype)

    def edit(
            self,
            img_path: str,
            out_path: str,
            cls_name: str,
            cap: str,
            edited_cap_dicts: List[Dict[str, str]],
    ):

        """
        Edit image
        Args:
            img_path: Path to original image
            out_path: Path to save edited image
            cls_name: Class name
            cap: Caption of image
            edited_cap_dicts: Edited captions of image
        """
        if self.verbose:
            logger.info("Editing image\n")
            logger.info("------------------------------------------------\n")

        img_name = img_path.split("/")[-1].split(".")[0]

        path = os.path.join(out_path)
        os.makedirs(path, exist_ok=True)

        prompt_dict = {
            "caption": cap,
            "image": out_path.split("/")[-1],
            "edits": {},
        }
        # Filter out edits that are more similar to the ground truth class than a threshold
        edited_cap_dicts_filtered = [
            edited_cap
            for edited_cap in edited_cap_dicts
            if (
                       self.clip_similarity.text_similarity(
                           [edited_cap["original"].strip().lower()], [cls_name]
                       )
                       < self.text_similarity_threshold
               )
               or (edited_cap["original"].strip().lower() == "")
        ]
        if len(edited_cap_dicts_filtered) == 0:
            if self.verbose:
                logger.warning(
                    "All target words are too similar to ground truth class. \
                    Skipping this image. Increase text_similarity_threshold if you \
                    want to force an edit."
                )
            return

        total_memory = (torch.cuda.mem_get_info()[0]) // 10 ** 9
        edit_batch_size = max(total_memory // 8, 1)

        try:
            for ix in range(0, len(edited_cap_dicts_filtered), edit_batch_size):
                edited_cap_dicts_filtered_curr = edited_cap_dicts_filtered[
                                                 ix: min(ix + edit_batch_size, len(edited_cap_dicts_filtered))
                                                 ]
                edited_caps = [
                    edited_cap["edited_caption"]
                    for edited_cap in edited_cap_dicts_filtered_curr
                ]
                originals = [
                    edited_cap["original"] for edited_cap in edited_cap_dicts_filtered_curr
                ]
                edits = [
                    edited_cap["edit"] for edited_cap in edited_cap_dicts_filtered_curr
                ]
                edit_types = [
                    edited_cap["perturbation_type"]
                    for edited_cap in edited_cap_dicts_filtered_curr
                ]
                edit_concat = "".join(
                    ["{}->{}\n".format(a, b) for a, b in zip(originals, edits)]

                )

                if self.verbose:
                    logger.info(f"Original prompt: {cap}")
                    logger.info(edit_concat)
                cur_prompts = [cap]
                ori_prompt = [cap]
                cur_prompts.extend(edited_caps)
                if self.verbose:
                    logger.info(f"Running sweep over editing hyperparams:\n")

                    # Attention replace edit only possible if lengths are identical
                    # length_check = [
                    #     len(ecap.split(" ")) == len(cap.split(" ")) for ecap in edited_caps
                    # ]
                    # attention_replace_edit = functools.reduce(lambda a, b: a & b, length_check)

                    # attention_replace_edit = False
                    # for self_replace_steps in self.self_replace_steps_range:
                    #     controller = make_controller(
                    #         cur_prompts,
                    #         self.device,
                    #         self.model.tokenizer,
                    #         attention_replace_edit,
                    #         self.cross_replace_steps,
                    #         float(self_replace_steps),
                    #         blend_words,
                    #         eq_params,
                    #     )
                    #     if self.verbose:
                    #         logger.debug(f"self_replace_steps={self_replace_steps}")
                    images = []
                    images.append(Image.open(img_path))
                    for cur_prompt in cur_prompts[1:]:
                        image = run_and_display(
                            img_path,
                            ori_prompt,
                            cur_prompt,
                            verbose=False,
                        )

                        if self.verbose:
                            logger.warning(
                                f"=> Image `{img_name}' already processed"
                            )

                        images.append(image)
                    ori_out_path = os.path.splitext(out_path)[
                                       0
                                   ] + "/{}.jpeg".format("original")

                    images[0].save(ori_out_path)
                    tns1 = transforms.ToTensor()(images[0]).unsqueeze(0).to(self.device)
                    tns2 = torch.cat(
                        [
                            transforms.ToTensor()(images[ix]).unsqueeze(0)
                            for ix in range(1, len(images))
                        ],
                        dim=0,
                    ).to(self.device)

                    if self.verbose:
                        logger.debug(f"Evaluating edit quality and consistency\t")
                    (
                        clip_sim_0,
                        clip_sim_1,
                        clip_sim_dir,
                        clip_sim_image,
                    ) = self.clip_similarity(tns1, tns2, [cap], edited_caps)
                    best_sim_dir = [-1 for _ in range(len(clip_sim_dir))]

                    for ix in range(len(clip_sim_dir)):
                        prediction_is_consistent = (
                            self.clip_similarity.pred_consistency(  # predictive consistency
                                tns2[ix: ix + 1], originals[ix], edits[ix]
                            )
                        )

                        if self.verbose:
                            logger.debug(
                                "[Metrics] I1I2={:.2f} I1T1={:.2f} I2T2={:.2f} <I1I2, T1T2>={:.2f} PC={}".format(
                                    clip_sim_image[ix].item(),
                                    clip_sim_0.item(),
                                    clip_sim_1[ix].item(),
                                    clip_sim_dir[ix].item(),
                                    prediction_is_consistent,
                                )
                            )

                        if (
                                clip_sim_image[ix]
                                >= self.clip_img_thresh  # image-image similarity
                                and clip_sim_0 >= self.clip_thresh  # image-text similarity
                                and clip_sim_1[ix] >= self.clip_thresh  # image-text similarity
                                and clip_sim_dir[ix]
                                >= self.clip_dir_thresh  # clip directional similarity
                                and clip_sim_dir[ix]
                                > best_sim_dir[ix]  # clip directional similarity
                                and prediction_is_consistent
                        ):
                            best_sim_dir[ix] = clip_sim_dir[ix]
                            edited_image = images[ix + 1]
                            full_out_path = os.path.splitext(out_path)[
                                                0
                                            ] + "/{}.jpeg".format(
                                str("_".join(edits[ix].split(" "))),
                            )

                            edited_image.save(full_out_path)
                            prompt_dict["edits"][full_out_path] = {
                                "edited_caption": edited_caps[ix],
                                "original": originals[ix],
                                "edit": edits[ix],
                                "edit_type": edit_types[ix],
                            }
                            if self.verbose:
                                logger.info(
                                    f"Saved edited images to {os.path.splitext(out_path)[0]}\t"
                                )

            with open(os.path.splitext(out_path)[0] + "/prompt_dict.json", "w") as f:
                json.dump(prompt_dict, f, indent=4)

        except Exception as e:
            logger.error(f"Critical error in image edit loop: {e}")

# if __name__ == "__main__":
#     # example usage
#     args = argparse.Namespace(
#         image_path="./data/gnochi_mirror.jpeg",
#         original_caption="a cat sitting next to a mirror",
#         out_path="outputs",
#         save_inversion=True,
#         ldm_type="stable_diffusion_v1_4",
#         edit_word_weight=2.0,
#         clip_img_thresh=-0.5,
#         clip_thresh=-0.5,
#         clip_dir_thresh=-0.5,
#     )
#     accelerator = Accelerator()
#     VERBOSE = True
#     if VERBOSE:
#         logger = get_logger()
#
#     image_editor = ImageEditor(
#         args,
#         device=accelerator.device,
#         similarity_threshold=0.5,
#         ldm_type=args.ldm_type,
#         verbose=VERBOSE,
#         self_replace_steps_range=[0.5],
#     )
#     edit_path = os.path.join(args.out_path, "gnochi_mirror", "edited")
#     os.makedirs(edit_path, exist_ok=True)
#     _, _, _, x_t, uncond_embeddings = image_editor.invert(
#         args.image_path, args.original_caption, args.out_path
#     )
#
#     new_prompt = args.original_caption.replace("cat", "tiger")
#     image_editor.edit(
#         edit_path,
#         "mirror",
#         x_t,
#         uncond_embeddings,
#         args.original_caption,
#         [
#             {
#                 "original_caption": args.original_caption,
#                 "edited_caption": new_prompt,
#                 "original": "cat",
#                 "edit": "tiger",
#             },
#             {
#                 "original_caption": args.original_caption,
#                 "edited_caption": args.original_caption.replace(
#                     "cat", "silver cat sculpture"
#                 ),
#                 "original": "cat",
#                 "edit": "silver cat sculpture",
#             },
#             {
#                 "original_caption": args.original_caption,
#                 "edited_caption": args.original_caption.replace(
#                     "a cat", "watercolor painting of a cat"
#                 ),
#                 "original": "a cat",
#                 "edit": "watercolor painting of a cat",
#             },
#         ],
#     )

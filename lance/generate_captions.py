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
import base64
import requests
import time
import argparse
from typing import Optional
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger

from lance.utils.misc_utils import *


class CaptionGenerator:
    def __init__(
            self,
            args: argparse.Namespace,
            device: torch.device,
            verbose: bool = False,
            name: Optional[str] = "gpt4_vision",
            api_key: str = "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",
            model_type: Optional[str] = "gpt-4o",
            repetition_penalty=1.0,
            min_caption_length=20,
            max_caption_length=100,
    ):
        self.args = args
        self.repetition_penalty = repetition_penalty
        self.min_length = min_caption_length
        self.max_length = max_caption_length
        self.device = device
        self.verbose = verbose
        self.api_key = api_key
        self.model_type = model_type
        if self.verbose:
            logger.info("[Initializing GPT-4 Vision model]")

    def encode_image_to_base64(self, img_path):
        """
        Encode the image to base64
        """
        with open(img_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate(self, img_path: str, num_retries: int = 3, retry_delay: int = 3):
        """ Generate caption for an image using GPT-4 Vision API """
        base64_image = self.encode_image_to_base64(img_path)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model_type,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "please help me provide a caption of this image in 10-30 words."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": self.max_length  # Adjusted to use self.max_length
        }

        # response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        # if response.status_code == 200:
        #     response_data = response.json()
        #
        #     if self.verbose:
        #         logger.info(response_data)
        #     # Extracting the generated caption from the response
        #     gencap = response_data['choices'][0]['message']['content'][1]['text']
        #     return gencap
        # else:
        #     if self.verbose:
        #         logger.error(f"Failed to generate caption: {response.text}")
        #     return ""

        attempt = 0
        while attempt < num_retries:
            try:
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload,
                                         timeout=30)
                response.raise_for_status()
                response_data = response.json()
                generated_caption = response_data['choices'][0]['message']['content']

                if self.verbose:
                    logger.info(f"=> Generated caption: {generated_caption}")

                return generated_caption

            except requests.exceptions.Timeout:
                attempt += 1
                logger.error("Request timed out. Retrying...")
                time.sleep(retry_delay)
            except requests.exceptions.HTTPError as err:
                logger.error(f"HTTP error occurred: {err}")
                return None
            except requests.exceptions.RequestException as err:
                logger.error(f"Error occurred during request: {err}")
                return None

        logger.error("Failed to generate caption after multiple attempts.")
        return None
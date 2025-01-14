# ELGCIG
Enhanced Language Guided Counterfactual Image Generation

# Env Install
conda env export  -n [env name] > ELGCIG.yaml or check the requirement.txt

# GPT API Key Input
Go to /lance/generate_captions.py and find class CaptionGenerator input your api key in api_key: str = "[api key]"

# Main File Running
accelerate launch --num_processes [number of gpu] --gpu_ids [id of gpu] main.py --dset_name ImageFolder --img_dir [original image dir] --lance_path [output counterfactual image dir] --text_similarity_threshold [text similarity threshold of BLIP (0.7 recommend)] --clip_img_thresh [CLIP image threshold (0.5 recommend)]

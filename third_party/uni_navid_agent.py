# coding: utf-8
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "Uni-NaVid"))

import numpy as np
import torch

from UniNavid.uninavid.mm_utils import (
    get_model_name_from_path,
    tokenizer_image_token,
    KeywordsStoppingCriteria,
)
from UniNavid.uninavid.model.builder import load_pretrained_model
from UniNavid.uninavid.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from UniNavid.uninavid.conversation import conv_templates, SeparatorStyle


seed = 30
torch.manual_seed(seed)
np.random.seed(seed)


class UniNaVid_Agent():
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(os.environ["UNINAVID_MODEL_PATH"], "uni-navid")

        print("Uni-NaVid initialization")

        self.conv_mode = "vicuna_v1"
        self.model_name = get_model_name_from_path(model_path)

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path,
            None,
            self.model_name,
            device_map="cuda",
        )

        assert self.image_processor is not None

        self.prompt_template = (
            "Imagine you are a robot programmed for navigation tasks. You have been given "
            "a video of historical observations and an image of the current observation <image>. "
            "Your assigned tasks is: '{}'. Analyze this sereis of images to determine your next "
            "four actions. The predicted action should be one of the following: forward, left, "
            "right or stop."
        )

        self.rgb_list = []
        self.count_id = 0
        self.reset()

        print("Initialization completed")

    def process_images(self, rgb_list):
        batch_image = np.asarray(rgb_list)
        self.model.get_model().new_frames = len(rgb_list)
        video = self.image_processor.preprocess(batch_image, return_tensors="pt")["pixel_values"].half().cuda()
        return [video]

    def predict_inference(self, prompt):
        question = prompt.replace(DEFAULT_IMAGE_TOKEN, "").replace("\n", "")
        qs = prompt

        VIDEO_START_SPECIAL_TOKEN = "<video_special>"
        VIDEO_END_SPECIAL_TOKEN = "</video_special>"
        IMAGE_START_TOKEN = "<image_special>"
        IMAGE_END_TOKEN = "</image_special>"
        NAVIGATION_SPECIAL_TOKEN = "[Navigation]"
        IAMGE_SEPARATOR = "<image_sep>"

        image_start_special_token = self.tokenizer(IMAGE_START_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_end_special_token = self.tokenizer(IMAGE_END_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_start_special_token = self.tokenizer(VIDEO_START_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_end_special_token = self.tokenizer(VIDEO_END_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        navigation_special_token = self.tokenizer(NAVIGATION_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_seperator = self.tokenizer(IAMGE_SEPARATOR, return_tensors="pt").input_ids[0][1:].cuda()

        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs.replace("<image>", "")
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs.replace("<image>", "")

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        token_prompt = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).cuda()

        indices_to_replace = torch.where(token_prompt == -200)[0]
        new_list = []

        while indices_to_replace.numel() > 0:
            idx = indices_to_replace[0]
            new_list.append(token_prompt[:idx])
            new_list.append(video_start_special_token)
            new_list.append(image_seperator)
            new_list.append(token_prompt[idx:idx + 1])
            new_list.append(video_end_special_token)
            new_list.append(image_start_special_token)
            new_list.append(image_end_special_token)
            new_list.append(navigation_special_token)
            token_prompt = token_prompt[idx + 1:]
            indices_to_replace = torch.where(token_prompt == -200)[0]

        if token_prompt.numel() > 0:
            new_list.append(token_prompt)

        input_ids = torch.cat(new_list, dim=0).unsqueeze(0)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        imgs = self.process_images(self.rgb_list)
        self.rgb_list = []

        cur_prompt = question

        with torch.inference_mode():
            self.model.update_prompt([[cur_prompt]])
            output_ids = self.model.generate(
                input_ids,
                images=imgs,
                do_sample=True,
                temperature=0.5,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")

        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        return outputs.strip()

    def reset(self, task_type="vln"):
        self.transformation_list = []
        self.rgb_list = []
        self.last_action = None
        self.count_id += 1
        self.count_stop = 0
        self.pending_action_list = []
        self.task_type = task_type

        self.first_forward = False
        self.executed_steps = 0
        self.model.config.run_type = "eval"
        self.model.get_model().initialize_online_inference_nav_feat_cache()
        self.model.get_model().new_frames = 0

    def act(self, data):
        rgb = data["observations"]
        self.rgb_list.append(rgb)

        navigation_qs = self.prompt_template.format(data["instruction"])
        navigation = self.predict_inference(navigation_qs)

        action_list = navigation.split(" ")
        if len(action_list) == 0:
            raise ValueError("No action found in the output")

        traj = [[0.0, 0.0, 0.0]]
        for action in action_list:
            if action == "stop":
                traj = [[0.0, 0.0, 0.0]] * 4
                break
            elif action == "forward":
                traj.append([x + y for x, y in zip(traj[-1], [0.5, 0.0, 0.0])])
            elif action == "left":
                traj.append([x + y for x, y in zip(traj[-1], [0.0, 0.0, -np.deg2rad(30)])])
            elif action == "right":
                traj.append([x + y for x, y in zip(traj[-1], [0.0, 0.0, np.deg2rad(30)])])

        self.executed_steps += 1
        self.latest_action = {"step": self.executed_steps, "path": [traj], "actions": action_list}
        return self.latest_action.copy()
import math
import pandas as pd
import fire
from utils.prompter import Prompter
import jsonlines
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import peft

instruction = """Refine the given code based on the provided code review comment.\n"""
input_ = "The comment is: ‘{}’\n The code is: ‘{}'"


def main(base_model: str = "", 
         lora_model: str = "",
         data_path: str = "Data/codeRefineTestDataset.jsonl",
         prompt_template_name: str = "llama",
         ):
    prompter = Prompter(prompt_template_name)

    if data_path.endswith(".jsonl"):
        with jsonlines.open(data_path, "r") as jsl:
            dataset = jsl.read()

        df = pd.DataFrame(dataset)

    elif data_path.endswith(".xlsx"):
        df = pd.read_excel(data_path)

    else:
        raise ValueError("Only jsonl or xlsx files are supported.")

    def get_tokenized_data(tokenizer, data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenizer(full_prompt)
        tokenized_full_prompt["labels"] = tokenized_full_prompt["input_ids"].copy()
        user_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point["input"]
        )
        tokenized_user_prompt = tokenizer(
            user_prompt
        )
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]

        return tokenized_full_prompt

    def request_model(tokenized_data_point):
        tokenized_data_point["input_ids"] = (torch.from_numpy(np.array(tokenized_data_point["input_ids"])).
                                             resize(1,len(tokenized_data_point["input_ids"])).cuda())
        tokenized_data_point["labels"] = (torch.from_numpy(np.array(tokenized_data_point["labels"])).
                                          resize(1,len(tokenized_data_point["labels"])).cuda())
        tokenized_data_point["attention_mask"] = (torch.from_numpy(
            np.array(tokenized_data_point["attention_mask"])).
                                                  resize(1,len(tokenized_data_point["attention_mask"])).cuda())
        return model(**tokenized_data_point)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if lora_model:
        model = peft.PeftModel.from_pretrained(model, lora_model)

    print(model)

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    scores = []

    with torch.no_grad():
        for index, data_point in tqdm(df.iterrows(), total=len(df)):
            try:
                review_datapoint = {
                    "instruction": instruction + input_.format(data_point["ReviewComment"], data_point["input"]),
                    "input": "",
                    "output": data_point["output"],
                }
                tokenized_condition_prompt = get_tokenized_data(tokenizer, review_datapoint)

                review_condition_out = request_model(tokenized_condition_prompt)["loss"].detach().cpu().numpy()

                no_review_datapoint = {
                    "instruction": instruction + input_.format("", data_point["input"]),
                    "input": "",
                    "output": data_point["output"],
                }
                tokenized_condition_prompt = get_tokenized_data(tokenizer, no_review_datapoint)

                origin_condition_out = request_model(tokenized_condition_prompt)["loss"].detach().cpu().numpy()

                scores.append(math.exp(review_condition_out) - math.exp(origin_condition_out))

            except torch.cuda.OutOfMemoryError:
                scores.append(99999999)

    df.insert(0, f"score_{base_model.split('/')[-1]}", scores)

    if data_path.endswith(".jsonl"):
        df.to_excel(f"Result/Condition_Review_score_Merge_{data_path.split('/')[-1].split('.')[0]}.xlsx",
                    index=False, engine='xlsxwriter')
    else:
        df.to_excel(data_path,
                    index=False, engine='xlsxwriter')

if __name__ == '__main__':
    fire.Fire(main)

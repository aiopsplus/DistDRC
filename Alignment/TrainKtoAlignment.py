import os
from typing import List
import fire
from utils.prompter import Prompter
from dataclasses import dataclass, field
import torch
import transformers
from utils.KtoTrainer import KtoPeftTrainer
from utils.KtoDataCollator import KtoDataCollatorWithPadding
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from lion_pytorch import Lion

IGNORE_INDEX = -100


def train(
        # model/data params
        base_model: str = "",  # the only required argument
        data_path: str = "yahma/alpaca-cleaned",
        output_dir: str = "./lora-alpaca",
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        cutoff_len: int = 256,
        val_set_size: int = 2000,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = [
            "q_proj",
            "v_proj",
        ],
        # Alignment
        average_log_prob: bool = True, 
        beta: float = 0.1,  
        add_eos_token: bool = False,
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training KTO with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"average_log_prob: {average_log_prob}\n"
            f"KTO_beta: {beta}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    @dataclass  
    class KTOTrainingArguments(transformers.TrainingArguments):
        average_log_prob: bool = field(default=True)
        beta: float = field(default=0.1)

    prompter = Prompter(prompt_template_name)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(model)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = (
        0
    )
    tokenizer.padding_side = "right"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False, 
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_directions(data_point):
        user_prompt = prompter.generate_prompt(
            data_point["instruction"]
        )
        tokenized_user_prompt = tokenize(
            user_prompt, add_eos_token=False
        )
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        full_prompt = prompter.generate_prompt(
            instruction=data_point["instruction"], label=data_point["output"]
        )

        tokenized_full_prompt = tokenize(full_prompt, add_eos_token)

        mask_label_ids = [IGNORE_INDEX] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]

        return {"input_ids": tokenized_full_prompt["input_ids"], "label_ids": mask_label_ids, "direction": float(data_point["direction"])}

    model.enable_input_require_grads()
    model.save_checkpoint = model.save_pretrained

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle(2024).map(generate_and_tokenize_directions)
        )
        val_data = (
            train_val["test"].shuffle(2024).map(generate_and_tokenize_directions)
        )
    else:
        train_data = data["train"].shuffle(2024).map(generate_and_tokenize_directions)
        val_data = None

    optimizer = Lion(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )

    len_dataset = len(train_data)
    total_steps = (len_dataset // batch_size) * num_epochs if len_dataset % batch_size == 0 \
        else (len_dataset // batch_size + 1) * num_epochs

    schedule = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=200,
        num_training_steps=total_steps
    )

    trainer = KtoPeftTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=KTOTrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=5,
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=100 if val_set_size > 0 else None,
            save_steps=100,
            output_dir=output_dir,
            save_total_limit=32,
            load_best_model_at_end=True if val_set_size > 0 else False,
            average_log_prob=average_log_prob,
            beta=beta,
        ),
        data_collator=KtoDataCollatorWithPadding(
            tokenizer, return_tensors="pt", padding=True
        ),
        optimizers=(optimizer, schedule),
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
import inspect
import os, sys, logging, re
import torch
from torch import nn
from transformers import PreTrainedTokenizerBase, Trainer, PreTrainedModel, Seq2SeqTrainer
from transformers.modeling_utils import unwrap_model
from typing import Any, Dict, List, Optional, Tuple, Union
from peft import get_peft_model, get_peft_model_state_dict
from peft import PeftModel
import torch.nn.functional as F
import json
from transformers.trainer import _is_peft_model

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

WEIGHTS_NAME = "adapter_model.bin"
TRAINING_ARGS_NAME = "training_args.json"


class KtoPeftTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metrics = dict()

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        if not isinstance(self.model, PreTrainedModel):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(unwrap_model(self.model), PreTrainedModel):
                unwrap_model(self.model).save_pretrained(output_dir, state_dict=state_dict)
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                torch.save(get_peft_model_state_dict(self.model, state_dict), os.path.join(output_dir, WEIGHTS_NAME))

            try:
                unwrap_model(self.model).peft_config.save_pretrained(output_dir)
            except AttributeError:
                unwrap_model(self.model).peft_config['default'].save_pretrained(output_dir)
        else:
            self.model.save_pretrained(output_dir, state_dict=state_dict)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        with open(os.path.join(output_dir, TRAINING_ARGS_NAME), "w") as f:
            json.dump(self.args.to_dict(), f)


    def get_log_probs(self, logits, labels):
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
        return log_probs_labels.squeeze(-1)


    def masked_mean(self, data, mask, dim=None, eps=1e-8):
        data = data * mask
        if dim is not None:
            return data.sum(dim=dim) / (mask.sum(dim=dim) + eps)
        else:
            return data.sum() / (mask.sum() + eps)


    def get_entropy(self, logits, mask):
        probs = torch.nn.functional.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = self.masked_mean(-torch.sum(probs * log_probs, dim=-1), mask)
        return entropy


    def get_model_output(self, model, inputs, is_ref_model=False):
        if is_ref_model:
            if isinstance(model, nn.parallel.DistributedDataParallel):
                with model.module.disable_adapter():
                    logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], return_dict=True).logits
            elif isinstance(model, PeftModel):
                with model.disable_adapter(): 
                    logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], return_dict=True).logits
            else:
                raise AttributeError(
                    f" model object [{model.__class__.__name__}] has no attribute [disable_adapter] "
                )
        else: 
            logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], return_dict=True).logits

    
        masks = inputs["labels"].ne(IGNORE_INDEX).long()

        log_probs = self.get_log_probs(logits[:, :-1, :], inputs["input_ids"][:, 1:])

        if self.args.average_log_prob:  
            logps = self.masked_mean(log_probs, masks[:, 1:], dim=-1)
        else:
            logps = (log_probs * masks[:, 1:]).sum(dim=-1)

        entropy = self.get_entropy(logits[:, :-1, :], masks[:, 1:])

        return entropy, logps

    def compute_loss(self, model, inputs, return_outputs=False):

        entropy, logps_mean = self.get_model_output(model, inputs)
        with torch.no_grad():
            ref_entropy, ref_logps_mean = self.get_model_output(model, inputs, is_ref_model=True)  # 计算原模型的输出

        kl_loss = (logps_mean - ref_logps_mean).mean().detach().clamp(min=0)

        """ https://arxiv.org/pdf/2402.01306
        If generation y ~ p_desirable, we have the 'desirable' loss:
            L(x, y) := 1 - sigmoid(beta * ([log p_policy(y|x) - log p_reference(y|x)] - KL(p_policy || p_reference)))
        If generation y ~ p_undesirable, we have the 'undesirable' loss:
            L(x, y) := 1 - sigmoid(beta * (KL(p_policy || p_reference) - [log p_policy(y|x) - log p_reference(y|x)]))
        """

        sigmoid_part = torch.nn.functional.sigmoid(self.args.beta * (logps_mean - ref_logps_mean - kl_loss))

        loss = 1.0 - (sigmoid_part * inputs["direction"]).mean()

        outputs = dict(
            target_entropy=entropy.detach(),
            kl_loss=kl_loss,
            loss=loss
        )

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
        inputs = self._prepare_inputs(inputs)

        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        loss = loss.detach()

        for k, v in outputs.items():
            self.metrics[k] = v.mean()

        logits = tuple(v for k, v in outputs.items() if k in ["accepts_reward", "rejects_reward"])
        if prediction_loss_only:
            return (loss, None, None)

        logits = torch.stack(logits, dim=1)
        labels = torch.zeros(logits.shape[0]).to(logits.device)
        return loss, logits, labels

    def log(self, logs):
        if len(self.metrics) > 0:
            for k, v in self.metrics.items():
                logs[f"eval_{k}"] = v.item()
            self.metrics.clear()

        return super().log(logs)


    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            model_to_inspect = self.model
            if _is_peft_model(self.model):
                if hasattr(self.model, "get_base_model"):
                    model_to_inspect = self.model.get_base_model()
                else:
                    # PeftMixedModel do not provide a `get_base_model` method
                    model_to_inspect = self.model.base_model.model
            signature = inspect.signature(model_to_inspect.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids", "direction"] + self.label_names))

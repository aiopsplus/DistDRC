import torch
import transformers
from transformers import DataCollatorWithPadding
from typing import Optional, Dict, Sequence, Union, List
from typing import Any, List, Union, Optional, Dict

IGNORE_INDEX = -100


class KtoDataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, instances: Sequence[Dict[str, Union[torch.Tensor, Sequence[int]]]]) -> Dict[str, torch.Tensor]:
        input_ids, labels, direction = [], [], []
        for instance in instances:
            input_id = instance["input_ids"]
            label = instance["label_ids"]
            direct = instance["direction"]

            input_ids.append(torch.LongTensor(input_id))
            labels.append(torch.LongTensor(label))
            direction.append(direct)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        direction = torch.Tensor(direction)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id).long(),
            direction=direction
        )

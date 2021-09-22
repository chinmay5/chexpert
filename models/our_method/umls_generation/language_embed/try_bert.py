# Trying the BERT model provided by pytorch itself
# Load pretrained model and tokenizer
import os

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    HfArgumentParser,
    set_seed,
    pipeline
)

class BERTEmbedModel:
    def __init__(self):
        model_name_or_path = 'dmis-lab/biobert-large-cased-v1.1'

        config = AutoConfig.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
            use_fast=False,
        )
        model = AutoModel.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
            cache_dir=None,
        )
        self.nlp = pipeline(task="feature-extraction", model=model, tokenizer=tokenizer)

    def get_embeds(self, text):
        output = self.nlp(text)
        wts = []
        for idx in range(len(output[0])):
            wts.append(torch.tensor(output[0][idx]))
        wts = torch.stack(wts).mean(axis=0)
        return wts

if __name__ == '__main__':
    model = BERTEmbedModel()
    print(model.get_embeds("What if a long text is sent").shape)


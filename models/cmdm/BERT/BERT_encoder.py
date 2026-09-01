import torch.nn as nn
import os

def load_bert(model_path, cond_len=None):
    bert = BERT(model_path, cond_len)
    bert.eval()
    bert.text_model.training = False
    for p in bert.parameters():
        p.requires_grad = False
    return bert

class BERT(nn.Module):
    def __init__(self, modelpath: str, cond_len):
        super().__init__()

        from transformers import AutoTokenizer, AutoModel
        from transformers import logging
        logging.set_verbosity_error()
        # Tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.cond_len = cond_len
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath)
        # Text model
        self.text_model = AutoModel.from_pretrained(modelpath)


    def forward(self, texts):
        if self.cond_len is not None:
            encoded_inputs = self.tokenizer(texts, return_tensors="pt", padding="max_length", max_length=self.cond_len, truncation=True)
            output = self.text_model(**encoded_inputs.to(self.text_model.device)).last_hidden_state
            mask = encoded_inputs.attention_mask.to(dtype=bool)
            # output = output * mask.unsqueeze(-1)
            return output, mask
        else:
            encoded_inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
            output = self.text_model(**encoded_inputs.to(self.text_model.device)).last_hidden_state
            mask = encoded_inputs.attention_mask.to(dtype=bool)
            return output, mask

import torch.nn as nn
from transformers import  BertModel


class FeedbackBERT(nn.Module):

    def __init__(self, num_op, dropout=0.3):
        super(FeedbackBERT, self).__init__()

        self.numC = num_op
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.op_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, self.numC)
        )

    def forward(self, ids, mask, token_type_ids):

        _, op = self.bert(ids,
                            attention_mask=mask,
                            token_type_ids=token_type_ids, return_dict=False)
        return self.op_layer(op)
        
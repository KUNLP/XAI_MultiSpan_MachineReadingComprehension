import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel
from transformers.modeling_bert import BertModel

class ATLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)

        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        # Sum two parts
        loss = loss1 + loss2
        loss = loss.mean()
        return loss

class MultiNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, num_label, dropout_rate):
        super(MultiNonLinearClassifier, self).__init__()
        self.num_label = num_label
        self.classifier1 = nn.Linear(hidden_size, hidden_size)
        self.classifier2 = nn.Linear(hidden_size, num_label)
        self.dropout = nn.Dropout(dropout_rate)

    def __call__(self, input_features):
        features_output1 = self.classifier1(input_features)
        features_output1 = F.gelu(features_output1)
        features_output1 = self.dropout(features_output1)
        features_output2 = self.classifier2(features_output1)
        return features_output2

# 문장 단위 표현을 위한 attention pooling 함수
class AttentivePooling(nn.Module):
    def __init__(self, hidden_size):
        super(AttentivePooling, self).__init__()
        self.hidden_size = hidden_size
        self.q_projection = nn.Linear(self.hidden_size, self.hidden_size)
        self.c_projection = nn.Linear(self.hidden_size, self.hidden_size)

    def __call__(self, query, context, context_mask):
        context_mask = context_mask.masked_fill(context_mask == 0, -100)
        num_sent = context_mask.size(-1)
        # query : [batch, hidden]
        # context : [batch, seq, hidden]
        # context_mask : [batch, seq, window]

        q = self.q_projection(query).unsqueeze(-1)
        c = self.c_projection(context)
        # q : [batch, hidden, 1]
        # c : [batch, seq, hidden]

        att = c.bmm(q)
        # att : [batch, seq, 1]

        expanded_att = att.expand(-1, -1, num_sent)
        # expanded_att : [batch, seq, window]
        masked_att = expanded_att + context_mask
        # masked_att : [batch, seq, window]

        att_alienment = F.softmax(masked_att, dim=1).transpose(1, 2)
        # att_alienment : [batch, window, seq]

        result = att_alienment.bmm(c)
        # result : [batch, window, hidden]
        return result

# Span Matrix를 위한 mask
def _make_triu_mask(seq_len):
    mask = torch.ones([seq_len, seq_len], dtype=torch.float)
    triu_mask = torch.triu(mask)
    triu_mask = triu_mask.masked_fill(triu_mask == 0, -100)
    triu_mask = triu_mask.masked_fill(triu_mask == 1, 0)

    return triu_mask


class RobertaForQuestionAnswering(BertModel):
    def __init__(self, config):
        super(ElectraForQuestionAnswering, self).__init__(config)
        # 분류 해야할 라벨 개수 (start/end)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size

        # ELECTRA 모델 선언
        self.roberta = RobertaaModel(config)

        # bi-gru layer 선언
        self.bi_gru = nn.GRU(input_size=config.hidden_size, hidden_size=int(config.hidden_size / 2),
                             num_layers=1, batch_first=True, dropout=0.2, bidirectional=True)

        # 단일 답변 학습을 위한 Layer
        self.qa_outputs = nn.Linear(int(config.hidden_size/2), config.num_labels)

        # 인코딩된 토큰 표현을 Start/End에 맞게 Projection하기 위한 Layer
        self.start_projection_layer = nn.Linear(config.hidden_size, int(config.hidden_size/4))
        self.end_projection_layer = nn.Linear(config.hidden_size, int(config.hidden_size/4))

        # Threshold Representation Layer
        self.th_outputs = MultiNonLinearClassifier(config.hidden_size, 1, 0.2) 

        # Adaotive Threshold Loss Function
        self.atloss = ATLoss()        

        # ELECTRA weight 초기화
        self.init_weights()

    def _span_matrix_with_valid_logits(self, encoded_vectors):
        # encoded_vectors : [batch, seq_len, hidden]

        start_vectors = self.start_projection_layer(encoded_vectors)
        end_vectors = self.end_projection_layer(encoded_vectors)
        # start_vectors : [batch, seq_len, hidden/4]
        # end_vectors : [batch, seq_len, hidden/4]

        batch_size = start_vectors.size(0)
        seq_len = start_vectors.size(1)

        span_matrix = start_vectors.bmm(end_vectors.transpose(1, 2))
        # span_matrix : [batch, seq_len, seq_len]

        triu_mask = _make_triu_mask(seq_len).unsqueeze(0).expand(batch_size, -1, -1).cuda()
        matrix_logits = span_matrix + triu_mask

        tok_outputs = torch.cat([start_vectors, end_vectors], -1)
        # tok_outputs : [batch, seq_len, hidden/2]

        return matrix_logits, tok_outputs

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            question_mask=None,
            sentence_mask=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions = None,
            end_positions = None,
            start_position=None,
            end_position=None,
            flag=None
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )
        # outputs : [1, batch, seq_len, hidden_size]

        sequence_output = outputs[0]
        # sequence_output : [batch_size, seq_len, hidden_size]

        cls_output = sequence_output[:, 0]
        # cls_output : [batch, hidden]

        th_logits = self.th_outputs(cls_output).squeeze(-1)
        # th_logits : [batch]

        self.bi_gru.flatten_parameters()

        tok_gru_output, _ = self.bi_gru(sequence_output)
        # tok_gru_output : [batch, seq_len, hidden_size]

        matrix_logits, tok_outputs = self._span_matrix_with_valid_logits(tok_gru_output)
        # matrix_logits : [batch, seq_len, seq_len]
        # tok_output : [batch, seq, hidden/2]

        tok_logits = self.qa_outputs(tok_outputs)
        # tok_logits : [batch, seq_len, 2]

        tok_start_logits, tok_end_logits = tok_logits.split(1, dim=-1)
        # tok_start_logits : [batch, seq_len, 1]
        # tok_end_logits : [batch, seq_len, 1]

        tok_start_logits = tok_start_logits.squeeze(-1)
        tok_end_logits = tok_end_logits.squeeze(-1)
        # tok_start_logits : [batch, seq_len]
        # tok_end_logits : [batch, seq_len]

        batch_size = tok_start_logits.size(0)
        tok_len = tok_end_logits.size(1)

        # 학습 
        if start_positions is not None and end_positions is not None:
            span_loss_fct = nn.CrossEntropyLoss()
            matrix_loss_fct = nn.CrossEntropyLoss(reduction='none')

            row_label = torch.zeros([batch_size, tok_len], dtype=torch.long).cuda()
            col_label = torch.zeros([batch_size, tok_len], dtype = torch.long).cuda()
            valid_label = torch.zeros([batch_size, tok_len, tok_len], dtype=torch.long).cuda()
            for batch_idx in range(batch_size):
                for answer_idx in range(len(start_positions[batch_idx])):
                    if start_positions[batch_idx][answer_idx] == 0:
                        break
                    row_label[batch_idx][start_positions[batch_idx][answer_idx]] = end_positions[batch_idx][answer_idx]
                    col_label[batch_idx][end_positions[batch_idx][answer_idx]] = start_positions[batch_idx][answer_idx]
                    valid_label[batch_idx][start_positions[batch_idx][answer_idx]][end_positions[batch_idx][answer_idx]] = 1
            if flag:
                # Threshold Loss
                final_matrix = matrix_logits.view(batch_size, -1)
                final_matrix[:, 0] = th_logits
                valid_loss = self.atloss(final_matrix, valid_label.view(batch_size, -1))
                return valid_loss
            else:
                # Matrix Loss
                row_mask = torch.sum(valid_label, 2).cuda()
                row_mask = row_mask.masked_fill(row_mask != 0, 1)
                # row_mask : 정답이 존재하는 row index (= 각 토큰에 대한 answer start point)
                col_mask = torch.sum(valid_label, 1).cuda()
                col_mask = col_mask.masked_fill(col_mask != 0, 1).cuda()
                # col_mask : 정답이 존재하는 col index (= 각 토큰에 대한 answer end point)

                row_loss = matrix_loss_fct(matrix_logits.view(-1, tok_len),
                    row_label.view(-1)).reshape(batch_size, tok_len)
                # span matrix의 모든 row에 대한 loss 계산
                col_loss = matrix_loss_fct(matrix_logits.transpose(1, 2).reshape(batch_size * tok_len, tok_len),
                col_label.view(-1)).reshape(batch_size, tok_len)
                # span matrix의 모든 col에 대한 loss 계산

                final_row_loss = torch.mean(torch.sum(row_loss * row_mask, 1))
                # 정답이 존재하지 않는 row masking
                final_col_loss = torch.mean(torch.sum(col_loss * col_mask, 1))
                # 정답이 존재하지 않는 col masking

                matrix_loss = (final_row_loss + final_col_loss) / 2

                # Single Span Loss
                start_span_loss = span_loss_fct(tok_start_logits, start_position)
                end_span_loss = span_loss_fct(tok_end_logits, end_position)

                span_loss = (start_span_loss + end_span_loss) / 2

                total_loss = (matrix_loss + span_loss) / 2

                return total_loss, matrix_loss, span_loss        
        return tok_start_logits, tok_end_logits, matrix_logits, th_logits

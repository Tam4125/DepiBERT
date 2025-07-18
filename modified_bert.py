from transformers import BertPreTrainedModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from transformers.models.bert.modeling_bert import BertSelfOutput, BertIntermediate, BertOutput, BertEmbeddings, BertPooler, BertPreTrainingHeads, BertForPreTrainingOutput, BertEncoder
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, SequenceClassifierOutput, BaseModelOutput
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.pytorch_utils import apply_chunking_to_forward



class MBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads 

        # Regularization
        self.attention_probs_dropout_prob = nn.Dropout(config.attention_probs_dropout_prob)
        self.hidden_dropout_prob = nn.Dropout(config.hidden_dropout_prob)
    
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)

    def transpose_for_scores(self, x):
        # Transpose x : [B, T, C] -> x: [B, nh, T, hs]

        B,T,C = x.size()

        return x.view(B, T, self.num_attention_heads, self.attention_head_size).transpose(1,2)

    def forward(self, 
                hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.FloatTensor] = None,
                output_attentions: Optional[bool] = False,
                ) -> tuple[torch.Tensor]:
        # hidden_states : [B, T, C]
        # attention_mask: [B, T]
        # dependency_matrix : [B, T, T]


        B,T,C = hidden_states.size()

        q,k,v = self.transpose_for_scores(self.query(hidden_states)), self.transpose_for_scores(self.key(hidden_states)), self.transpose_for_scores(self.value(hidden_states))  # [B, nh, T, hs]
        
        # Original self attention
        self_attn = q @ k.transpose(-2,-1) * 1/math.sqrt(k.size(-1))    # [B, nh, T, T]

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1 - attention_mask) * -10000.0
            self_attn = self_attn + attention_mask

        self_attn_probs = F.softmax(self_attn, dim=-1)    # [B, nh, T, T]
        self_attn_probs = self.attention_probs_dropout_prob(self_attn_probs)    # [B, nh, T, T]
        osa = torch.matmul(self_attn_probs, v)    # [B, nh, T, nh]
        osa = osa.transpose(1,2).contiguous().view(B, T, C) # [B,T,C]
        context_layer = osa

        outputs = (context_layer,)
        if output_attentions:
            outputs = outputs + (self_attn_probs,)

        return outputs
    


class MBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = MBertSelfAttention(config)
        self.output = BertSelfOutput(config)
    
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = False,
    ) -> tuple[torch.Tensor]:
        self_output = self.self(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        attention_output = self.output(self_output[0], hidden_states)

        outputs = (attention_output,) + self_output[1:]
        return outputs

    

class MBertlayer(GradientCheckpointingLayer):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_ff = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = MBertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, 
                hidden_states: torch.tensor, 
                attention_mask: Optional[torch.FloatTensor] = None,
                output_attentions: Optional[bool] = False,
                ) -> tuple[torch.tensor]:
        
        self_attention_outputs = self.attention(hidden_states=hidden_states,
                                                attention_mask=attention_mask,
                                                output_attentions=output_attentions,
                                                )
        
        attention_output = self_attention_outputs[0]

        # Splitting a large tensor along a dimension into smaller pieces, then processing each piece separately and combining the results.
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_ff, self.seq_len_dim, attention_output
        )


        return (layer_output,) + self_attention_outputs[1:]
    
    def feed_forward_chunk(self, attention_ouput):
        intermediate_output = self.intermediate(attention_ouput)
        layer_output = self.output(intermediate_output, attention_ouput)
        return layer_output

class MBertEncoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([MBertlayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            output_hidden_states: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
    ) -> BaseModelOutput:
        
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for layer_module in self.layer:
            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
        
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class MBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = MBertEncoder(config)
        self.pooler = BertPooler(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> BaseModelOutputWithPoolingAndCrossAttentions:

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        B, T = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :T]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(B, T)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros((B,T), dtype=torch.long, device=device)

        embedding_output = self.embeddings(input_ids=input_ids, 
                                           token_type_ids=token_type_ids,
                                           position_ids=position_ids,
                                           inputs_embeds=inputs_embeds,
                                           )
        
        if attention_mask is None:
            attention_mask = torch.ones((B,T), dtype=torch.long, device=device)
        
        fusion_outputs = self.fusion(
            hidden_states = embedding_output,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        
        encoder_outputs = self.encoder(
            hidden_states = fusion_outputs[0],
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    
custom_intro="""
    Dependency Bert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `next
    sentence prediction (classification)` head.
    """
class DependencyBertForPretraining(BertPreTrainedModel):
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.bert = MBertModel(config)
        self.cls = BertPreTrainingHeads(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
        self.cls.predictions.bias = new_embeddings.bias

    def get_input_embeddings(self):
        return self.bert.embeddings.word_embeddings
    
    def set_input_embeddings(self, new_embeddings):
        self.bert.embeddings.word_embeddings = new_embeddings 

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        next_sentence_label: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> BertForPreTrainingOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked),
            the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        next_sentence_label (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence
            pair (see `input_ids` docstring) Indices should be in `[0, 1]`:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.
        """

        outputs = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            attention_mask=attention_mask,
        )

        # sequence_output: [B,T,C]
        # pooled_output: [B,C]
        sequence_output, pooled_output = outputs.last_hidden_state, outputs.pooler_output

        # prediction_scores: [B,T,V]
        # seq_relationship_score: [B, 2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss

        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
        )


class DependencyBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.bert = MBertModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout ) if config.classifier_dropout is not None else nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        ):
        """
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. 
            If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), 
            If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # Compute Loss
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


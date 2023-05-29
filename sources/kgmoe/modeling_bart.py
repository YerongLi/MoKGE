"""PyTorch BART model, ported from the fairseq repo."""
import warnings
from typing import Optional, Tuple
import math
import torch
import random
import logging as logger
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.configuration_bart import BartConfig
# from transformers.models.bart.configuration_bart import BartConfig
from transformers.file_utils import (
    add_code_sample_docstrings,
    # add_start_docstrings_to_callable,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.tokenization_bart import BartTokenizer
from transformers.modeling_bart import (
    BART_INPUTS_DOCSTRING,
    shift_tokens_right,
    _prepare_bart_decoder_inputs,
    _make_linear_from_emb,
    _reorder_buffer,

)

from transformers.modeling_bart import (
    BartEncoder,
    BartDecoder,
    EncoderLayer,
    LayerNorm,
    invert_mask,
    PretrainedBartModel,
    SinusoidalPositionalEmbedding,
    LearnedPositionalEmbedding,
)

from sources.kgmoe.graph_layer import GraphEncoder

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "BartConfig"
_TOKENIZER_FOR_DOC = "BartTokenizer"

BART_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/bart-base",
    "facebook/bart-large",
]

class BartMoEModel(PretrainedBartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        # DEBUG

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        self.mixture_embeddings = nn.Embedding(config.mixtures, config.d_model, padding_idx=None) 
        self.encoder.mixture_embeddings = self.mixture_embeddings

        self.gnn = GraphEncoder(config.d_model, gamma=0.8, alpha=1, 
            beta=1, aggregate_method="max", tokenizer=None, hop_number=2)
        self.gnn.embed_word = self.shared
        self.gnn.mixture_embed = self.mixture_embeddings

        self.init_weights()

    # @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="facebook/bart-large",
        output_type=Seq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids,
        lm_mixture_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs: Optional[Tuple] = None,
        decoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ## KG Inputs
        concept_ids=None,
        kg_mixture_ids=None,
        concept_distances=None,
        concept_labels=None,
        head_ids=None,
        tail_ids=None,
        relation_ids=None,
        triple_labels=None,
        graph_outputs=None,
        **kwargs,
    ):
        if "decoder_past_key_values" in kwargs:
            warnings.warn("The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.", FutureWarning,)
            past_key_values = kwargs.pop("decoder_past_key_values")

        if decoder_input_ids is None:
            use_cache = False

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # make masks if user doesn't supply
        if not use_cache:
            decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_padding_mask=decoder_attention_mask,
                causal_mask_dtype=self.shared.weight.dtype,
            )
        else:
            decoder_padding_mask, causal_mask = None, None

        assert decoder_input_ids is not None

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                mixture_ids=lm_mixture_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            concept_outputs, _ = self.gnn(
                concept_ids=concept_ids, 
                distance=concept_distances, 
                head=head_ids, 
                tail=tail_ids, 
                relation=relation_ids, 
                triple_label=triple_labels,
                mixture_ids=kg_mixture_ids,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOuput when return_dict=False
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            decoder_input_ids,
            encoder_outputs[0],
            attention_mask,
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs, concept_outputs

        output_dict = Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

        return output_dict, None

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_output_embeddings(self):
        return _make_linear_from_emb(self.shared)  # make it on the fly


class BartEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    is a :class:`EncoderLayer`.

    Args:
        config: BartConfig
    """

    def __init__(self, config: BartConfig, embed_tokens):
        super().__init__()

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = config.max_position_embeddings

        self.embed_tokens = embed_tokens
        
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(
                config.max_position_embeddings, embed_dim, self.padding_idx
            )
        else:
            self.embed_positions = LearnedPositionalEmbedding(
                    config.max_position_embeddings,
                    embed_dim,
                    self.padding_idx,
                    config.extra_pos_embeddings,
            )


        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = LayerNorm(embed_dim) if config.normalize_embedding else nn.Identity()
        # mbart has one extra layer_norm
        self.layer_norm = LayerNorm(config.d_model) if config.add_final_layer_norm else None

    def forward(
        self, 
        input_ids, 
        mixture_ids=None,
        attention_mask=None, 
        output_attentions=False, 
        output_hidden_states=False, 
        return_dict=False,
    ):
        # check attention mask and invert
        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)

        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        
        embed_pos = self.embed_positions(input_ids)
        x = inputs_embeds + embed_pos

        if mixture_ids is not None:
            embed_mix = self.mixture_embeddings(mixture_ids)
            x = x + 1.0 * embed_mix

        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = [] if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states.append(x)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                attn = None
            else:
                x, attn = encoder_layer(x, attention_mask, output_attentions=output_attentions)

            if output_attentions:
                all_attentions = all_attentions + (attn,)

        if self.layer_norm:
            x = self.layer_norm(x)
        if output_hidden_states:
            encoder_states.append(x)
            # T x B x C -> B x T x C
            encoder_states = tuple(hidden_state.transpose(0, 1) for hidden_state in encoder_states)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if not return_dict:
            return tuple(v for v in [x, encoder_states, all_attentions] if v is not None)

        return BaseModelOutput(last_hidden_state=x, hidden_states=encoder_states, attentions=all_attentions)


class BartKGMoEForConditionalGeneration(PretrainedBartModel):
    base_model_prefix = "model"
    authorized_missing_keys = [r"final_logits_bias", r"encoder\.version", r"decoder\.version"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        base_model = BartMoEModel(config)
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

        self.model = base_model
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))

        self.kg_output_layer = nn.Sequential(nn.Linear(config.d_model, config.d_model),
            nn.Tanh(), nn.Linear(config.d_model, 1))

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        old_num_tokens = self.model.shared.num_embeddings
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self.model.shared = new_embeddings
        self._resize_final_logits_bias(new_num_tokens, old_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int, old_num_tokens: int) -> None:
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def forward(
        self,
        input_ids,
        lm_mixture_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        past_key_values=None,
        lm_labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        # New!
        concept_ids=None,
        kg_mixture_ids=None,
        concept_distances=None,
        concept_labels=None,
        head_ids=None,
        tail_ids=None,
        relation_ids=None,
        triple_labels=None,
        **unused,
    ):
        # logger.info('Entering forward')
        # correct
        if "lm_labels" in unused:
            warnings.warn("The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.", FutureWarning,)
            lm_labels = unused.pop("lm_labels")
        if "decoder_cached_states" in unused:
            warnings.warn("The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.", FutureWarning,)
            past_key_values = unused.pop("decoder_cached_states")
        if "decoder_past_key_values" in unused:
            warnings.warn("The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.", FutureWarning,)
            past_key_values = unused.pop("decoder_past_key_values")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if lm_labels is not None:
            use_cache = False
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(lm_labels, self.config.pad_token_id)
        # logger.info('input_ids concept_ids')
        # logger.info(self.tokenizer.decode(input_ids[0], skip_special_tokens=True))
        # logger.info(self.tokenizer.decode(concept_ids[0], skip_special_tokens=True))
        # 2023-05-28 17:43:46 INFO - modeling_bart.py - input_ids concept_ids
        # 2023-05-28 17:43:46 INFO - modeling_bart.py -  All cats chirp. bird container box apartment mouth mouse sun ranch backyard device car live computer floor land dust apply house sofa nature rug bird container box apartment mouth mouse sun ranch backyard device car live computer floor land dust apply house sofa nature rug
        # 2023-05-28 17:43:46 INFO - modeling_bart.py -  cat adult ground ally enemy animal apartment house apply sleep art reply home baby plate basement attic bath wet bed chair couch floor sofa bedroom food car book box meat cream town flat dog alley backyard bag barn cage cartoon charge closet desk dirt field fight game grass heat hose jungle kitchen lap laundry microwave nature park ranch roof rug shelf story sun table trouble vet wild zoo fur kitten mouse m pet scratch vomit control eat milk nap squirrel hair nose mammal noun gossip woman washed tail anchor chase creature eater eating household killer lion litter rat tiger jazz tripod human place room building horse rabbit sheep dust garden fox cow living farm plant area cover fish chicken nest tree camp carpet play type board bear snake furniture material water space surface bat pig store body street thing computer stuff paint action good country seat child chick work duck foot land stand term cell doll city unit rest wall square school item container soup children bar cellar level yard road family bird live lizard monkey forest life eye mess potato refrigerator garage wood inside office hold time device mouth wash man base earth hill dirty fungus mammoth oil trash puppy
        # 2023-05-28 17:43:47 INFO - modeling_bart.py - input_ids concept_ids
        # 2023-05-28 17:43:47 INFO - modeling_bart.py -  All cats chirp. bird container box apartment mouth mouse sun ranch backyard device car live computer floor land dust apply house sofa nature rug bird container box apartment mouth mouse sun ranch backyard device car live computer floor land dust apply house sofa nature rug
        # 2023-05-28 17:43:47 INFO - modeling_bart.py -  cat adult ground ally enemy animal apartment house apply sleep art reply home baby plate basement attic bath wet bed chair couch floor sofa bedroom food car book box meat cream town flat dog alley backyard bag barn cage cartoon charge closet desk dirt field fight game grass heat hose jungle kitchen lap laundry microwave nature park ranch roof rug shelf story sun table trouble vet wild zoo fur kitten mouse m pet scratch vomit control eat milk nap squirrel hair nose mammal noun gossip woman washed tail anchor chase creature eater eating household killer lion litter rat tiger jazz tripod human place room building horse rabbit sheep dust garden fox cow living farm plant area cover fish chicken nest tree camp carpet play type board bear snake furniture material water space surface bat pig store body street thing computer stuff paint action good country seat child chick work duck foot land stand term cell doll city unit rest wall square school item container soup children bar cellar level yard road family bird live lizard monkey forest life eye mess potato refrigerator garage wood inside office hold time device mouth wash man base earth hill dirty fungus mammoth oil trash puppy
        # 2023-05-28 17:43:48 INFO - modeling_bart.py - input_ids concept_ids
        # 2023-05-28 17:43:48 INFO - modeling_bart.py -  He took a nap on the tomato. opaque pile plate paper sponge cherry hill lie mineral water wood card fridge steak meat health book peach kitchen dish opaque pile plate paper sponge cherry hill lie mineral water wood card fridge steak meat health book peach kitchen dish
        # 2023-05-28 17:43:48 INFO - modeling_bart.py -  tomato nap red plant vegetable apple paste round health fruit software slip soup salad refrigerator juice sauce herb eaten cooking rest short sleep sleeping plate card cat trick fabric pile bid leather surface bet sponge texture snap food action water bed skin eating eat grape type animal house cake bean energy ginger orange record table book corn rice chicken hand paper time flat nest home play green potato dress rabbit ground meat seed grass lettuce grain form spice banana item pie shape white clean liquid egg bowl motion cheese flash smooth dish kitchen couch lie clothing apply ball cell cherry fall flesh heat wine steak strawberry wood mineral tree garden pot cotton olive peach pear pineapple hill set pumpkin carrot milk oil peel pizza brain fridge opaque circle good inside design cover solid paint jar mouth polish
        # 2023-05-28 17:43:48 INFO - modeling_bart.py - input_ids concept_ids
        # 2023-05-28 17:43:48 INFO - modeling_bart.py -  He took a nap on the tomato. opaque pile plate paper sponge cherry hill lie mineral water wood card fridge steak meat health book peach kitchen dish opaque pile plate paper sponge cherry hill lie mineral water wood card fridge steak meat health book peach kitchen dish
        # 2023-05-28 17:43:48 INFO - modeling_bart.py -  tomato nap red plant vegetable apple paste round health fruit software slip soup salad refrigerator juice sauce herb eaten cooking rest short sleep sleeping plate card cat trick fabric pile bid leather surface bet sponge texture snap food action water bed skin eating eat grape type animal house cake bean energy ginger orange record table book corn rice chicken hand paper time flat nest home play green potato dress rabbit ground meat seed grass lettuce grain form spice banana item pie shape white clean liquid egg bowl motion cheese flash smooth dish kitchen couch lie clothing apply ball cell cherry fall flesh heat wine steak strawberry wood mineral tree garden pot cotton olive peach pear pineapple hill set pumpkin carrot milk oil peel pizza brain fridge opaque circle good inside design cover solid paint jar mouth polish
        # 2023-05-28 17:43:48 INFO - modeling_bart.py - input_ids concept_ids
        # 2023-05-28 17:43:48 INFO - modeling_bart.py -  Music is a tasting activity,. camp flat paint hop heal metal toy product literature harmony pitch ballet music lead pleasure language talk learning wave point camp flat paint hop heal metal toy product literature harmony pitch ballet music lead pleasure language talk learning wave point
        # 2023-05-28 17:43:48 INFO - modeling_bart.py -  music activity rest art fun pleasure dance sing hobby listen use play sound singing profession entertainment recreation listening playing party speaker arrangement good band guitar cover drum bell time head heal kind harmony school flat gift term catch note country close club conductor silence respect metal group singer record performance relax finger fusion form heavy sick output ballet staff medicine rock stereo movie noise racket television cabinet elevator opera radio theatre skate dancing perform concert jazz language punishment song tune accord beat bounce breakdown composition culture disc garage har hop instrument jungle melody audio category expression leisure piano plays pop product rhythm musical work action game human writing voice paper house bar stage place tap type business ball talk cut board making reading card room body function audience company plate round break home set field literature car toy camp point musician ground snap tone ring book exercise measure class production result joy surprise theater collection hit crowd animal fish event wood pan motion sleep relaxation gathering vacation paint genre draw practice style learning bat drink enjoy laugh pain foot bird sign ear paste touch wave character pitch organ air city word theme box lead hammer heat material bed blow communication medium stuff
        # 2023-05-28 17:43:49 INFO - modeling_bart.py - input_ids concept_ids
        # 2023-05-28 17:43:49 INFO - modeling_bart.py -  Music is a tasting activity,. camp flat paint hop heal metal toy product literature harmony pitch ballet music lead pleasure language talk learning wave point camp flat paint hop heal metal toy product literature harmony pitch ballet music lead pleasure language talk learning wave point
        # 2023-05-28 17:43:49 INFO - modeling_bart.py -  music activity rest art fun pleasure dance sing hobby listen use play sound singing profession entertainment recreation listening playing party speaker arrangement good band guitar cover drum bell time head heal kind harmony school flat gift term catch note country close club conductor silence respect metal group singer record performance relax finger fusion form heavy sick output ballet staff medicine rock stereo movie noise racket television cabinet elevator opera radio theatre skate dancing perform concert jazz language punishment song tune accord beat bounce breakdown composition culture disc garage har hop instrument jungle melody audio category expression leisure piano plays pop product rhythm musical work action game human writing voice paper house bar stage place tap type business ball talk cut board making reading card room body function audience company plate round break home set field literature car toy camp point musician ground snap tone ring book exercise measure class production result joy surprise theater collection hit crowd animal fish event wood pan motion sleep relaxation gathering vacation paint genre draw practice style learning bat drink enjoy laugh pain foot bird sign ear paste touch wave character pitch organ air city word theme box lead hammer heat material bed blow communication medium stuff
        # 2023-05-28 17:43:49 INFO - modeling_bart.py - input_ids concept_ids
        # 2023-05-28 17:43:49 INFO - modeling_bart.py -  The air is super cool in egypt. place water vehicle poison surround form container magic projectile board transport resistance plain city smoke aircraft balloon mixture iron element land place water vehicle poison surround form container magic projectile board transport resistance plain city smoke aircraft balloon mixture iron element land
        # 2023-05-28 17:43:49 INFO - modeling_bart.py -  air base agent earth ground land vacuum water heat smoke space steam wind airport balloon bridge bus house jar pocket sky supermarket theatre aircraft airplane arm ash bird butterfly cloud dust helicopter helium humidity hydrogen jet magic moisture molecule moon nose odor owl oxygen plain plane pollen pollution projectile rain sound vapor pressure breathing living atmosphere element fluid gas mixture quality medium region band breath note vibration dry satellite breathe breeze clear component contains content light material matter mix resistance see solution source stuff substance term feeling limit open sense surround thought broadcast discuss airflow airline airliner army ball bleed blow liquid body oil human fly wood sun solid tree color point atom field place coal money sand mouth head salt floor metal making smell city board wave surface glass play change condition energy card paper snow hand transport dirt rock drop area state container form inside weather force thunder home chemical box power iron paint sea thing stone tin fog building pot property river waste machine table burn milk product room book vessel bread farm soap action star poison vehicle ocean live plant pipe roof grass type range cell environment touch closet food drink soup street wet blood cold branch
        # 2023-05-28 17:43:49 INFO - modeling_bart.py - input_ids concept_ids
        # 2023-05-28 17:43:50 INFO - modeling_bart.py -  The air is super cool in egypt. place water vehicle poison surround form container magic projectile board transport resistance plain city smoke aircraft balloon mixture iron element land place water vehicle poison surround form container magic projectile board transport resistance plain city smoke aircraft balloon mixture iron element land
        # 2023-05-28 17:43:50 INFO - modeling_bart.py -  air base agent earth ground land vacuum water heat smoke space steam wind airport balloon bridge bus house jar pocket sky supermarket theatre aircraft airplane arm ash bird butterfly cloud dust helicopter helium humidity hydrogen jet magic moisture molecule moon nose odor owl oxygen plain plane pollen pollution projectile rain sound vapor pressure breathing living atmosphere element fluid gas mixture quality medium region band breath note vibration dry satellite breathe breeze clear component contains content light material matter mix resistance see solution source stuff substance term feeling limit open sense surround thought broadcast discuss airflow airline airliner army ball bleed blow liquid body oil human fly wood sun solid tree color point atom field place coal money sand mouth head salt floor metal making smell city board wave surface glass play change condition energy card paper snow hand transport dirt rock drop area state container form inside weather force thunder home chemical box power iron paint sea thing stone tin fog building pot property river waste machine table burn milk product room book vessel bread farm soap action star poison vehicle ocean live plant pipe roof grass type range cell environment touch closet food drink soup street wet blood cold branch
        lm_outputs, kg_outputs = self.model(
            input_ids,
            lm_mixture_ids=lm_mixture_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # KG Inputs!
            concept_ids=concept_ids,
            kg_mixture_ids=kg_mixture_ids,
            concept_distances=concept_distances,
            concept_labels=concept_labels,
            head_ids=head_ids,
            tail_ids=tail_ids,
            relation_ids=relation_ids,
            triple_labels=triple_labels,      
        )
        lm_logits = F.linear(lm_outputs[0], self.model.shared.weight, bias=self.final_logits_bias)

        masked_lm_loss = None
        if lm_labels is not None: # None during training and not None during validation
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), lm_labels.view(-1))
        
        if not return_dict: # No during training and Yes during validation
            lm_output = (lm_logits, lm_outputs[-1]) + lm_outputs[1: -1] # training only this output
            kg_logits = self._calculate_kg_logits(kg_outputs)
            if masked_lm_loss is not None:
                return ((masked_lm_loss,) + lm_output), kg_logits
            else:
                return lm_output, kg_logits

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=lm_outputs.past_key_values,
            decoder_hidden_states=lm_outputs.decoder_hidden_states,
            decoder_attentions=lm_outputs.decoder_attentions,
            encoder_last_hidden_state=lm_outputs.encoder_last_hidden_state,
            encoder_hidden_states=lm_outputs.encoder_hidden_states,
            encoder_attentions=lm_outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(self, decoder_input_ids, past, attention_mask, use_cache, encoder_outputs, **kwargs):
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        if cur_len == 1 and self.config.force_bos_token_to_be_generated:
            self._force_token_ids_generation(logits, self.config.bos_token_id)
        elif cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_ids_generation(logits, self.config.eos_token_id)
        return logits

    def _force_token_ids_generation(self, scores, token_id) -> None:
        """force one of token_ids to be generated by setting prob of all other tokens to 0 (logprob=-float("inf"))"""
        scores[:, [x for x in range(self.config.vocab_size) if x != token_id]] = -float("inf")

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = []
        for layer_past in past:
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            layer_past_new = {
                attn_key: _reorder_buffer(attn_cache, beam_idx) for attn_key, attn_cache in layer_past.items()
            }
            reordered_past.append(layer_past_new)
        return reordered_past

    def get_encoder(self):
        return self.model.encoder, self.model.gnn

    def get_output_embeddings(self):
        return _make_linear_from_emb(self.model.shared)  # make it on the fly

    def _calculate_kg_logits(self, kg_hidden):
        return self.kg_output_layer(kg_hidden).squeeze(dim=2)
    
    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        max_length=None,
        min_length=None,
        do_sample=None,
        early_stopping=None,
        num_beams=None,
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        bad_words_ids=None,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_id=None,
        length_penalty=None,
        no_repeat_ngram_size=None,
        num_return_sequences=None,
        attention_mask=None,
        decoder_start_token_id=None,
        use_cache=None,
        expert_prompt=None,
        # KG Inputs!
        concept_ids: Optional[torch.LongTensor] = None,
        concept_labels: Optional[torch.LongTensor] = None,
        concept_distances: Optional[torch.LongTensor] = None,
        head_ids: Optional[torch.LongTensor] = None,
        tail_ids: Optional[torch.LongTensor] = None,
        relation_ids: Optional[torch.LongTensor] = None,
        triple_labels: Optional[torch.LongTensor] = None,
        **model_kwargs
    ) -> torch.LongTensor:

        # We cannot generate if the model does not have a LM head
        if self.get_output_embeddings() is None:
            raise AttributeError(
                "You tried to generate sequences with a model that does not have a LM Head."
                "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`, `XLMWithLMHeadModel`, `BartForConditionalGeneration` )"
            )

        max_length = max_length if max_length is not None else self.config.max_length
        min_length = min_length if min_length is not None else self.config.min_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        no_repeat_ngram_size = (no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size)
        bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
        num_return_sequences = (num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences)
        decoder_start_token_id = (decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id)

        if input_ids is not None:
            _batch_size = input_ids.shape[0]
            batch_size = input_ids.shape[0] * num_beams  # overriden by the input batch_size
        else:
            _batch_size = batch_size = 1

        assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
        assert isinstance(min_length, int) and min_length >= 0, "`min_length` should be a positive integer."
        assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
        assert isinstance(early_stopping, bool), "`early_stopping` should be a boolean."
        assert isinstance(use_cache, bool), "`use_cache` should be a boolean."
        assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictly positive integer."
        assert temperature > 0, "`temperature` should be strictly positive."
        assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
        assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
        assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
        assert input_ids is not None or (
            isinstance(bos_token_id, int) and bos_token_id >= 0
        ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
        assert pad_token_id is None or (
            isinstance(pad_token_id, int) and (pad_token_id >= 0)
        ), "`pad_token_id` should be a positive integer."
        assert (eos_token_id is None) or (
            isinstance(eos_token_id, int) and (eos_token_id >= 0)
        ), "`eos_token_id` should be a positive integer."
        assert length_penalty > 0, "`length_penalty` should be strictly positive."
        assert (
            isinstance(no_repeat_ngram_size, int) and no_repeat_ngram_size >= 0
        ), "`no_repeat_ngram_size` should be a positive integer."
        assert (
            isinstance(num_return_sequences, int) and num_return_sequences > 0
        ), "`num_return_sequences` should be a strictly positive integer."

        if input_ids is None:
            assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
                "you should either supply a context to complete as `input_ids` input "
                "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
            )
            input_ids = torch.full(
                (batch_size, 1),
                bos_token_id,
                dtype=torch.long,
                device=next(self.parameters()).device,
            )
        else:
            assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."

        # not allow to duplicate outputs when greedy decoding
        if do_sample is False:
            if num_beams == 1:
                # no_beam_search greedy generation conditions
                assert (num_return_sequences == 1
                ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

            else:
                # beam_search greedy generation conditions
                assert (num_beams >= num_return_sequences
                ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"

        # create attention mask if necessary
        # TODO (PVP): this should later be handled by the forward fn() in each model in the future see PR 3140
        if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids):
            attention_mask = input_ids.ne(pad_token_id).long()
        elif attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        # set pad_token_id to eos_token_id if not set. Important that this is done after
        # attention_mask is created
        if pad_token_id is None and eos_token_id is not None:
            logger.warning(
                "Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(eos_token_id)
            )
            pad_token_id = eos_token_id

        # vocab size
        if hasattr(self.config, "vocab_size"):
            vocab_size = self.config.vocab_size
        elif (
            self.config.is_encoder_decoder
            and hasattr(self.config, "decoder")
            and hasattr(self.config.decoder, "vocab_size")
        ):
            vocab_size = self.config.decoder.vocab_size
        else:
            raise ValueError("either self.config.vocab_size or self.config.decoder.vocab_size needs to be defined")

        # set effective batch size and effective batch multiplier according to do_sample
        if do_sample:
            effective_batch_size = batch_size * num_return_sequences
            effective_batch_mult = num_return_sequences
        else:
            effective_batch_size = batch_size
            effective_batch_mult = 1

        if self.config.is_encoder_decoder:
            if decoder_start_token_id is None:
                # see if BOS token can be used for decoder_start_token_id
                if bos_token_id is not None:
                    decoder_start_token_id = bos_token_id
                elif (
                    hasattr(self.config, "decoder")
                    and hasattr(self.config.decoder, "bos_token_id")
                    and self.config.decoder.bos_token_id is not None
                ):
                    decoder_start_token_id = self.config.decoder.bos_token_id
                else:
                    raise ValueError(
                        "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"
                    )

            assert hasattr(self, "get_encoder"), "{} should have a 'get_encoder' function defined".format(self)
            assert callable(self.get_encoder), "{} should be a method".format(self.get_encoder)

            # get encoder and store encoder outputs
            text_encoder, graph_encoder = self.get_encoder()

            concept_ids = self.repeat(concept_ids, num_beams)
            concept_labels = self.repeat(concept_labels, num_beams)
            concept_distances = self.repeat(concept_distances, num_beams)
            head_ids = self.repeat(head_ids, num_beams)
            tail_ids = self.repeat(tail_ids, num_beams)
            relation_ids = self.repeat(relation_ids, num_beams)
            triple_labels = self.repeat(triple_labels, num_beams)

            input_ids = self.repeat(input_ids, num_beams)
            attention_mask = self.repeat(attention_mask, num_beams)

            mixture_tmp = torch.arange(num_beams, dtype=torch.long, device=input_ids.device).view(
                    num_beams, 1).repeat(_batch_size, 1)
            kg_mixture_ids = mixture_tmp.expand(concept_ids.shape)
            concept_outputs, _ = graph_encoder(concept_ids, distance=concept_distances, 
                head=head_ids, tail=tail_ids, relation=relation_ids, 
                triple_label=triple_labels, mixture_ids=kg_mixture_ids)

            kg_logits = self._calculate_kg_logits(concept_outputs)
            
            kg_logits = kg_logits.masked_fill(concept_ids == self.config.pad_token_id, float('-inf'))
            kg_logits = F.softmax(kg_logits, dim=1)

            top_k, choose_k = 50, 20
            k_logits = torch.topk(kg_logits, top_k, dim=1)[0][:, top_k-1].reshape(batch_size, 1)
            kg_logits = kg_logits.masked_fill_(kg_logits - k_logits < 0, 0)

            top_index = torch.multinomial(kg_logits, choose_k, replacement=False)
            
            gather_index = torch.gather(concept_ids, dim=1, index=top_index)
            gather_mask = (gather_index != self.config.pad_token_id).float()
            input_ids = torch.cat([input_ids, gather_index], dim=1)
            attention_mask = torch.cat([attention_mask, gather_mask], dim=1)

            if self.config.mixture_embedding:
                lm_mixture_ids = mixture_tmp.expand(input_ids.shape)
                encoder_outputs = text_encoder(input_ids, mixture_ids=lm_mixture_ids, attention_mask=attention_mask, return_dict=True)
            else: # use prompt for different experts
                input_ids_prompt = expert_prompt.repeat(_batch_size, 1).to(input_ids.device)
                attention_prompt = torch.full(input_ids_prompt.shape, 1).to(attention_mask.device)
                
                input_ids = torch.cat([input_ids_prompt, input_ids], dim=1)
                attention_mask = torch.cat([attention_prompt, attention_mask], dim=1)
                encoder_outputs = text_encoder(input_ids, attention_mask=attention_mask, return_dict=True)
            
        # Expand input ids if num_beams > 1 or num_return_sequences > 1
        if num_return_sequences > 1 or num_beams > 1:
            input_ids_len = input_ids.shape[-1]
            input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_ids_len)
            attention_mask = attention_mask.unsqueeze(1).expand(
                batch_size, effective_batch_mult * num_beams, input_ids_len
            )

            input_ids = input_ids.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
            attention_mask = attention_mask.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
            
        if self.config.is_encoder_decoder:
            # create empty decoder input_ids
            input_ids = torch.full(
                (effective_batch_size * num_beams, 1), decoder_start_token_id, dtype=torch.long, device=next(self.parameters()).device,)
            cur_len = 1
            assert (
                batch_size == encoder_outputs.last_hidden_state.shape[0]
            ), f"expected encoder_outputs.last_hidden_state to have 1st dimension bs={batch_size}, got {encoder_outputs.last_hidden_state.shape[0]} "

            # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
            expanded_batch_idxs = (
                torch.arange(batch_size).view(-1, 1).repeat(1, num_beams * effective_batch_mult).view(-1).to(input_ids.device))

            # expand encoder_outputs
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_batch_idxs)

            # save encoder_outputs in `model_kwargs`
            model_kwargs["encoder_outputs"] = encoder_outputs

        else:
            cur_len = input_ids.shape[-1]

        assert (
            cur_len < max_length
        ), f"The context has {cur_len} number of tokens, but `max_length` is only {max_length}. Please make sure that `max_length` is bigger than the number of tokens, by setting either `generate(max_length=...,...)` or `config.max_length = ...`"

        if num_beams > 1:
            output = self._generate_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                early_stopping=early_stopping,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                num_return_sequences=num_return_sequences,
                length_penalty=length_penalty,
                num_beams=num_beams,
                vocab_size=vocab_size,
                attention_mask=attention_mask,
                use_cache=use_cache,
                model_kwargs=model_kwargs,
            )
        else:
            output = self._generate_no_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                attention_mask=attention_mask,
                use_cache=use_cache,
                graph_outputs=concept_outputs,
                model_kwargs=model_kwargs,
            )

        return output

    @staticmethod
    def repeat(tensor, K):
        # [B, ...] => [B*K, ...]
        if isinstance(tensor, torch.Tensor):
            B, *size = tensor.size()
            # repeat_size = [1] + [K] + [1] * (tensor.dim() - 1)
            # tensor = tensor.unsqueeze(1).repeat(*repeat_size).view(B * K, *size)
            expand_size = B, K, *size
            tensor = tensor.unsqueeze(1).expand(*expand_size).contiguous().view(B * K, *size)
            return tensor
        elif isinstance(tensor, list):
            out = []
            for x in tensor:
                for _ in range(K):
                    out.append(x.copy())
            return out
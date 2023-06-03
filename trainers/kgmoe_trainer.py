import warnings
import logging
import numpy as np
from sklearn.metrics import f1_score, precision_score

from typing import Any, Dict, Optional, Tuple, Union
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from trainers.seq2seq_trainer import Seq2SeqTrainer

from transformers.trainer_utils import (
    EvalPrediction,
    PredictionOutput,
    nested_concat,
    nested_numpify,
)

from transformers.integrations import (
    is_optuna_available,
    is_tensorboard_available,
)

_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

if is_tensorboard_available():
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        from tensorboardX import SummaryWriter

if is_optuna_available():
    import optuna

from evals.eval_acc_div import eval_accuracy_diversity
logger = logging.getLogger(__name__)


class KGMoESeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mixtures = self.data_args.mixtures
        self.expert_prompt = self.data_args.expert_prompt
        self.mixture_embedding = self.data_args.mixture_embedding
        self.pows = self.data_args.pows
        self.loss_ratio = self.data_args.loss_ratio
        # self.compute_metrics=compute_metrics

    def _training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], optimizer) -> torch.Tensor:
        # logging.info("inputs")
        # logging.info(inputs.keys())
        # 2023-05-28 02:55:32 INFO     dict_keys(['input_ids', 'attention_mask', 'decoder_input_ids', 'labels', 'concept_ids', 'concept_distances', 'concept_labels', 'oracle_concept_ids', 'head_ids', 'tail_ids', 'relation_ids', 'triple_labels'])
        
        # logging.info(self.tokenizer)
        # <transformers.tokenization_bart.BartTokenizer object at 0x2acbab8e8490>
        # logging.info(self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))
        # logging.info(self.tokenizer.decode(inputs['labels'][0],skip_special_tokens=True))
        # 2023-05-28 03:34:22 INFO - kgmoe_trainer.py - inputs
        # 2023-05-28 03:34:22 INFO - kgmoe_trainer.py -  All cats chirp. bird container box apartment mouth mouse sun ranch backyard device car live computer floor land dust apply house sofa nature rug bird container box apartment mouth mouse sun ranch backyard device car live computer floor land dust apply house sofa nature rug
        # 2023-05-28 03:34:22 INFO - kgmoe_trainer.py -  only birds chirp.
        # 2023-05-28 03:34:23 INFO - kgmoe_trainer.py - inputs
        # 2023-05-28 03:34:23 INFO - kgmoe_trainer.py -  He took a nap on the tomato. opaque pile plate paper sponge cherry hill lie mineral water wood card fridge steak meat health book peach kitchen dish opaque pile plate paper sponge cherry hill lie mineral water wood card fridge steak meat health book peach kitchen dish
        # 2023-05-28 03:34:23 INFO - kgmoe_trainer.py -  tomato is very small, no one can be fit to take nap on it.
        # 2023-05-28 03:34:24 INFO - kgmoe_trainer.py - inputs
        # 2023-05-28 03:34:24 INFO - kgmoe_trainer.py -  Music is a tasting activity,. camp flat paint hop heal metal toy product literature harmony pitch ballet music lead pleasure language talk learning wave point camp flat paint hop heal metal toy product literature harmony pitch ballet music lead pleasure language talk learning wave point

        # try:
        #     logging.info(self.tokenizer.decode(inputs['concept_labels'][0],skip_special_tokens=True))
        # except:
        #     logging.info(inputs['concept_labels'][0])
        # 2023-05-28 03:42:14 INFO - kgmoe_trainer.py - tensor([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        #          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        #          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        #          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        #          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        #          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        #          0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        #          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        #          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        #          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        #          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        #          0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        #         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        #         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        #         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        #         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        #         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        self.B, self.L = inputs['labels'].shape
        self.BC, self.LC = inputs['concept_labels'].shape
        assert self.B == self.BC
        self.pad_mask = (inputs['labels'] == self.config.pad_token_id).view(self.B, 1, self.L).to(self.args.device)
        self.concept_pad_mask = (inputs['concept_labels'] == self.config.pad_token_id).view(self.BC, 1, self.LC).to(self.args.device)

        inputs = self._prepare_inputs(inputs)

        mixture_tmp = torch.arange(self.mixtures, dtype=torch.long, device=inputs['input_ids'].device).view(self.mixtures, 1)
        kg_mixture_ids = mixture_tmp.repeat(inputs['concept_ids'].shape)

        if self.mixture_embedding:
            
            lm_mixture_ids = mixture_tmp.repeat(inputs['input_ids'].shape)
            mixture_inputs = {k: self.repeat(v, self.mixtures) for k, v in inputs.items()}
            mixture_inputs['lm_mixture_ids'] = lm_mixture_ids
            mixture_inputs['kg_mixture_ids'] = kg_mixture_ids
            model.eval()

            mixture_ids = self.compute_mixture_ids(model, mixture_inputs)
            inputs['lm_mixture_ids'] = mixture_ids.expand(inputs['input_ids'].shape)
            inputs['kg_mixture_ids'] = mixture_ids.expand(inputs['concept_ids'].shape)

        else: # using prompt as different expert

            mixture_ids_prompt = self.expert_prompt.repeat(self.B, 1).to(self.args.device)
            mixture_att_prompt = torch.full(mixture_ids_prompt.shape, 1).to(self.args.device)

            mixture_inputs = {k: self.repeat(v, self.mixtures) for k, v in inputs.items()}
            mixture_inputs['kg_mixture_ids'] = kg_mixture_ids
            mixture_inputs['input_ids'] = torch.cat([mixture_ids_prompt, mixture_inputs['input_ids']], dim=1)
            mixture_inputs['attention_mask'] = torch.cat([mixture_att_prompt, mixture_inputs['attention_mask']], dim=1)
            
            model.eval()
            mixture_ids = self.compute_mixture_ids(model, mixture_inputs)
            expanded_mixture_ids = mixture_ids.expand(self.B, self.data_args.prompt_nums).unsqueeze(dim=1)
            input_ids_prompt = torch.gather(mixture_ids_prompt.view(
                self.B, self.mixtures, -1), dim=1, index=expanded_mixture_ids).squeeze()
            attention_prompt = torch.full(input_ids_prompt.shape, 1).to(self.args.device)
            inputs['kg_mixture_ids'] = mixture_ids.expand(inputs['concept_ids'].shape)
            inputs['input_ids'] = torch.cat([input_ids_prompt, inputs['input_ids']], dim=1)
            inputs['attention_mask'] = torch.cat([attention_prompt, inputs['attention_mask']], dim=1)
            
        # do the expert training!
        model.train()
        # logging.info('model')
        # logging.info(model)
        lm_loss, kg_loss = self.compute_loss(model, inputs)
        loss = lm_loss + self.loss_ratio * kg_loss

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16 and _use_native_amp:
            self.scaler.scale(loss).backward()
        elif self.args.fp16 and _use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss.detach()

    def compute_loss(self, model, inputs):
        
        lm_labels = inputs.pop("labels")
        kg_labels = inputs.pop("concept_labels")
        lm_outputs, kg_logits = model(**inputs, use_cache=False)
        lm_logits = lm_outputs[0]
        lm_loss = self._compute_loss(lm_logits, lm_labels)
        kg_loss = self._compute_kg_loss(kg_logits, kg_labels)
        return lm_loss, kg_loss

    def compute_mixture_ids(self, model, inputs):
        
        _inputs = inputs.copy()
        _lm_labels = _inputs.pop("labels")
        _kg_labels = _inputs.pop("concept_labels")
        lm_outputs, kg_logits = model(**_inputs, use_cache=False)
        lm_logits = lm_outputs[0]
        mixture_ids = self._compute_mixture_loss(lm_logits, kg_logits, _lm_labels, _kg_labels)
        return mixture_ids

    def _compute_mixture_loss(self, lm_logits, kg_logits, lm_labels, kg_labels):

        assert lm_logits.shape[:2] == lm_labels.shape
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id, reduction='none')
        
        lm_loss = loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), lm_labels.view(-1)).reshape(self.B, self.mixtures, self.L)
        lm_loss = lm_loss.masked_fill(self.pad_mask, 0).sum(dim=2)
        
        kg_loss = self._compute_kg_loss(kg_logits, kg_labels, reduction='none').view(self.BC, self.mixtures, self.LC)
        kg_loss = kg_loss.masked_fill(self.concept_pad_mask, 0).sum(dim=2)

        mixture_ids = lm_loss.argmin(dim=1).unsqueeze(dim=1).type(torch.int64)

        return mixture_ids

    def _compute_kg_loss(self, node_logits, node_labels, reduction='mean'):
        
        loss_weights = (node_labels + 1).pow(self.pows)

        if node_logits.shape != node_labels.shape:
            node_logits = node_logits[:, : node_labels.shape[-1]]
        
        node_loss = F.binary_cross_entropy_with_logits(
            node_logits.float(), node_labels.float(), 
            weight=loss_weights, reduction='none')
        
        valid_mask = ~(node_labels == -1)
        labels_len = valid_mask.float().sum(dim=1)

        if reduction == 'mean':
            _node_loss = node_loss.sum(dim=1) / labels_len
            _node_loss = _node_loss.mean()
            return _node_loss
        
        return node_loss

    # def compute_metrics(self, eval_prediction):
    #     # Extract the predictions and labels from the EvalPrediction object
    #     logging.info('Entering compute_metrics')
    #     predictions = eval_prediction.predictions
    #     label_ids = eval_prediction.label_ids

    #     # Perform your desired computations on the predictions and labels
    #     # Here's an example of computing accuracy
    #     predictions = torch.argmax(predictions, dim=1)
    #     correct = (predictions == label_ids).sum().item()
    #     accuracy = correct / len(label_ids)

    #     # Return a dictionary containing the computed metrics
    #     metrics = {
    #         'accuracy': accuracy
    #     }
    #     return metrics

    def prediction_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)


        with torch.no_grad():
            # logger.info('self.args.predict_with_generate and not self.args.prediction_loss_only')
            # logger.info(self.args.predict_with_generate and not self.args.prediction_loss_only)
            # True
            if self.args.predict_with_generate and not self.args.prediction_loss_only:
                num_return_sequences = self.data_args.eval_beams if self.data_args.do_sample else None
                expert_prompt = self.data_args.expert_prompt if hasattr(self.data_args, 'expert_prompt') else None
                logger.info("iput_ids")
                # logger.info(inputs["input_ids"].shape)   X torch.Size([70, 43])
                # logger.info(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
                # logger.info(self.tokenizer.decode(inputs["concept_ids"][0], skip_special_tokens=True))
                # logger.info(self.tokenizer.decode(inputs["relation_ids"][0], skip_special_tokens=True))

                # 2023-06-03 01:52:25 INFO - kgmoe_trainer.py - iput_ids
                # 2023-06-03 01:52:25 INFO - kgmoe_trainer.py -  he put elephant into the jug.
                # 2023-06-03 01:52:25 INFO - kgmoe_trainer.py -  jug elephant water handle bottle breast edge serve pour ear mouth woman jail glass stew vessel container pig animal bull trunk cow summer walk doll nose mammal heavy war memory wax circus zoo mammoth grey jaw ton trial trumpet human body food metal liquid hair face eye horse milk drink case pot device cup plate head jar seal tap bowl instrument round organ bone female ground fish bridge pipe mineral inside stuff car middle place shape steer balloon cell dish use feature cat dog monkey bear play air dolphin bucket street whale blood grass fruit brim camel coat drop issue lead paste shark steel stone drinking making substance white wine ball cover stock field open point smooth pitcher finger house store polish skin chicken chest heart bank living coin table light action ability sense squirrel children adult man male
                # 2023-06-03 01:52:25 INFO - kgmoe_trainer.py - 's on the on are on are on on on on are on on the are the are are are are are are as the are are are are are on on are are are on on are on are and's on's are's's on's are " are are are are� are are's are as and " are are are are it are are are and are " are are are are on are are are on on are and on on's� on are are and the's's are as on are on are on on are and are are as are are are are " are are on are are are are as are are� are are are on are are are on are are are are are the are are are on are on are are are are on on are are are's on are are are are are are are are are are are it are on on on on on is it it it on on on on on on on on on on on on on on on on's on on on the on are are are are on on are in on on on on on on are to are are are are it are are are are be " are are on are on on and on� on on on on on on on are on on on on on on on on on on on on on on on on on are are are on are are on are are are with on are are are on on are are are on are on on on on are on's on on on are on on are are are are are's are are are on and are are are on on on on are are are's are on on are and as are on "'s are are on on are are are are are on on on are are on are on on are are on on are are on the are " are on on on on on on on on on on on on on on on are's on on on on and and are are on are are are are on on on are the on on's on on are are's's on are are are are on with on's the are the the the on are " on on are are and are's are on are and are " on on are are and are's are are are are are are are are on to is on
                # 2023-06-03 01:52:28 INFO - kgmoe_trainer.py - iput_ids
                # 2023-06-03 01:52:28 INFO - kgmoe_trainer.py -  I ate grilled and peppered red Stone for dinner yesterday.
                # 2023-06-03 01:52:28 INFO - kgmoe_trainer.py -  stone yesterday eat grill pepper dinner table throw fruit restaurant ground seed delivery wood lunch plate food fish bread meat breakfast chicken corn cook d cooking dining nutrition meal eating soup spoon attack devil oven steak barbecue salt earth water mineral house fall set column stick band bar bed construct piece brick bullet burn pleasure paper castle opaque solid ladder coal diamond color construction wall metal smooth light deposit raise round dirt draw sack mass matter making weed ruin weight heavy glass grave point chill pit material kill liberty sand cave driveway nature pond river shoe stream sword rock ruby pound building quarter thrown broom cement plant place type stuff surface oil substance floor board dust cut element heat flat pot tree land grain steel home structure lead nest animal hand unit square bowl dish kitchen wax action ball drop body head paint build term rice apple container concrete tin silver form white bone gold cover room work polish space area butter preserve roll box cake coin snow brown mud item door good break tool bridge change smoke measure iron cat dog eaten spot use feed air cast drink garden grass ice money lot egg cotton product supper machine gas street book magazine carpet fork game
                # 2023-06-03 01:52:28 INFO - kgmoe_trainer.py -  on on are on on on are on are are on are are on on are are are on� are be are are a on are are are on on are are on� on are on are on on are is to are on on are on on on are on have the are on are� are are are on are on on are's " as and and that are are on on the on on on on on the on it are at on have are and are are's are are are are are are are are are are have's to on on on are it are are are are are at's's are at at's's at at are are are's� are are on are are to it is it are it is are are on on have are have are are are on that on that are on are on are are are are are on are on are are are's on on on have's are are on have are are are have's on on on are on to on are on are are are are the are on are on's on are are it have on on are on the are are are on are are are on are are are's. are on on are on on on on are on on are to are are's are on on are are are are are are are are's's the's are's the's on on on on on are on on on on's on's are's's are's's's on's on's on " on on's are are are are's� on are's are with are are are the on as as on on on on on on are " on on on are and the are on on on on on are are are on on are's are on are to are are are are are are are it� are are are to on's are are on are are are are are are are on the are on are's are are are are are are are� are's on are are� are on are are are are on are on�'s on's are's on on are on's on are are on on on on it on are are on are on on are are on on on on on on have on are are are are are are are are it are on are are on on on are are are are are are are on on on are on on on on on that are " on the are on are are� are are are are are are are are on are are are on are on on the on on a on the on on are are are are� it are on are's are on on are are on are are on on are on are on on a on on on are are are on are are are are are are are are on on to on to are to have on's
                # 2023-06-03 01:52:31 INFO - kgmoe_trainer.py - iput_ids
                # 2023-06-03 01:52:31 INFO - kgmoe_trainer.py -  He felt pain in teeth, so he went to police station.
                # 2023-06-03 01:52:31 INFO - kgmoe_trainer.py -  pain tooth feel police station human anger sense band work burn panic heat fear ill experience feeling neck hunger dentist suffer sensation sting emotion nerve physical handle filling taste department child adjective agony animal annoy attack comfort good breathing body muscle pleasure response hurt cause rain distress disease cut life love doctor effort period upset joy suffering wound illness sick trouble lying humiliation sentient tolerance grief misery sorrow punch kill crying learning war major medicine nuisance relief ass injury listening punching remembering running singing typing unpleasant ac anguish discomfort symptom stitch torment alleviate blister complain headache labour moan aspirin bakery internet killer result visit penalty action touch shock care die health sound condition energy type anxiety play activity time exercise event punishment school blood dance living happiness fun term dead cold sadness state skin head brain smell red calm harm operation effect bite food stomach hospital fit laugh poison heart home mouth evil flesh hand art fever man fight making break game job rack issue force damage death smile money nature baby dog eye face finger memory sex annoyance cross hate air cool tool mechanism music relaxation satisfaction passion practice product place consequence scar danger fine sickness enjoy reading movement drug ease heal killing peace
                # 2023-06-03 01:52:31 INFO - kgmoe_trainer.py -  with on's on the are are it by are on are are the the The with are are it on are are are on on The are are are on are and The on on are are on are are are to are are are the on are The are are are are are are are on it it on The and on on are are are The are it The The The The The a are are the the on the " are are it are are are are are are on on on on on on on on on on on on on are are are are are are are are are are on " are are are on are are are are are and's are are are are are it are are are on are is The is are The on " on on on it to it for on are on on are are are are on are are are " are are on on on the on are are on are on are " are are are are on it are are are are are are are on are are on are are are are are are are are are are are are the on are are on are are the it on on on on are are are are on on are on are's are are are on on are are " on on the are, on The on The a on are are are on on are are and are on and on are are are are are are are at are are on are are are are are are are are are are on are� are on are are are are are are are are are are are on as are on on are are are are are on are on. are are on on on on it are it on on on on on on on are are are are are are " are� have are are are on are are are on are are are on on are on� are are " are are The are are are on on on are are on are are are are are are on on on are on are are are are are " are are on are are are it on are the on on on on on on have are on are are are are are on on are it on the on on are on on on " are on is� are as are are are on are� are " are on " on are's are " on are The are " are " it are are� are it it on are on " on are the are the on on on are on on on on are on on on on on on on are on on on are are are on on are on are on on are on the on on on are on " are are are on are are on are on have on is on are are are are are are are are The " are are on are on's are are are have are� are are are are to are are on on
                # 2023-06-03 01:52:33 INFO - kgmoe_trainer.py - iput_ids
                # 2023-06-03 01:52:33 INFO - kgmoe_trainer.py -  The mother fries stones for her children in the morning.
                # 2023-06-03 01:52:33 INFO - kgmoe_trainer.py -  stone child mother fry morning nature offspring chicken children earth ground house burn light wood relative birth adult parent father family baby animal daughter girl orphan son chick human member mom female egg water mineral fall set column stick band bar bed construct piece brick bullet pleasure paper castle table opaque solid ladder coal diamond color construction seed wall metal smooth deposit raise round dirt draw throw sack mass matter delivery making weed ruin fruit weight heavy glass grave point chill pit material kill liberty sand cave driveway pond river shoe stream sword rock ruby pound building quarter thrown broom cement counter flag gravel industry type plant place surface nest unit tree head ball boy home stuff element man land substance floor dog dust drop body structure wax board eye plate food flat concrete term build square cut steel horse lead dad space oil area paint work hand brother white game pot snow mud iron cotton grain change cover form gold issue air bird brown eat farm layer road group woman roof grass ice money lot slip apple street door bank cell action bridge waste polish bear bone dress tool magazine cat city country forest state garden quality kid produce finger sex doll paste play
                # 2023-06-03 01:52:33 INFO - kgmoe_trainer.py -  on are on are on on on are are on are are on are are on on are are are are on� are be a on are are are on on are on� on are on on are on on are is are on on are on on are on are on are� are are are on are on on are on are's " as and and that are are are are are are on on on on on on are. on on are and are are it are are the on are are are are are the on the on on are are on " are are are are are on are on are are's are are are are are are are are on are are on on are on are are are on on on are on on on on are on on on on are are are are are on are are are are are are are are on are on " are are are are are are are are are are on on are's are on are are are are on are are on on on are on on are are are are on are on are on are are on are are are are are on are are are are are on on the's's's are are on are are on on� are are's are on on on on on are are are are on the on on are are the on on on on are are are are on on on on on are are are are on on are are are are's is on is on are are on on on on on on on on on, are on are are are are on on are are are are on on are are on on are are are are on are on on on on on's on on The. are on are on the on are have's are are on� on are on on " on on as the as on the the on are the on are on are on are are are are are are are are are are� are are on's are are are on's are on are's are with are are�'s on's are's, on are on are are are are it on on on on on on on are on are are are are are are are are are are are's are are are� and are's are are are are are are are� are the are on are the are are are are are are are on are on are on are on on are on on are are are are are are with are are are on on on on on are are are are are on are are are are on that are are are are are are on� are on on are are are are are are are are on are on are are are are are are's are are are are are are are are the on to are to on are are on the the are on are are are
                # 2023-06-03 01:52:36 INFO - kgmoe_trainer.py - iput_ids
                # 2023-06-03 01:52:36 INFO - kgmoe_trainer.py -  The man thought the task was too spicy.
                # 2023-06-03 01:52:36 INFO - kgmoe_trainer.py -  man task work piece establishment presence child adult affair human youth agent alien animal bird baby dress farmer lady boy female girl husband woman women nature politician prisoner servant machine bachelor black boyfriend bull father fellow guy male lover policeman stud tile white staff art artificial bear blood board body break bridge brother butcher canal captain chest clown club cookie count dad demand doctor dog doll engineer eye fisherman flesh gender ginger goat god good grandfather guard hand health hero hunt jack jacket judge king lawyer leader lumber area beard bicycle cave fish form grown house living network penis sex son species head member type food home chick play mother family daughter children bar creature game job horse queen school life occupation master chicken hair plant farm bed card figure love cow sheep cut fox class activity party business stuff term bone space face egg cat relation room thing group cell nest bank boss paint worker building table place profession material shape statue kid wife charge city kind mammal wood tree sister pass water action position book office money minister tool company computer structure force plate bread crown slip toy state market parent age pig country milk sound officer fly hen friend
                # 2023-06-03 01:52:36 INFO - kgmoe_trainer.py -  are on are are are are are are� are are� are� are are are� are " " are are are are are the are " " " on to are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are on on on on on on on on on on on on are on on and are on on on on on on are on on are to on on are on are on are it " are are are are are are are on the on on on are are it on on on are are on on on are's on on are " are are on are are on are it are are on are are the are the are " are on� on are " are " are are on it on on are are are are on's are are it are The, " on on it are on is on it are are are it on on on on are are on on are are on on on are� on on with are are are are are are are are are's on on on on on on on on on on on the the the the on on on on on on on on on on on on on on the on on on are on on to are on on are are on are are are on are are are on on are are's are� on are are are on are are are are are are are are are are on are are on are are are are on are are are are are are are are are are on are are are are on on on on are are on on on are are are on on are are on are are are are " on are on, on are are on are are are� are are� are are are are are are are are on on are on on are as are on are are are are on on are on on the are are are are are are are are are are are are on are are are are are are are are are on are's are are are on are on on on are are on are on on are on on are are on on on on on on on on are on are on on are on on on on are on are are are are on are are on are are on are on on on on are� on the on on on on on are on on on are are are are are on are are are on on are are are on are are are are on on on are on are's's's are are are are is are are are are are are is on on on are on are on on are are are are are on are on are on on on on are are are are are are are are on are on
                # 2023-06-03 01:52:39 INFO - kgmoe_trainer.py - iput_ids
                # 2023-06-03 01:52:39 INFO - kgmoe_trainer.py -  everyone is never too old to study breathing.
                # 2023-06-03 01:52:39 INFO - kgmoe_trainer.py -  study breathing house art logic science room see book work harmony text school class student heat compare concern horse degree desk dictionary concentration discipline electronics knowledge specialist play journal motivation teacher read concentrate graduate learn learning assignment buying driving reading major minor education biology chemistry college experiment geography history investigate know law lesson literature material mathematics museum observe physician research artwork examination thought minute review test theory weather attention cabinet survey report revise population university human air dead sleep living blow coughing life pain relaxation drink lung oxygen choke cough live yawn smoke exercise neck nose roar throat ventilation wind breath survival action activity body library writing classroom paper place subject condition bar term time note doctor practice teaching state field form use experience job homework animal stuff business earth physical board eye fun breathe rest city building head type result die news child cell property entertainment good skill element thinking language physics story working world bed cat dog table home thing area inside space office party town product sound order atom energy chair set man plant newspaper shelf band box camp department doll drive institution nest dance nature item statue math discover matter method post judge view check smell face examine
                # 2023-06-03 01:52:39 INFO - kgmoe_trainer.py -  and are are are the are are are are have are are are are it on are are are are are on are are are� The The is The to are it to it it are it it and are are are are are are are are are are are are are are are are are on on on on on are are are are on on on on have have are are� it it are is are are to to it to " are are are are are are are are is on it are on are on on with on are are are are are on it are on are " are on are are are have are's "'s's are are's at " are " on on are are are are are's's are's on on on on's are are are are on on are the are the are are the are on the the on on are on are and it the are are are are are on on are the are� are are are are are " are are are are are on on are are are's are are it are are on are are on are on on the on on on are are are " on on on on on are are on the the on are are are on on on on are on on� are " on are are the are the are are on are are are are on are are are are to are are are on on on on the on on are are are " " on on and are on is on is is on on on on on on on on are on on on a on on on are on are on on it on on on on on on are are on are on on are are are are are are the on are are on on on on are on are on are on on on on on the on the the on are on on to are are are are are are are are the are are on on to it it have on are are on are on on on " are are's are are the are are are are are are are the on on are are are it are on are are are are are are " the are on on are are are on " are on on on on on on on on are are are on are on on are on are on on are are are are are are on it are it are are on are on are " are are are� are to are are are on have have are are are on are on on on on are are on are are on on are are have on are are are are are are " it on on are's are are on " on on the on on on on on are to it are are The is a are are's is's are on is it is on on is it on on are on on " on on's on on are the as are are are are are are to are are
                # 2023-06-03 01:52:42 INFO - kgmoe_trainer.py - iput_ids
                # 2023-06-03 01:52:42 INFO - kgmoe_trainer.py -  Young people should not do volunteer actions.
                # 2023-06-03 01:52:42 INFO - kgmoe_trainer.py -  action volunteer people work decide plant join accept purpose doll rest dress accomplishment event expression mechanism drive plot state software activity agency aggression application arrival ban behavior change choice cooking driving economy employment exercise fetch hug kindness mistake movement operation performance play playing reference remembering resistance reverence swing thing challenge gun keyboard key acceptance acting agent assembly capital charge combat deed device guitar lawsuit motion piano share spin stock string effect request activate actor add adopt adventure agree appeal apply arrive attack attempt attention battle beg begin beginning believe bid bit bite bless blessing blow boil bowling brain break bring build burn bury cut set point turn time use energy effort force form working leave game office job computer function machine process drop human animal fun paint plan tool case raise house bridge aim character running record fight war dance pass press condition instrument problem art book paper service design test touch trick place term cause bed train twist buy accident body kill music ball camp rush shock movie good sit practice business draw power stitch stuff close judge water board start stick protest answer stretch support role type ride wave fly opening memory sound situation stroke develop control punch bat pull walk
                # 2023-06-03 01:52:42 INFO - kgmoe_trainer.py -  are are are are are with " the the are to the are the the the are are " " " " " are " " " " " are " " " are " " are are are " " " " " " are and and as are are are on on are on are on on on on on on on on on are on are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are it are are are are are are are on on on on are on are are on are on it are on are " are " are are� are are are are are are are are are are are are are on are are are are it are are are it are are are are are the on are are are are are are are are it are are on on on are are� on are are are are to are are are are are it are's are are are are on are it are are on are are are are are are it are are on are are are are are are� are are are are it are " are are are are are are are are are are on are " " are are are are are are are The " on are are are are are on are are are are are are are are are are are are are are it are are are are are are are are are are are are are are are are are on " are are are it to are are are are on are are are are it are are are are are are are on on are are are are are are are are " are and are it are are on are are it are on are are are are are are on on on " " " on on are's's's are on on are to on are are on on are are are on are are are are are on are on on are are the are are are is are a are " are are are's are on are are on are are are it are " are on it have it are on are on on " are are are the are is are and are have on are on on is on are are are are are are are on to on are on are are are on are have� on on on on on are on on on on are on on on are it it to are on on are are are on on are on " it are are are " are are on are are are are are it are it the are are as on on it are are are are are are on are on are are are to the are are the " " " have "'s have are on are are are are " on " are are " on are are is are are are are on are are are are to the " the the the are it on are it on are are are are on� are on are on on on are are have on are are on
                # logger.info(self.tokenizer.decode(inputs["triple_labels"][0], skip_special_tokens=True))

                logger.info(inputs["triple_labels"].shape)
                generated_tokens = model.generate(
                    # Text Input!
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    # Graph Input!
                    concept_ids=inputs["concept_ids"],
                    concept_distances=inputs["concept_distances"],
                    concept_labels=inputs["concept_labels"],
                    head_ids=inputs["head_ids"],
                    tail_ids=inputs["tail_ids"],
                    relation_ids=inputs["relation_ids"],
                    triple_labels=inputs["triple_labels"],
                    # Others!
                    num_beams=self.data_args.eval_beams,
                    num_return_sequences=num_return_sequences,
                    max_length=self.max_gen_length,
                    do_sample=self.data_args.do_sample,
                    top_k=self.data_args.top_k,
                    top_p=self.data_args.top_p,
                    expert_prompt=expert_prompt,
                    use_cache=True,
                )

                # in case the batch is shorter than max length, the output should be padded
                generated_tokens = self._pad_tensors_to_max_len(generated_tokens, self.max_gen_length)

            lm_labels = inputs.get("labels")

            lm_outputs, _ = model(**inputs, use_cache=False)
            loss = self._compute_loss(lm_outputs[0], lm_labels)
            loss = loss.mean().item()
            if self.args.prediction_loss_only:
                return (loss, None, None)

            lm_logits = generated_tokens if self.args.predict_with_generate else lm_outputs[1]
        
        lm_labels = self.repeat(lm_labels, self.data_args.eval_beams)
        lm_labels = self._pad_tensors_to_max_len(lm_labels.detach(), self.max_gen_length)
        return lm_logits, lm_labels
    
    def prediction_loop(
        self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:

        if hasattr(self, "_prediction_loop"):
            warnings.warn(
                "The `_prediction_loop` method is deprecated and won't be called in a future version, define `prediction_loop` in your subclass.",
                FutureWarning,)
            return self._prediction_loop(dataloader, description, prediction_loss_only=prediction_loss_only)

        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only)

        assert not getattr(
            self.model.config, "output_attentions", False
        ), "The prediction loop does not work with `output_attentions=True`."
        assert not getattr(
            self.model.config, "output_hidden_states", False
        ), "The prediction loop does not work with `output_hidden_states=True`."

        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(self.model)
        else:
            model = self.model
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        logger.info('Entering kgmoe_trainer.py prediction_loop')
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        preds: torch.Tensor = None
        label_ids: torch.Tensor = None
        model.eval()

        if self.args.past_index >= 0:
            self._past = None

        disable_tqdm = not self.is_local_process_zero() or self.args.disable_tqdm
        ''' eval all datas in the dev set '''
        for inputs in tqdm(dataloader, desc=description, disable=disable_tqdm):
            # logging.info('input_ids and labels')
            # logging.info(self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))
            # logging.info(self.tokenizer.decode(inputs['labels'][0], skip_special_tokens=True))
            # 2023-06-02 23:48:27 INFO - kgmoe_trainer.py - input_ids and labels
            # 2023-06-02 23:48:27 INFO - kgmoe_trainer.py -  he put elephant into the jug.
            # 2023-06-02 23:48:27 INFO - kgmoe_trainer.py -  a jug is so small than an elephant.      An elephant won't fit into a jug - they are too big.    The jug would break under the weight.
            # 2023-06-02 23:48:31 INFO - kgmoe_trainer.py - input_ids and labels
            # 2023-06-02 23:48:31 INFO - kgmoe_trainer.py -  I ate grilled and peppered red Stone for dinner yesterday.
            # 2023-06-02 23:48:31 INFO - kgmoe_trainer.py -  Stone iscannot no food and cannot be eaten.      red stone will break your teeth no matter how well grilled.     You cannot eat stone.
            # 2023-06-02 23:48:33 INFO - kgmoe_trainer.py - input_ids and labels
            # 2023-06-02 23:48:33 INFO - kgmoe_trainer.py -  He felt pain in teeth, so he went to police station. 2023-06-02 23:48:33 INFO - kgmoe_trainer.py -  Hospital has dentist, who can cure teeth pain.   Police can not help you with your teeth.        the police station does not have dentists.
            # 2023-06-02 23:48:36 INFO - kgmoe_trainer.py - input_ids and labels
            # 2023-06-02 23:48:36 INFO - kgmoe_trainer.py -  The mother fries stones for her children in the morning.
            # 2023-06-02 23:48:36 INFO - kgmoe_trainer.py -  The mother should know that stones can't be eaten. no one eat stones.      stones aren't food.
            # 2023-06-02 23:48:39 INFO - kgmoe_trainer.py - input_ids and labels
            # 2023-06-02 23:48:39 INFO - kgmoe_trainer.py -  The man thought the task was too spicy.
            # 2023-06-02 23:48:39 INFO - kgmoe_trainer.py -  A task is a concept and has no flavor.   Tasks has no taste spicy or otherwise unless the task is to eat food.   Tasks are done and not tasted.
            lm_logits, lm_labels = self.prediction_step(model, inputs, prediction_loss_only)
            batch_size = inputs[list(inputs.keys())[0]].shape[0]
            if lm_logits is not None:
                preds = lm_logits if preds is None else nested_concat(preds, lm_logits, dim=0)
            if lm_labels is not None:
                label_ids = lm_labels if label_ids is None else nested_concat(label_ids, lm_labels, dim=0)

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Finally, turn the aggregated tensors into numpy arrays.
        if preds is not None:
            preds = nested_numpify(preds)
        if label_ids is not None:
            label_ids = nested_numpify(label_ids)

        assert preds.shape[0] == label_ids.shape[0]
        # ppredictions and label_ids are numpy arrays
        # logging.info("Before processing")
        # logging.info(f"Shape {preds.shape} {label_ids.shape}")
        # 2023-05-29 17:58:34 INFO - kgmoe_trainer.py - Before processing
        # 2023-05-29 17:58:34 INFO - kgmoe_trainer.py - Shape (1428, 60) (1428, 60)

        # logging.info('max')
        # logging.info(max(label_ids))


        # # Extract the predictions and labels from the EvalPrediction object
        # logging.info('Entering compute_metrics')
        # eval_prediction=EvalPrediction(predictions=preds, label_ids=label_ids)

        # # Get the predicted and true labels from the EvalPrediction object
        # preds = eval_prediction.predictions.argmax(axis=1)
        # labels = eval_prediction.label_ids

        # # Compute F1 score
        # f1 = f1_score(labels, preds, average='macro')

        # # Compute precision
        # precision = precision_score(labels, preds, average='macro')
        # metrics = {
        #     'f1' :f1,
        #     'precision': precision
        # }
        # logging.info('metrics')
        # logging.info(metrics)

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

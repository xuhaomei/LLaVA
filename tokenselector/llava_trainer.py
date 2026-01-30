from llava.train.llava_trainer import LLaVATrainer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer import Trainer
import torch
import os
from torch.nn import CrossEntropyLoss
from .eval import eval_textvqa, eval_gqa, eval_scienceqa, eval_pope, eval_mme

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return

class LLaVATokenSelectorTrainer(LLaVATrainer):
    def create_optimizer(self):
        """
        Setup the optimizer.
        This method is nearly identical to the one in Trainer, but only optimizes
        the token selector parameters.
        """
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(self.model.model.token_selector.parameters(), **optimizer_kwargs)

        return self.optimizer
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        reward_list = []
        log_probs_list = []
        has_no_image = False
        for _ in range(self.args.sample_num):
            outputs_tuple = model(**inputs)
            outputs = outputs_tuple[0]  # Get the main output
            labels = outputs_tuple[1]
            # Save past state if it exists
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            shift_logits = outputs["logits"][..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="none")
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            ce_loss = loss_fct(shift_logits, shift_labels)
            ce_loss = ce_loss.reshape(outputs["logits"].size(0),-1).mean(-1)

            # Bernoulli sample
            if model.mode == 1:
                lambda_r = self.args.lambda_r
                mask = outputs_tuple[3]
                keep_ratio = mask.sum() / 576
                r_target = model.k / 576
                reward = - ce_loss.detach() - lambda_r * (keep_ratio - r_target)**2   # Detach to avoid backprop through reward
            else:
                reward = - ce_loss.detach()
            log_probs = outputs_tuple[2] 
            reward_list.append(reward)
            log_probs_list.append(log_probs)
            if log_probs is None:
                has_no_image = True
                break

        if has_no_image:
            loss = torch.tensor(0.0, device=model.device, requires_grad=False)
            return (loss, outputs) if return_outputs else loss
        alpha = self.args.alpha_pg_loss
        reward_list = torch.stack(reward_list) # (Sample_num,B)
        self.log({'reward': reward_list.mean().item()})

        reward_list = reward_list - reward_list.mean(dim=0)
        log_probs_list = torch.stack(log_probs_list) #(Sample_num,B)
        loss = - (log_probs_list * reward_list).mean() * alpha

        return (loss, outputs) if return_outputs else loss

    def evaluate(
        self,
        eval_dataset = None,
        ignore_keys = None,
        metric_key_prefix: str = "eval",
    ):
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (Union[`Dataset`, Dict[str, `Dataset`]), *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. If it is a dictionary, it will
                evaluate on each dataset, prepending the dictionary key to the metric name. Datasets must implement the
                `__len__` method.

                <Tip>

                If you pass a dictionary with names of datasets as keys and datasets as values, evaluate will run
                separate evaluations on each dataset. This can be useful to monitor how training affects other
                datasets or simply to get a more fine-grained evaluation.
                When used with `load_best_model_at_end`, make sure `metric_for_best_model` references exactly one
                of the datasets. If you, for example, pass in `{"data1": data1, "data2": data2}` for two datasets
                `data1` and `data2`, you could specify `metric_for_best_model="eval_data1_loss"` for using the
                loss on `data1` and `metric_for_best_model="eval_data2_loss"` for the loss on `data2`.

                </Tip>

            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        model_path = os.path.join(self.args.output_dir, f"step-{self.state.global_step}")
        ckpt = model_path.split('/')[-2] + '-' + model_path.split('/')[-1]
        self.model.eval()
        mme_acc = eval_mme(ckpt, self.model)
        pope_acc = eval_pope(ckpt, self.model)
        scienceqa_acc = eval_scienceqa(ckpt, self.model)
        gqa_acc = eval_gqa(ckpt, self.model)
        textvqa_acc = eval_textvqa(ckpt, self.model)
        metrics = {
            "textvqa_acc": float(textvqa_acc),
            "gqa_acc": float(gqa_acc),
            "scienceqa_acc": float(scienceqa_acc),
            "pope_acc": float(pope_acc),
            "mme_acc": float(mme_acc)
        }
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)
        self.model.train()
        return {}
    
    def _save_checkpoint(self, model, trial, metrics=None):
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        # Only save Selector
        keys_to_match = ['token_selector']

        weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

        if self.args.local_rank == 0 or self.args.local_rank == -1:
            self.model.config.save_pretrained(output_dir)
            torch.save(weight_to_save, os.path.join(output_dir, "token_selector.bin"))

    def _save(self, output_dir=None, state_dict=None):
        pass
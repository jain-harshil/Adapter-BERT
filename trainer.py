import argparse
import glob
import json
import logging
import os
import random
import helper
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def prepare_optimizer_and_scheduler(model, t_total, params):
    c = params["mcqa_config"]
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": c.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=c.learning_rate, eps=c.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=c.warmup_steps, num_training_steps=t_total)

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(c.model_name_or_path, "optimizer.pt")) and os.path.isfile(os.path.join(c.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(c.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(c.model_name_or_path, "scheduler.pt")))

    return optimizer, scheduler

def save_model(model, optimizer, scheduler, step, dev_scores, params):
    # Save model checkpoint
    c = params["mcqa_config"]

    output_dir = os.path.join(c.output_dir, "best")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_to_save = (model.module if hasattr(model, "module") else model)  
    model_to_save.save_pretrained(output_dir)

    #torch.save(c, os.path.join(output_dir, "training_c.bin"))
    logger.info("Saving model checkpoint to %s", output_dir)

    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

    global_step_path = os.path.join(c.output_dir, "best", "global_step.txt")
    helper.write_list(global_step_path, [str(step)])

    if dev_scores:
        eval_output_dir = os.path.join(c.output_dir, "eval")
        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)

        output_eval_file = os.path.join(eval_output_dir, "eval_result.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format("CONCEPTNET DEV"))
            for key in sorted(dev_scores.keys()):
                logger.info("  %s = %s", key, str(dev_scores[key]))
                writer.write("%s = %s\n" % (key, str(dev_scores[key])))

    logger.info("Saving optimizer and scheduler states to %s", output_dir) 

def train(train_dataset, eval_dataset, model, params):
    """ Train the model """
    c = params["mcqa_config"]
    
    # enabling TensorBoard
    tb_writer = SummaryWriter()

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=c.train_batch_size)

    if c.max_steps > 0:
        t_total = c.max_steps
        c.num_train_epochs = c.max_steps // len(train_dataloader) + 1
    else:
        t_total = len(train_dataloader) * c.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer, scheduler = prepare_optimizer_and_scheduler(model, t_total, params)
    
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", c.num_train_epochs)
    logger.info("  Train batch size = %d", c.train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    # Check if continuing training from a checkpoint
    if os.path.exists(c.model_name_or_path):
        # set global_step to global_step of last saved checkpoint from model path
        try:
            global_step = int(helper.load_lines(os.path.join(c.model_name_or_path, "global_step.txt"))[0].strip())
        except ValueError:
            global_step = 0
        epochs_trained = global_step // len(train_dataloader)
        steps_trained_in_current_epoch = global_step % len(train_dataloader)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    train_iterator = trange(epochs_trained, int(c.num_train_epochs), desc="Epoch", disable=False)

    set_seed(c.seed)  # Added here for reproductibility, every new training starts from the same seed, i.e., same parameter initialization

    eval_steps_no_improvement = 0
    stop_training = False
    best_eval_res = -1000000 if c.eval_metric_increasing else 1000000

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train() 
            batch = tuple(t.to(c.device) for t in batch)

            if params["task_type"] == "mcqa":
                outputs = model(batch, params["model_params"])
            else:
                outputs = model(input_ids=batch[0], attention_mask=batch[1], token_type_ids=batch[2], labels=batch[3])

            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            
            # parameter updates
            loss.backward()
            tr_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), c.max_grad_norm)

            optimizer.step()
            scheduler.step()  # Update learning rate schedule

            model.zero_grad() # zeroing gradients afer update

            global_step += 1

            if c.logging_steps > 0 and global_step % c.logging_steps == 0:
                print("Global step: " + str(global_step))
                logs = {}
                results = evaluate(eval_dataset, model, params)
                for key, value in results.items():
                    eval_key = "eval_{}".format(key)
                    logs[eval_key] = value

                loss_scalar = (tr_loss - logging_loss) / c.logging_steps
                learning_rate_scalar = scheduler.get_lr()[0]
                logs["learning_rate"] = learning_rate_scalar
                logs["loss"] = loss_scalar
                logging_loss = tr_loss

                for key, value in logs.items():
                    tb_writer.add_scalar(key, value, global_step)
                print(json.dumps({**logs, **{"step": global_step}}))

                eval_res = results[c.eval_stop_metric]
                if (c.eval_metric_increasing and eval_res < best_eval_res) or (not c.eval_metric_increasing and eval_res > best_eval_res):
                    eval_steps_no_improvement += 1
                else:
                    eval_steps_no_improvement = 0
                
                if eval_steps_no_improvement == c.num_evals_early_stop:
                    print("Early stopping training. ")
                    stop_training = True
                    break
                    
                if eval_steps_no_improvement == 0:
                    best_eval_res = eval_res
                    print("New best eval " + c.eval_stop_metric + ": " + str(best_eval_res))
                    print("Saving best model...")
                    save_model(model, optimizer, scheduler, global_step, results, params)
                    print("New best model saved!")
                else:
                    print("No improvement for " + str(eval_steps_no_improvement) + " steps!")
                    print("Current Eval " + c.eval_stop_metric + ": " + str(eval_res))
                    print("Best Eval " + c.eval_stop_metric + " so far: " + str(best_eval_res))
                
            if c.max_steps > 0 and global_step > c.max_steps:
                epoch_iterator.close()
                break
        
        if (c.max_steps > 0 and global_step > c.max_steps) or stop_training:
            train_iterator.close()
            break
    tb_writer.close()

    return global_step, tr_loss / global_step, best_eval_res

def compute_performance(preds, golds):
    if len(preds) != len(golds):
        raise ValueError("Predictions and gold labels not of same length!")

    results = {}
    acc = len([i for i in range(len(preds)) if preds[i] == golds[i]]) / len(preds)
    results["Accuracy"] = acc
    
    return results

def evaluate(eval_dataset, model, params):
    results = {}
    c = params["mcqa_config"]
    c.eval_batch_size = c.train_batch_size

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=c.eval_batch_size)
    
    # Eval!
    logger.info("***** Running evaluation {} *****".format("CONCEPTNET VAL"))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", c.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    golds = None
    
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(c.device) for t in batch)

        with torch.no_grad():
            if params["task_type"] == "mcqa":
                outputs = model(batch, params["model_params"])
            else:
                outputs = model(input_ids=batch[0], attention_mask=batch[1], token_type_ids=batch[2], labels=batch[3])

            loss = outputs[0]
            logits = outputs[1]
            eval_loss += loss.mean().item()
            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                golds = batch[-1].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                golds = np.append(golds, batch[-1].detach().cpu().numpy())

    eval_loss = eval_loss / nb_eval_steps
    results["Loss"] = eval_loss
    preds = np.argmax(preds, axis=1)

    result = compute_performance(preds, golds)
    results.update(result)

    return results

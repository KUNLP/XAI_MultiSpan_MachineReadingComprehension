import os
import timeit

import torch
from fastprogress.fastprogress import master_bar, progress_bar
from processor import SquadResult
from squad_metric import (
    compute_predictions_logits
)
from utils import load_examples, set_seed, to_list, get_best_span, span_freeze, th_freeze
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)

from multi_measure import evaluate_prediction_file

def train(args, model, tokenizer, logger):
    global_step = 1
    mb = master_bar(range(int(args.num_train_epochs)))
    best_f1 = 0
    
    for cur_epo in range(args.num_train_epochs * args.num_datas):
        # 전체 데이터를 args.num_datas 수에 맞게 분할 및 학습 데이터 생성
        train_dataset = load_examples(args, tokenizer, evaluate=False, output_examples=False, num=cur_epo%args.num_datas)
        
        """ Train the model """
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Train batch size per GPU = %d", args.train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            args.train_batch_size
            * args.gradient_accumulation_steps)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        logger.info("  Starting fine-tuning.")

        model.zero_grad()
        # Added here for reproductibility
        set_seed(args)

        epoch_iterator = progress_bar(train_dataloader, parent=mb)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "question_mask":batch[2],
                "sentence_mask": batch[3],
                "token_type_ids": batch[4],
                "start_positions":batch[5],
                "end_positions":batch[6],
                "start_position": batch[7],
                "end_position": batch[8],
                "flag" : False
            }
            '''
               input_ids : [batch, seq_len] -> [[1, 2, 5, 23, 753 ...], ... , ]
               attention_mask : [batch, seq_len] -> [[1, 1, 1, 1, ..., 0, 0, 0, 0], ..., ]
               question_mask : [batch, seq_len] -> [[1, 1, 1, 1, 0, 0, 0, ..., ], ... ]
                               입력 sequence에서 질문에 해당하는 토큰  mask
               sentence_mask : [batch, seq_len] -> [[0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, ..., 0, 0, 0, ], ... ]
                               입력 sequence의 문장 번호
               start_position : [batch] -> [45, 12, 283, ...]
               end_position : [batch] -> [47, 18, 301, ...]
               start_positions : [batch, num_answers] -> [[45, 12, 283, ...], [45, 12, 283, ...], [45, 12, 283, .    ..], ...]
               end_positions : [batch, num_answers] -> [[47, 18, 301, ...], [47, 18, 301, ...], [47, 18, 301, ...    ], ...]
               flag : Threshold layer를 학습할지 결정하는 boolean 값
            ''' 
            
            # flag : False? -> Threshold와 관련된 Layer freeze
            model = th_freeze(model)
            # Span Prediction과 관련된 layer의 loss
            loss, matrix_loss, span_loss = model(**inputs)

            loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args,gradient_accumulation_steps

            loss.backward()
            
            # flag : True? -> Threshold와 관련된 layer unfreeze
            # Span Prediction과 관련된 layer freeze
            model = span_freeze(model)
            inputs["flag"] = True

            # Threshold Loss
            valid_loss = model(**inputs)
            
            

            valid_loss = valid_loss.mean()
            span_loss = span_loss.mean()
            matrix_loss = matrix_loss.mean()
            valid_loss.backward()             

            # model outputs are always tuple in transformers (see doc)
            if (global_step +1) % 50 == 0:
                print("{} Processing,,,, Current Total Loss : {}".format(global_step+1, loss.item()))
                print("Matrix Loss : {} \t Span Loss : {} \t valid Loss : {}".format(matrix_loss.item(), span_loss.item(), valid_loss.item()))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # # Log metrics
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    cur_f1, em = evaluate(args=args, model=model, tokenizer=tokenizer, global_step=global_step)
                    #cur_f1 = 30
                    if cur_f1 > best_f1:
                        best_f1 = cur_f1
                        model.module.save_pretrained(args.output_dir)
                        tokenizer.save_pretrained(args.output_dir)
                        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", args.output_dir)

def evaluate(args, model, tokenizer, global_step=None):
    dataset, examples, features = load_examples(args, tokenizer, evaluate=True, output_examples=True)
    
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)


    all_results = []
    start_time = timeit.default_timer()

    for batch in progress_bar(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "question_mask":batch[2],
                "sentence_mask": batch[3],
                "token_type_ids": batch[4],
            }

            feature_indices = batch[5]

            outputs = model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]

            unique_id = int(eval_feature.unique_id)

            start_logits, end_logits, matrix_logits, th_logits = [to_list(output[i]) for output in outputs]
            best_tok_spans = get_best_span(len(eval_feature.tokens), eval_feature.token_to_orig_map, args.n_best_size,
                                           start_logits, end_logits, matrix_logits, th_logits)
            result = SquadResult(unique_id, best_tok_spans)

            all_results.append(result)
    evalTime = timeit.default_timer() - start_time
    print(evalTime)
    output_null_log_odds_file = None
    
    prediction_file = './results/predictions_{}.json'.format(global_step)
    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        args.do_lower_case,
        prediction_file,
        None,
        output_null_log_odds_file,
        args.verbose_logging,
        args.version_2_with_negative,
        args.null_score_diff_threshold,
        tokenizer,
    )
    # Write the evaluation result on file
    best_f1 = 0
    best_thres = -1
    em, f1 = evaluate_prediction_file(predictions, args.dev_file_path)
    
    return f1, em

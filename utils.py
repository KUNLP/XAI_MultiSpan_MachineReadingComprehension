import logging
import random

import numpy as np
import torch

from processor import (
    SquadV1Processor,
    SquadV2Processor,
    squad_convert_examples_to_features
)


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

# tensor를 list 형으로 변환하기위한 함수
def to_list(tensor):
    return tensor.detach().cpu().tolist()

# dataset을 load 하는 함수
def load_examples(args, tokenizer, evaluate=False, output_examples=False, num=-1):
    '''

    :param args: 하이퍼 파라미터
    :param tokenizer: tokenization에 사용되는 tokenizer
    :param evaluate: 평가나 open test시, True
    :param output_examples: 평가나 open test 시, True / True 일 경우, examples와 features를 같이 return
    :return:
    examples : max_length 상관 없이, 원문으로 각 데이터를 저장한 리스트
    features : max_length에 따라 분할 및 tokenize된 원문 리스트
    dataset : max_length에 따라 분할 및 학습에 직접적으로 사용되는 tensor 형태로 변환된 입력 ids
    '''

    # processor 선언
    processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()

    # 평가 시 
    if evaluate:
        examples = processor.get_dev_examples(args.dev_file_path)
    # 학습 시
    else:
        examples = processor.get_train_examples(args.train_file_path)
    if num >= 0:
        part = int(len(examples)/args.num_datas)
        examples = examples[part*num:part*(num+1)]
    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=not evaluate,
        return_dataset="pt",
        threads=args.threads,
    )

    if output_examples:
        return dataset, examples, features
    return dataset

def get_best_index(logits, n_best_size):
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes

def get_best_span(valid_range, valid_dict, n_best_size, start_logits, end_logits, span_matrix, threshold):
    spans = []
    start_indexes = get_best_index(start_logits, n_best_size)
    end_indexes = get_best_index(end_logits, n_best_size)
    # if we could have irrelevant answers, get the min score of irrelevant
    for start_index in start_indexes:
        for end_index in end_indexes:
            if start_index >= valid_range:
                continue
            if end_index >= valid_range:
                continue
            if start_index not in valid_dict:
                continue
            if end_index not in valid_dict:
                continue
            if end_index < start_index:
                continue
            span = [start_index, end_index, start_logits[start_index], end_logits[end_index]
                , span_matrix[start_index][end_index], threshold]
            spans.append(span)
    return spans

def span_freeze(model):
    for name, param in model.module.named_parameters():
        if 'th_outputs' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    return model

def th_freeze(model):
    for name, param in model.module.named_parameters():
        if 'th_outputs' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    return model



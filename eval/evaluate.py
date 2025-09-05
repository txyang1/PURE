import argparse
import numpy as np
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError

from grader import *

from parser import *
from utils import load_jsonl
from python_executor import PythonExecutor


'''def evaluate(data_name, prompt_type, samples: list=None, file_path: str=None, max_num_samples=None, execute=False):
    assert samples or file_path, "samples or file_path must be provided"
    if not samples:
        samples = list(load_jsonl(file_path))
    if 'idx' in samples[0]:
        samples = {sample['idx']: sample for sample in samples}.values()
        samples = sorted(samples, key=lambda x: x['idx']) 
    else:
        samples = [dict(idx=idx, **sample) for idx, sample in enumerate(samples)]

    if max_num_samples:
        print(f"max_num_samples: {max_num_samples} / {len(samples)}")
        samples = samples[:max_num_samples]
    
    # parse gt
    for sample in samples:
        sample['gt_cot'], sample['gt'] = parse_ground_truth(sample, data_name)
    params = [(idx, pred, sample['gt']) for idx, sample in enumerate(samples) for pred in sample['pred']]

    scores = []
    timeout_cnt = 0 

    with ProcessPool(max_workers=1) as pool:
        future = pool.map(math_equal_process, params, timeout=3)
        iterator = future.result()
        with tqdm(total=len(samples), desc="Evaluate") as progress_bar:
            while True:
                try:
                    result = next(iterator)
                    scores.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error)
                    scores.append(False)
                    timeout_cnt += 1
                except Exception as error:
                    print(error.traceback)
                    exit()
                progress_bar.update(1) 

    idx = 0
    score_mat = []
    for sample in samples:
        sample['score'] = scores[idx: idx+len(sample['pred'])]
        assert len(sample['score']) == len(sample['pred'])
        score_mat.append(sample['score'])
        idx += len(sample['pred'])

    max_len = max([len(s) for s in score_mat])

    for i, s in enumerate(score_mat):
        if len(s) < max_len:
            score_mat[i] = s + [s[-1]] * (max_len - len(s)) # pad

    # output mean of each column of scores
    col_means= np.array(score_mat).mean(axis=0)
    mean_score = list(np.round(col_means * 100, decimals=1))

    result_json = {
        "num_samples": len(samples),
        "num_scores": len(scores),
        "timeout_samples": timeout_cnt,
        "empty_samples": len([s for s in samples if not s['pred'][-1]]),
        "acc": mean_score[0]
    }

    # each type score
    if "type" in samples[0]:
        type_scores = {}
        for sample in samples:
            if sample['type'] not in type_scores:
                type_scores[sample['type']] = []
            type_scores[sample['type']].append(sample['score'][-1])
        type_scores = {k: np.round(np.array(v).mean() * 100, decimals=1) for k, v in type_scores.items()}
        type_scores = {k: v for k, v in sorted(type_scores.items(), key=lambda item: item[0])}
        result_json['type_acc'] = type_scores

    print(result_json)
    return samples, result_json'''
def evaluate(
    data_name,
    prompt_type,
    samples: list = None,
    file_path: str = None,
    max_num_samples=None,
    execute=False,
    pass_k=None,            # ← 新增：int 或 list[int]；None 表示等价于 [1]
    unbiased=False,         # ← 新增：是否用无偏估计
    dedup=False,            # ← 新增：是否对同题重复预测去重
    ):
    from math import comb
    assert samples or file_path, "samples or file_path must be provided"
    if not samples:
        samples = list(load_jsonl(file_path))
    if 'idx' in samples[0]:
        samples = {sample['idx']: sample for sample in samples}.values()
        samples = sorted(samples, key=lambda x: x['idx'])
    else:
        samples = [dict(idx=idx, **sample) for idx, sample in enumerate(samples)]

    if max_num_samples:
        print(f"max_num_samples: {max_num_samples} / {len(samples)}")
        samples = samples[:max_num_samples]

    # parse gt
    for sample in samples:
        sample['gt_cot'], sample['gt'] = parse_ground_truth(sample, data_name)

    # 将所有 (样本索引, 该次预测, 标准答案) 展开为判分任务
    params = []
    for idx, sample in enumerate(samples):
        for pred in sample['pred']:
            params.append((idx, pred, sample['gt']))

    scores = []
    timeout_cnt = 0

    # 注意：这里的 total 应为 len(params)，不是 len(samples)
    with ProcessPool(max_workers=1) as pool:
        future = pool.map(math_equal_process, params, timeout=3)
        iterator = future.result()
        with tqdm(total=len(params), desc="Evaluate") as progress_bar:
            while True:
                try:
                    result = next(iterator)
                    scores.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error)
                    scores.append(False)
                    timeout_cnt += 1
                except Exception as error:
                    print(error.traceback)
                    exit()
                progress_bar.update(1)

    # 回填每道题的逐次判分结果
    idx = 0
    for sample in samples:
        n = len(sample['pred'])
        sample['score'] = scores[idx: idx + n]  # 布尔列表
        assert len(sample['score']) == n
        idx += n

    # 组装 pass@k 统计
    if pass_k is None:
        k_list = [1]  # 与旧版兼容：默认相当于只看第 1 次
    elif isinstance(pass_k, int):
        k_list = [pass_k]
    else:
        k_list = list(pass_k)

    per_k_hits = {k: [] for k in k_list}

    for sample in samples:
        preds = sample['pred']
        sc = sample['score']

        # （可选）对重复预测去重（按文本）
        if dedup:
            seen = set()
            new_preds, new_sc = [], []
            for p, s in zip(preds, sc):
                if p not in seen:
                    seen.add(p)
                    new_preds.append(p)
                    new_sc.append(s)
            preds, sc = new_preds, new_sc

        n = len(sc)
        c = sum(1 for x in sc if x)

        for k in k_list:
            kk = min(k, n)  # 防越界
            if kk == 0:
                hit = 0.0
            elif unbiased:
                # HumanEval 无偏估计：1 - C(n-c, k)/C(n, k)
                if n < kk:
                    hit = 1.0 if c > 0 else 0.0
                elif n - c < kk:
                    hit = 1.0
                else:
                    hit = 1.0 - (comb(n - c, kk) / comb(n, kk))
            else:
                # 任一命中即对（常用口径）
                hit = 1.0 if any(sc[:kk]) else 0.0
            per_k_hits[k].append(hit)

    pass_at_k = {str(k): round(100 * float(np.mean(per_k_hits[k])), 1) if per_k_hits[k] else 0.0
                 for k in k_list}

    # 为了兼容旧字段，把 acc 定义为第一个 k 的分数
    primary_k = k_list[0]
    result_json = {
        "num_samples": len(samples),
        "num_scores": len(scores),
        "timeout_samples": timeout_cnt,
        "empty_samples": len([s for s in samples if not s['pred'][-1]]),
        "acc": pass_at_k[str(primary_k)],
        "pass_at_k": pass_at_k,
        "primary_k": primary_k,
        "unbiased": unbiased,
        "dedup": dedup,
    }

    # 分类型成绩（按 primary_k）
    if "type" in samples[0]:
        type_scores = {}
        for sample in samples:
            preds = sample['pred']
            sc = sample['score']
            if dedup:
                seen = set(); new_sc = []
                for p, s in zip(preds, sc):
                    if p not in seen:
                        seen.add(p)
                        new_sc.append(s)
                sc = new_sc
            kk = min(primary_k, len(sc))
            hit = 1.0 if kk > 0 and any(sc[:kk]) else 0.0
            type_scores.setdefault(sample['type'], []).append(hit)
        type_scores = {k: round(100 * float(np.mean(v)), 1) for k, v in type_scores.items()}
        type_scores = dict(sorted(type_scores.items(), key=lambda item: item[0]))
        result_json['type_acc'] = type_scores

    print(result_json)
    return samples, result_json



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="math")
    parser.add_argument("--prompt_type", type=str, default="tool-integrated")
    parser.add_argument("--file_path", type=str, default=None, required=True)
    parser.add_argument("--max_num_samples", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    evaluate(data_name=args.data_name, prompt_type=args.prompt_type, file_path=args.file_path,
             max_num_samples=args.max_num_samples, execute=args.execute)

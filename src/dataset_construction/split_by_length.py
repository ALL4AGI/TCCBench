"""Level 1. (Balance): 2. (Keyword): Keyword 3. (WordOnlyOnce) Level 1: [5, 25) Level 2: [25, 45) Level 3: [45, 65) # Level 4: [65, 85) # Level 5: [85, 120) 200"""

import json
import os
from collections import defaultdict

# --- Configuration Parameters ---
INPUT_FILE = r'D:\01project\wordorder\dataset02\cc_stories\cc_stories1_en02_dedup.jsonl'
OUTPUT_TEST_FILE = r'D:\01project\wordorder\dataset02\test\cc_stories1_en02_test.jsonl'
OUTPUT_REPORT_FILE = r'D:\01project\wordorder\dataset02\test\cc_stories1_en02_test.txt'
OUTPUT_TRAIN_FILE = r'D:\01project\wordorder\dataset02\train\cc_stories1_en02_train.jsonl'

LEVEL_CONFIG = {
    "Level_1": {"range": (5, 25), "target": 200},
    "Level_2": {"range": (25, 45), "target": 200},
    "Level_3": {"range": (45, 65), "target": 200},
}

def load_data(filepath):
    print(f"正在加载原始数据: {filepath} ...")
    data_pool = defaultdict(lambda: defaultdict(list))
    all_items = []
    if not os.path.exists(filepath):
        print(f"错误: 找不到输入文件 {filepath}")
        return None, None

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            all_items.append(item)
            try:
                count = int(item.get("word_count", 0))
            except:
                continue
            for lvl_name, config in LEVEL_CONFIG.items():
                min_l, max_l = config["range"]
                if min_l <= count < max_l:
                    data_pool[lvl_name][count].append(item)
                    break
    return data_pool, all_items

def water_filling_allocation(buckets, target_total):
    available_lengths = list(buckets.keys())
    if not available_lengths: return {}
    capacities = {l: len(buckets[l]) for l in available_lengths}
    allocation = {l: 0 for l in available_lengths}
    remaining_target = target_total
    while remaining_target > 0:
        not_full_lengths = [l for l in available_lengths if allocation[l] < capacities[l]]
        if not not_full_lengths: break
        count = len(not_full_lengths)
        distribute = max(1, remaining_target // count)
        if remaining_target < count:
            distribute = 1
            not_full_lengths = not_full_lengths[:remaining_target]
        allocated_this_round = 0
        for l in not_full_lengths:
            actual = min(distribute, capacities[l] - allocation[l])
            allocation[l] += actual
            remaining_target -= actual
            allocated_this_round += actual
            if remaining_target <= 0: break
        if allocated_this_round == 0: break
    return allocation

def select_mixed_items(candidates, quota, global_seen_keywords):

    yes_group = [i for i in candidates if i.get("word_only_once") == "yes"]
    no_group = [i for i in candidates if i.get("word_only_once") == "no"]

    # Keyword
    def get_score(item):
        return 1 if item.get("l4_keyword", "") not in global_seen_keywords else 0

    yes_group.sort(key=get_score, reverse=True)
    no_group.sort(key=get_score, reverse=True)

    selected = []

    target_yes = quota // 2
    target_no = quota - target_yes    # NO

    # 1.
    picked_yes = yes_group[:target_yes]
    picked_no = no_group[:target_no]

    selected.extend(picked_yes)
    selected.extend(picked_no)

    # 2.
    if len(selected) < quota:
        remaining_needed = quota - len(selected)
        # YES not,  no_group
        if len(picked_yes) < target_yes:
            extra_no = no_group[target_no: target_no + remaining_needed]
            selected.extend(extra_no)
        # NO not,  yes_group
        elif len(picked_no) < target_no:
            extra_yes = yes_group[target_yes: target_yes + remaining_needed]
            selected.extend(extra_yes)

    # Keyword
    for item in selected:
        kw = item.get("l4_keyword", "")
        if kw: global_seen_keywords.add(kw)

    return selected

def main():
    for path in [OUTPUT_TEST_FILE, OUTPUT_REPORT_FILE, OUTPUT_TRAIN_FILE]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    data_pool, all_items = load_data(INPUT_FILE)
    if not data_pool: return

    final_test_set = []
    selected_sentences = set()
    global_seen_keywords = set()
    stats_report_content = []

    print("开始采样 (双向补充逻辑)...")

    for level_name, config in LEVEL_CONFIG.items():
        level_buckets = data_pool.get(level_name, {})
        target = config["target"]
        min_l, max_l = config["range"]

        allocation = water_filling_allocation(level_buckets, target)
        level_items = []
        sorted_lengths = sorted(allocation.keys())

        for length in sorted_lengths:
            quota = allocation[length]
            level_items.extend(select_mixed_items(level_buckets[length], quota, global_seen_keywords))

        final_test_set.extend(level_items)
        for item in level_items:
            selected_sentences.add(item.get("original_sentence"))

        # --- ---
        actual_total = sum(len(v) for v in level_buckets.values())
        level_selected_count = len(level_items)
        unique_kws = set(item.get("l4_keyword") for item in level_items)
        word_once_count = sum(1 for item in level_items if item.get("word_only_once") == "yes")

        report_lines = [
            f"=== {level_name} 分析 ===",
            f"字数范围: [{min_l}, {max_l})",
            f"原始库存: {actual_total} 条",
            f"目标数量: {target} 条",
            f"实际选中: {level_selected_count} 条 (满足率: {level_selected_count / target:.1%})",
            f"Keyword覆盖数: {len(unique_kws)} 个",
            f"WordOnlyOnce占比: {word_once_count}/{level_selected_count}",
            f"长度分布明细 (长度: 选中数/库存数):"
        ]
        dist_details = [f"{l}字: {allocation[l]}/{len(level_buckets[l])}" for l in sorted_lengths]
        report_lines.append(", ".join(dist_details))
        report_lines.append("\n")
        stats_report_content.extend(report_lines)

    train_set = [item for item in all_items if item.get("original_sentence") not in selected_sentences]

    with open(OUTPUT_TEST_FILE, 'w', encoding='utf-8') as f:
        for item in final_test_set:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    with open(OUTPUT_TRAIN_FILE, 'w', encoding='utf-8') as f:
        for item in train_set:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    with open(OUTPUT_REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(stats_report_content))
        f.write(f"\nTotal Selected: {len(final_test_set)} | Total Train: {len(train_set)}\n")

    print(f"采样完成.测试集: {len(final_test_set)}, 训练集: {len(train_set)}")

if __name__ == "__main__":
    main()

# ############################################

# # --- Configuration Parameters ---
# # (min_len, max_len, target_count)
# }
# # (int)
# continue
# # Level
# min_l, max_l = config["range"]
# break
# # Level
# , .
# buckets (dict: length -> list of items), target_total (int)
# allocation (dict: length -> int quota)
# #
# # : 0
# # ""
# break # ,
# #
# # not1,
# # remaining_target
# #
# # attempts
# # (not)
# remaining_target -= actual
# allocated_in_round += actual
# break
# # (not, )
# break
# Priority quota
# Priority: 1. Keyword > 2. WordOnlyOnce
# # Greedy selection loop
# # "Keyword" attempts
# # , .
# # ,  Quota ( 10 ), +
# # "Keyword" .
# #
# # candidates
# break
# #
# # B: Keyword
# score += 10
# # C: not
# score += 1
# #
# selected.append(picked)
# # Keyword
# global_seen_keywords.add(kw)
# # 1. Output directory
# # 2.
# # 3. Level
# min_l, max_l = config["range"]
# # 3.1
# # allocation: {length: quota}
# # 3.2 ,
# # ,
# #
# level_items.extend(selected_items)
# level_selected_count += len(selected_items)
# #
# final_dataset.extend(level_items)
# # --- Level ---
# #
# dist_str.append(f"{l}: {q}/{total}")
# report_lines.append(", ".join(dist_str))
# report_lines.append("\n")
# stats_report.extend(report_lines)
# # 4.
# f.write(json.dumps(item, ensure_ascii=False) + '\n')
# # 5.
# f.write("\n".join(stats_report))
# f.write(f"\nTotal Selected: {len(final_dataset)}\n")

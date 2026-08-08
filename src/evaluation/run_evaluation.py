import json
import os
import time
import re
import sys
import math
from collections import Counter
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# Try to import ZhipuAI client
try:
    from zai import ZhipuAiClient
    ZHIPU_AVAILABLE = True
except ImportError:
    ZHIPU_AVAILABLE = False
    print("[Warning] ZhipuAI SDK not installed. Install with: pip install zai")

# =========================================
# 1. Configuration
# =========================================

INPUT_FILE_PATH = r"D:\01project\wordorder\LLMtest2split\TCCBench_gpt_split.jsonl"
OUTPUT_FILE_PATH = r"D:\01project\wordorder\LLMtest3evaluation\TCCBench_gpt_evalu.jsonl"
LOG_FILE_PATH = r"D:\01project\wordorder\LLMtest3evaluation\TCCBench_gpt_evalu.log"

# 3 LLM Agents
LLM_AGENTS = {
    "agent01": {
        "api_key": "YOUR_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-5-mini",
        "sdk": "openai"
    },
    "agent02": {
        "api_key": "YOUR_API_KEY",
        "base_url": "https://api.deepseek.com/v1/",
        "model": "deepseek-chat",
        "sdk": "openai"
    },
    "agent03": {
        "api_key": "YOUR_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "sdk": "openai"
    }
}

MAX_RETRIES = 5
RETRY_DELAY = 20
MAX_WORKERS = 3    # 3 agents

log_file = None

# =========================================
# Logging utility functions
# =========================================

def init_log_file():

    global log_file
    try:
        log_file = open(LOG_FILE_PATH, 'a', encoding='utf-8')
    except Exception as e:
        print(f"[Error] 无法打开日志文件: {e}")

def close_log_file():

    global log_file
    if log_file:
        log_file.close()

def log_message(message):

    global log_file
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()

def log_separator(line_number):

    separator = "*" * 60
    log_message(f"\n{separator}")
    log_message(f"line {line_number}")

# =========================================
# 2. Utility functions: Metric computation
# =========================================

def safe_div(n, d):
    return n / d if d > 0 else 0.0

def calculate_f1(precision, recall):
    return safe_div(2 * precision * recall, precision + recall)

def get_tokens(text, language):
    """token"""
    if not text:
        return []
    if language == 'zh':
        # Chinese: split by character
        return [c for c in text if c.strip()]
    else:
        # English: split by whitespacesplit
        # , orsplit
        # , n-gramor.split
        return text.split()

def compute_compliance_f1(word_list, response_list):
    """F1 Score Counter"""
    ref_counter = Counter(word_list)
    hyp_counter = Counter(response_list)

    intersection = ref_counter & hyp_counter
    match_count = sum(intersection.values())

    len_ref = len(word_list)
    len_hyp = len(response_list)

    precision = safe_div(match_count, len_hyp)
    recall = safe_div(match_count, len_ref)

    return calculate_f1(precision, recall)

def compute_ngram_f1(hyp_text, ref_text, language, n):
    """N-gram F1"""
    hyp_tokens = get_tokens(hyp_text, language)
    ref_tokens = get_tokens(ref_text, language)

    if len(hyp_tokens) < n or len(ref_tokens) < n:
        return 0.0

    # Generate n-grams
    hyp_ngrams = [tuple(hyp_tokens[i:i + n]) for i in range(len(hyp_tokens) - n + 1)]
    ref_ngrams = [tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1)]

    hyp_counter = Counter(hyp_ngrams)
    ref_counter = Counter(ref_ngrams)

    intersection = hyp_counter & ref_counter
    match_count = sum(intersection.values())

    precision = safe_div(match_count, len(hyp_ngrams))
    recall = safe_div(match_count, len(ref_ngrams))

    return calculate_f1(precision, recall)

def compute_rouge_l_f1(hyp_text, ref_text, language):
    """ROUGE-L F1 ( LCS) LCS"""
    hyp_tokens = get_tokens(hyp_text, language)
    ref_tokens = get_tokens(ref_text, language)

    if not hyp_tokens or not ref_tokens:
        return 0.0

    len_hyp = len(hyp_tokens)
    len_ref = len(ref_tokens)

    # LCS
    dp = [[0] * (len_ref + 1) for _ in range(len_hyp + 1)]

    for i in range(1, len_hyp + 1):
        for j in range(1, len_ref + 1):
            if hyp_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[len_hyp][len_ref]

    precision = safe_div(lcs_len, len_hyp)
    recall = safe_div(lcs_len, len_ref)

    # ROUGE-L beta 1,  F1
    return calculate_f1(precision, recall)

# =========================================
# 3. LLM client initialization
# =========================================

def get_client(agent_name):
    """agent"""
    config = LLM_AGENTS.get(agent_name)

    if config["sdk"] == "openai":
        return OpenAI(api_key=config["api_key"], base_url=config["base_url"]), config["model"], "openai"
    elif config["sdk"] == "zhipu":
        if not ZHIPU_AVAILABLE:
            raise RuntimeError("ZhipuAI SDK not available. Please install: pip install zai")
        client = ZhipuAiClient(api_key=config["api_key"], base_url=config["base_url"])
        return client, config["model"], "zhipu"
    else:
        raise ValueError(f"Unknown SDK: {config['sdk']}")

def call_llm_with_retry(agent_name, system_prompt, user_prompt, log_prompt=True):
    """LLM"""
    config = LLM_AGENTS.get(agent_name)

    log_message(f"\n[LLM Call] Agent: {agent_name}, Model: {config['model']}")

    # Log prompt only on first call
    if log_prompt:
        log_message(f"[System Prompt]\n{system_prompt}")
        log_message(f"[User Prompt]\n{user_prompt}")
    else:
        log_message("[System Prompt] 略")
        log_message("[User Prompt] 略")

    for attempt in range(MAX_RETRIES):
        try:
            client, model_name, sdk_type = get_client(agent_name)

            if sdk_type == "openai":
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.00,
                    stream=False
                )
                response_content = response.choices[0].message.content
            else:  # zhipu
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    thinking={"type": "disabled"},
                    max_tokens=65536,
                    temperature=0.00
                )
                response_content = response.choices[0].message.content

            log_message(f"[LLM Response from {agent_name}]\n{response_content}")
            return response_content

        except Exception as e:
            error_msg = f"\n[Warning] {agent_name} API调用失败，第 {attempt + 1}/{MAX_RETRIES} 次尝试。错误信息: {e}"
            print(error_msg)
            log_message(error_msg)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                log_message(f"\n[Error] {agent_name} 网络连接失败次数过多")
                return None

# =========================================
# 4. Quality evaluation functions
# =========================================

def agent_evaluate_quality_single(agent_name, text, language="zh", log_prompt=True):
    """agent"""
    if not text or not text.strip():
        log_message(f"[{agent_name} Quality Evaluation] Text is empty, returning 0")
        return 0

    if language == "en":
        system_prompt = (
            "You are a professional language evaluation expert skilled in quantitative quality analysis of texts based on rigorous evaluation dimensions.\n\n"
            "## Evaluation Dimensions and Standards\n\n"
            "### 1. Grammar Accuracy\n"
            "- 80-100: No grammatical errors, proper punctuation, correct tense, word order, and sentence structure.\n"
            "- 50-79: Minor flaws present (e.g., slight punctuation errors or individual word collocation issues), but do not affect overall readability.\n"
            "- 20-49: Numerous grammatical errors (e.g., subject-verb disagreement, tense confusion, missing components), hindering reading experience.\n"
            "- 0-19: Chaotic structure, cannot be recognized as valid language expression.\n\n"
            "### 2. Semantic Coherence\n"
            "- 80-100: Expression is smooth and fluent, context connections are natural.\n"
            "- 50-79: Generally fluent, but some passages have awkward transitions or unnecessary redundant vocabulary.\n"
            "- 20-49: Meaning is unclear, obvious logical gaps exist, readers need repeated reading to understand.\n"
            "- 0-19: Fragmented meaning, cannot form a complete semantic closure.\n\n"
            "Output only JSON format results without any additional explanation."
        )
        user_prompt = (
            f"Please evaluate the following text, providing separate scores for grammar accuracy and semantic coherence (0-100), then calculate the combined score (0.6×Grammar+0.4×Semantic):\n\n"
            f"Text to evaluate: {text}\n\n"
            f"Please output results in the following JSON format:\n"
            f'{{"Grammar Accuracy": <0-100>, "Semantic Coherence": <0-100>, "Combined Score": <0-100>, "Feedback": "<brief explanation>"}}'
        )
        json_key_combined = "Combined Score"
    else:  # language == "zh"
        system_prompt = (
            "你是一位专业的语言评估专家，擅长根据严密的评分维度对文本进行质量定量分析。\n\n"
            "  # # \n\n"
            "  # ## 1. (Grammar Accuracy)\n"
            "- 80-100: 无语法错误，标点规范，时态、词序及成分无错误。\n"
            "- 50-79: 存在少量轻微瑕疵（如轻微的标点误用或个别词汇搭配不当），不影响整体阅读。\n"
            "- 20-49: 语法错误较多（如主谓不一致、时态混乱、成分残缺），阅读体验受阻。\n"
            "- 0-19: 结构混乱，无法识别为有效的语言表达。\n\n"
            "  # ## 2. (Semantic Coherence)\n"
            "- 80-100: 表达丝滑顺畅，上下文衔接自然。\n"
            "- 50-79: 基本流畅，但部分段落衔接略显生硬，或使用了不必要的冗余词汇。\n"
            "- 20-49: 意思表达模糊，存在明显的逻辑断层，读者需反复阅读才能理解。\n"
            "- 0-19: 意义碎片化，无法构成完整的语意闭环。\n\n"
            "请仅输出 JSON 格式的结果，不要包含任何多余的解释。"
        )
        user_prompt = (
            f"请对以下文本进行评估，分别给出语法正确性和语义连贯性的评分（0-100），然后计算综合评分（0.6×语法+0.4×语义）：\n\n"
            f"待评估文本：{text}\n\n"
            f"请按以下 JSON 格式输出结果：\n"
            f'{{"语法正确性": <0-100>, "语义连贯性": <0-100>, "综合评分": <0-100>, "综合点评": "<简要说明>"}}'
        )
        json_key_combined = "综合评分"

    response = call_llm_with_retry(agent_name, system_prompt, user_prompt, log_prompt=log_prompt)

    if response is None:
        return 0

    try:
        result = json.loads(response)
        combined_score = result.get(json_key_combined, 0)
        final_score = max(0, min(100, int(combined_score)))
        log_message(f"[{agent_name} Quality Score] Final score: {final_score}")
        return final_score
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        log_message(f"[{agent_name} Quality Score] JSON parsing failed: {e}, attempting regex extraction")
        numbers = re.findall(r'\d+', response)
        if numbers:
            final_score = max(0, min(100, int(numbers[0])))
            log_message(f"[{agent_name} Quality Score] Regex extracted score: {final_score}")
            return final_score
        log_message(f"[{agent_name} Quality Score] No score found, returning 0")
        return 0

def agent_evaluate_quality(text, language="zh", log_first_prompt=True):
    """3 agent 3"""
    if not text or not text.strip():
        return {"agent01": 0, "agent02": 0, "agent03": 0, "average": 0}

    log_message(f"\n[Quality Evaluation - 3 Agents in Parallel]")

    scores = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for idx, agent_name in enumerate(["agent01", "agent02", "agent03"]):
            # Log prompt only for first agent on first call
            log_prompt = log_first_prompt and (idx == 0)
            futures[executor.submit(agent_evaluate_quality_single, agent_name, text, language, log_prompt)] = agent_name

        for future in as_completed(futures):
            agent_name = futures[future]
            try:
                score = future.result()
                scores[agent_name] = score
            except Exception as e:
                log_message(f"[Error] {agent_name} evaluation failed: {e}")
                scores[agent_name] = 0

    average_score = sum(scores.values()) / 3.0
    scores["average"] = round(average_score, 2)

    log_message(f"[Quality Scores] agent01: {scores['agent01']}, agent02: {scores['agent02']}, agent03: {scores['agent03']}, average: {scores['average']}")

    return scores

# =========================================
# 5. Factuality evaluation functions
# =========================================

def agent_evaluate_factuality_single(agent_name, text, quality_score, language="zh", log_prompt=True):
    """agent"""
    if quality_score < 50:
        log_message(f"[{agent_name} Factuality Evaluation] Quality score ({quality_score}) < 50, returning 0")
        return 0

    if not text or not text.strip():
        log_message(f"[{agent_name} Factuality Evaluation] Text is empty, returning 0")
        return 0

    if language == "en":
        system_prompt = (
            "You are a factuality verification expert. Please judge whether the following text contains obvious factual or logical errors.\n\n"
            "## Scoring Standards\n"
            "1 (PASS): The text has no factual or logical issues; or it is a non-factual statement (e.g., emotional expression, casual conversation, personal intention), treated as error-free by default.\n"
            "0 (FAIL): The text contains obvious factual errors.\n\n"
            "Please ignore grammatical issues and focus only on factual logic.\n"
            "Output only JSON format results without any additional explanation."
        )
        user_prompt = (
            f"Please evaluate the factuality of the following text:\n\n"
            f"Text to evaluate: {text}\n\n"
            f"Please output results in the following JSON format:\n"
            f'{{"Factuality Score": <0 or 1>, "Comment": "<brief explanation of deductions>"}}'
        )
        json_key_score = "Factuality Score"
    else:  # language == "zh"
        system_prompt = (
            "你是一个事实性核查专家。请判断以下文本是否存在明显的事实性、原理性错误。\n\n"
            "  # # \n"
            "1分 (PASS)：文本描述没有事实、原理问题；或者是非事实性陈述（如情感表达、日常对话、主观意愿），默认视为无错。\n"
            "0分 (FAIL)：文本描述存在明显的事实错误。\n\n"
            "请忽略语法问题，只关注事实逻辑。\n"
            "请仅输出 JSON 格式的结果，不要包含任何多余的解释。"
        )
        user_prompt = (
            f"请对以下文本进行事实性评估：\n\n"
            f"待评估文本：{text}\n\n"
            f"请按以下 JSON 格式输出结果：\n"
            f'{{"事实性评分": <0或1>, "点评": "<简要说明扣分点>"}}'
        )
        json_key_score = "事实性评分"

    response = call_llm_with_retry(agent_name, system_prompt, user_prompt, log_prompt=log_prompt)

    if response is None:
        return 0

    try:
        result = json.loads(response)
        score = result.get(json_key_score, 0)
        final_score = 100 if int(score) >= 1 else 0
        log_message(f"[{agent_name} Factuality Score] Final score: {final_score}")
        return final_score
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        log_message(f"[{agent_name} Factuality Score] JSON parsing failed: {e}, attempting regex extraction")
        numbers = re.findall(r'\d+', response)
        if numbers:
            final_score = 100 if int(numbers[0]) >= 1 else 0
            log_message(f"[{agent_name} Factuality Score] Regex extracted score: {final_score}")
            return final_score
        log_message(f"[{agent_name} Factuality Score] No score found, returning 0")
        return 0

def agent_evaluate_factuality(text, quality_score, language="zh", task_type="factuality", log_first_prompt=True):
    """3 agent 2"""
    # < 50,  0
    if quality_score < 50:
        log_message(f"[Factuality Evaluation] Quality score ({quality_score}) < 50, returning 0")
        return {"agent01": 0, "agent02": 0, "agent03": 0, "final": 0}

    if not text or not text.strip():
        log_message("[Factuality Evaluation] Text is empty, returning 0")
        return {"agent01": 0, "agent02": 0, "agent03": 0, "final": 0}

    log_message(f"\n[Factuality Evaluation - 3 Agents in Parallel]")

    scores = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for idx, agent_name in enumerate(["agent01", "agent02", "agent03"]):
            # Log prompt only for first agent on first call
            log_prompt = log_first_prompt and (idx == 0)
            futures[executor.submit(agent_evaluate_factuality_single, agent_name, text, quality_score, language, log_prompt)] = agent_name

        for future in as_completed(futures):
            agent_name = futures[future]
            try:
                score = future.result()
                scores[agent_name] = score
            except Exception as e:
                log_message(f"[Error] {agent_name} evaluation failed: {e}")
                scores[agent_name] = 0

    # 2
    score_values = [scores["agent01"], scores["agent02"], scores["agent03"]]
    score_counts = {}
    for score in score_values:
        score_counts[score] = score_counts.get(score, 0) + 1

    # Find score appearing at least twice
    final_score = 0
    for score, count in score_counts.items():
        if count >= 2:
            final_score = score
            break

    scores["final"] = final_score
    log_message(f"[Factuality Scores] agent01: {scores['agent01']}, agent02: {scores['agent02']}, agent03: {scores['agent03']}, final: {scores['final']}")

    return scores

# =========================================
# 5. Main processing logic
# =========================================

def main():
    global log_file

    # Initialize log file
    init_log_file()
    log_message(f"Processing started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Check input file exists
    if not os.path.exists(INPUT_FILE_PATH):
        error_msg = f"错误：输入文件不存在 -> {INPUT_FILE_PATH}"
        print(error_msg)
        log_message(error_msg)
        close_log_file()
        return

    # 2. Check checkpoint for resume
    processed_count = 0
    if os.path.exists(OUTPUT_FILE_PATH):
        with open(OUTPUT_FILE_PATH, 'r', encoding='utf-8') as f_out:
            for _ in f_out:
                processed_count += 1
        msg = f"检测到输出文件，已处理 {processed_count} 条数据，将从第 {processed_count + 1} 条开始继续。"
        print(msg)
        log_message(msg)
    else:
        msg = "未检测到输出文件，将开始新的处理任务。"
        print(msg)
        log_message(msg)

    # 3. Read input and process
    with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
        total_lines = len(lines)

    # Output file
    with open(OUTPUT_FILE_PATH, 'a+', encoding='utf-8') as f_out:

        for i, line in enumerate(lines):
            # Skip processed lines
            if i < processed_count:
                continue

            # Log separator and line number
            log_separator(i + 1)

            print(f"现在正在处理第 {i + 1}/{total_lines} 条 ...")
            log_message(f"现在正在处理第 {i + 1}/{total_lines} 条")

            try:
                data = json.loads(line.strip())

                # Extract fields
                original_text = data.get("original_sentence", "")
                word_list = data.get("word_list", [])
                response_text = data.get("response", "")
                response_list = data.get("response_list", [])
                lang = data.get("language", "zh")

                log_message(f"[Extracted Fields] Language: {lang}")

                # =========================================
                # =========================================
                log_message(f"\n[Step 1] Computing Compliance F1...")
                compliance_f1 = compute_compliance_f1(word_list, response_list)
                log_message(f"Compliance F1: {compliance_f1}")

                # =========================================
                # =========================================
                log_message(f"\n[Step 2] Quality Evaluation...")
                # 1) 3 Agent
                log_message("[Step 2.1] 3 Agent Quality Scores")
                quality_scores = agent_evaluate_quality(response_text, lang, log_first_prompt=True)

                # 2) N-gram F1 (Reference)
                log_message("[Step 2.2] N-gram Metrics")
                f1_2gram = compute_ngram_f1(response_text, original_text, lang, 2)
                f1_3gram = compute_ngram_f1(response_text, original_text, lang, 3)
                f1_4gram = compute_ngram_f1(response_text, original_text, lang, 4)
                avg_ngram = (f1_2gram + f1_3gram + f1_4gram) / 3.0
                log_message(f"2-gram F1: {f1_2gram}, 3-gram F1: {f1_3gram}, 4-gram F1: {f1_4gram}, Avg: {avg_ngram}")

                # 3) ROUGE-L F1 (Reference)
                log_message("[Step 2.3] ROUGE-L Metric")
                rouge_l_f1 = compute_rouge_l_f1(response_text, original_text, lang)
                log_message(f"ROUGE-L F1: {rouge_l_f1}")

                # =========================================
                # =========================================
                log_message(f"\n[Step 3] Factuality Evaluation...")
                factuality_scores = agent_evaluate_factuality(response_text, quality_scores["average"], lang, "factuality", log_first_prompt=True)

                # =========================================
                # =========================================
                log_message(f"\n[Step 4] Computing Overall Scores...")
                factuality_coeff = 1.0 if factuality_scores["final"] == 100 else 0.0

                # Agent
                overall_score_agent = compliance_f1 * float(quality_scores["average"]) * factuality_coeff

                # N-gram
                overall_score_ngram = compliance_f1 * avg_ngram * 100 * factuality_coeff

                # ROUGE-L
                overall_score_rouge = compliance_f1 * rouge_l_f1 * 100 * factuality_coeff
                log_message(f"Overall Score (Agent): {overall_score_agent}")
                log_message(f"Overall Score (N-gram): {overall_score_ngram}")
                log_message(f"Overall Score (ROUGE): {overall_score_rouge}")

                # ##########################################################################
                # # (5) & (6) - COMMENTED OUT
                # ##########################################################################
                # # ( Agent )
                # ##########################################################################

                # =========================================
                # =========================================
                log_message(f"\n[Step 6] Building Evaluation Result...")
                evaluation_result = {
                    "compliance_f1": round(compliance_f1, 4),
                    "quality_agent_scores": {
                        "agent01": quality_scores["agent01"],
                        "agent02": quality_scores["agent02"],
                        "agent03": quality_scores["agent03"],
                        "average": quality_scores["average"]
                    },
                    "ngram_metrics": {
                        "2gram_f1": round(f1_2gram, 4),
                        "3gram_f1": round(f1_3gram, 4),
                        "4gram_f1": round(f1_4gram, 4),
                        "avg_ngram_f1": round(avg_ngram, 4)
                    },
                    "rouge_l_f1": round(rouge_l_f1, 4),
                    "factuality_scores": {
                        "agent01": factuality_scores["agent01"],
                        "agent02": factuality_scores["agent02"],
                        "agent03": factuality_scores["agent03"],
                        "final": factuality_scores["final"]
                    },
                    "overall_score_agent": round(overall_score_agent, 2),
                    "overall_score_ngram": round(overall_score_ngram, 2),
                    "overall_score_rouge": round(overall_score_rouge, 2),
                    # ##########################################################################
                    # # original_eval section - COMMENTED OUT
                    # ##########################################################################
                    # },
                    # },
                    # }
                    # ##########################################################################
                }

                log_message(f"[Evaluation Result]\n{json.dumps(evaluation_result, ensure_ascii=False, indent=2)}")

                data["evaluation"] = evaluation_result

                # Write to file
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                f_out.flush()

                log_message(f"[Step 7] Data written to output file successfully")

            except json.JSONDecodeError as e:
                error_msg = f"[Error] 第 {i + 1} 行 JSON 解析失败: {e}"
                print(error_msg)
                log_message(error_msg)
            except Exception as e:
                error_msg = f"[Error] 处理第 {i + 1} 行时发生未知错误: {e}"
                print(error_msg)
                log_message(error_msg)
                continue

    finish_msg = "\n所有任务处理完成！"
    print(finish_msg)
    log_message(finish_msg)
    log_message(f"结果已保存至: {OUTPUT_FILE_PATH}")
    log_message(f"Processing finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    close_log_file()

if __name__ == "__main__":
    main()
import json
import os
import time
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# =========================================
# 1. Configuration
# =========================================

# --- API ---
API_CONFIG = {
    "api_key": "YOUR_API_KEY",    # Replace with your API key
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",    # Replace with the corresponding Base URL
    "model_name": "qwen3-max"    # Model name
}

# --- Prompt template setup (multilingual) ---
PROMPT_TEMPLATES = {
    "zh": (
        "你是一个语言学专家。请利用给定的【字/词列表】重组为一个通顺的句子或段落。\n\n"
        "【核心要求】\n"
        "1. 句子语法正确、语言流畅、逻辑连贯、事实正确。\n"
        "2. 必须使用列表中的每一个元素，且每个元素使用的次数必须与列表中出现的次数完全一致。\n"
        "3. 严禁添加列表中不存在的任何汉字（包括“的”、“了”等虚词）。\n"
        "4. 允许自由添加标点符号以辅助断句。\n"
        "5. 直接输出重组后的文本，不要包含任何解释性语言。\n\n"
        "输入列表: {word_list_str}\n"
        "输出结果:"
    ),
    "en": (
        "You are a linguistics expert. Please reconstruct a coherent sentence or paragraph using the given [word list].\n\n"
        "[Core Requirements]\n"
        "1. The sentence must be grammatically correct, fluent, logically coherent, and factually accurate.\n"
        "2. You must use every element in the list, and the frequency of each element must match exactly.\n"
        "3. Do NOT add any words that do not exist in the list (including function words like 'the', 'is').\n"
        "4. Punctuation marks can be added freely to aid structure.\n"
        "5. Output the reconstructed text directly without any explanatory text.\n\n"
        "Input List: {word_list_str}\n"
        "Output:"
    )
}

# --- Runtime Parameters ---
MAX_WORKERS = 1    # (low concurrency recommended)
INPUT_PATH = r"D:\01project\wordorder\buchong_discrete.jsonl"
OUTPUT_PATH = r"D:\01project\wordorder\buchong_discrete_qwen.jsonl"
LOG_DIR = r"D:\01project\wordorder"
LOG_FILE = os.path.join(LOG_DIR, "buchong_discrete_qwen_log.txt")

# --- Retry Strategy ---
MAX_RETRIES = 5
RETRY_DELAY = 20    # sec

# =========================================
# 2. Setup
# =========================================

# Ensure output and log directories exist
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# File logging
file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console logging
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# File write lock
file_write_lock = threading.Lock()

# Initialize OpenAI client
client = OpenAI(api_key=API_CONFIG["api_key"], base_url=API_CONFIG["base_url"])

# =========================================
# 3. Core Functions (Core Functions)
# =========================================

def construct_prompt(word_list, language):
    """Prompt"""
    # Convert list to string
    word_list_str = str(word_list)

    # Get template by language, default to English 'en' (or 'zh' )
    template = PROMPT_TEMPLATES.get(language, PROMPT_TEMPLATES['en'])

    # Fill template
    prompt = template.format(word_list_str=word_list_str)
    return prompt

def call_llm_with_retry(prompt, model_name, line_idx):
    """LLM API"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,    # low temp for determinism
                stream=False
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.warning(f"[Line {line_idx}] API Error (Attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                logger.info(f"[Line {line_idx}] Waiting {RETRY_DELAY} seconds before retrying...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"[Line {line_idx}] Max retries reached. Stopping execution to prevent data skip.")
                raise RuntimeError(f"Failed to process line {line_idx} after {MAX_RETRIES} attempts.")

def process_single_item(item, line_idx):

    try:
        word_list = item.get("word_list", [])
        language = item.get("language", "zh")    # get languagedefault: Chinese

        # Construct prompt
        prompt = construct_prompt(word_list, language)

        # Log start
        logger.info(f"[Line {line_idx}] Processing ({language})... Input len: {len(word_list)}")

        # Call model
        response_text = call_llm_with_retry(prompt, API_CONFIG["model_name"], line_idx)

        # Build result
        result_item = item.copy()
        result_item["test_model"] = API_CONFIG["model_name"]
        result_item["response"] = response_text

        # Print partial result for monitoring
        display_res = response_text.replace('\n', ' ')
        logger.info(f"[Line {line_idx}] Success. Response: {display_res[:30]}...")

        return result_item

    except RuntimeError as re:
        raise re
    except Exception as e:
        logger.error(f"[Line {line_idx}] Unexpected error: {e}")
        raise e

# =========================================
# 4. Main Loop (Main Loop)
# =========================================

def main():
    logger.info(f"Task Started. Model: {API_CONFIG['model_name']}, Threads: {MAX_WORKERS}")

    # 1. Resume from checkpoint
    completed_lines = 0
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
            for _ in f:
                completed_lines += 1
        logger.info(
            f"Found existing output file with {completed_lines} lines. Resuming from Line {completed_lines + 1}...")
    else:
        logger.info("No existing output file found. Starting from beginning.")

    # 2. Read input data
    try:
        with open(INPUT_PATH, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
    except FileNotFoundError:
        logger.error(f"Input file not found: {INPUT_PATH}")
        return

    total_data = len(all_lines)
    logger.info(f"Total data loaded: {total_data} lines.")

    # 3. Prepare tasks with original line numbers
    tasks_data = []
    for idx, line in enumerate(all_lines):
        if idx < completed_lines:
            continue
        try:
            data = json.loads(line.strip())
            tasks_data.append((data, idx + 1))
        except json.JSONDecodeError:
            logger.warning(f"Skipping invalid JSON at line {idx + 1}")

    if not tasks_data:
        logger.info("All data has already been processed.")
        return

    # 4. Process and write results (multi/single-threaded)
    stop_event = threading.Event()

    def safe_process(task_info):
        if stop_event.is_set():
            return None
        data, original_idx = task_info
        try:
            return process_single_item(data, original_idx)
        except RuntimeError:
            stop_event.set()
            raise
        except Exception:
            return None

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_data = {executor.submit(safe_process, task): task for task in tasks_data}

            for future in as_completed(future_to_data):
                try:
                    result = future.result()

                    if result is None:
                        if stop_event.is_set():
                            logger.critical("Execution halting due to previous errors.")
                            executor.shutdown(wait=False, cancel_futures=True)
                            sys.exit(1)
                        continue

                    # Write result immediately
                    with file_write_lock:
                        with open(OUTPUT_PATH, 'a', encoding='utf-8') as f_out:
                            f_out.write(json.dumps(result, ensure_ascii=False) + '\n')

                except RuntimeError as e:
                    logger.critical(f"Critical Failure: {e}")
                    stop_event.set()
                    executor.shutdown(wait=False, cancel_futures=True)
                    sys.exit(1)
                except Exception as e:
                    logger.error(f"Unhandled exception in thread: {e}")

    except KeyboardInterrupt:
        logger.info("User interrupted processing.")
        sys.exit(0)

    logger.info(f"Processing complete. Results saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
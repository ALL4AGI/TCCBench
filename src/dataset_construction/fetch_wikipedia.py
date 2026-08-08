"""JSONL python-wikipedia, OpenCC data_raw/ │── outlines/ │ ├── physics_outline_zh.json │ ├── physics_outline_en.json │ └── physics_outline_custom.json │ └── fetch_wiki.py #"""

import wikipedia
import re
import time
import json
import os
import opencc

"""28 31 35 39"""

# --- 1. and ---

# and;# ("fr"); # ("es") ;  # ("de") ; # ("ru") ;  # ("ko")
TARGET_LANG = "en"
# TARGET_LANG , CONVERSION_PROFILE None
# TARGET_LANG , (: tw2s.json)
CONVERSION_PROFILE = None #'tw2s.json'

OUTPUT_DIR = r"D:\01project\wordorder\dataraw\wikipedia"
OUTPUT_FILENAME = f"chemistry_{TARGET_LANG}4.jsonl"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

# outline
OUTLINE_PATH = r"D:\01project\wordorder\dataraw\outlines\chemistry_en4.json"

# wikipedia
wikipedia.set_lang(TARGET_LANG)

# ---- outline ----

def load_outline(outline_path):
    with open(outline_path, "r", encoding="utf-8") as f:
        return json.load(f)

PHYSICS_OUTLINE = load_outline(OUTLINE_PATH)

# --- 2. and ---

def clean_text(text):

    # ,  [1], [note 2]
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[note \d+\]', '', text)
    # ,  (: xxx)
    text = re.sub(r'\(参见[^\)]*?\)', '', text)
    # and
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def fetch_wiki_content():
    """JSONL"""
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_articles = 0

    # URL,
    scraped_urls = set()

    # OpenCC ---
    CC = None
    needs_conversion = False

    # (TARGET_LANG zh or zh-)
    if TARGET_LANG == "zh" or TARGET_LANG.startswith("zh-"):
        if CONVERSION_PROFILE:
            try:
                CC = opencc.OpenCC(CONVERSION_PROFILE)
                needs_conversion = True
                print(f"检测到中文目标语言 ({TARGET_LANG}), 已启用 OpenCC ({CONVERSION_PROFILE}) 转换.")
            except Exception as e:
                print(f"警告: OpenCC 初始化失败, 内容将以原始繁体/混合形式保存.错误: {e}")
        else:
            print("警告: TARGET_LANG 为中文, 但 CONVERSION_PROFILE 未定义, 内容将以原始形式保存.")

    print(f"开始抓取维基百科中文内容.目标路径: {OUTPUT_PATH}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as outfile:

        discipline = PHYSICS_OUTLINE["discipline"]

        # Iterate L1, L2, L3 levels
        for chapter in PHYSICS_OUTLINE["chapters"]:
            l1_chapter = chapter["l1_chapter"]

            for section in chapter["sections"]:
                l2_section = section["l2_section"]

                for topic in section["topics"]:
                    l3_topic = topic["l3_topic"]
                    keywords = topic["keywords"]

                    print(f"\n--- 正在处理 L3 主题: [{l3_topic}] ---")

                    # L4
                    for keyword in keywords:
                        try:
                            # 1. Top 1
                            search_results = wikipedia.search(keyword, results=1)

                            if not search_results:
                                print(f"  [跳过] 关键词 '{keyword}' 无搜索结果.")
                                continue

                            title = search_results[0]

                            # --- 2. ---
                            # and
                            # continue

                            # 3.
                            page = wikipedia.page(title=title, auto_suggest=False, redirect=True)

                            # --- 4. URL ---
                            if page.url in scraped_urls:
                                print(f"  [跳过] 词条 '{title}' URL已存在, 避免重复抓取.")
                                continue

                            scraped_urls.add(page.url)   # URL

                            # 4.
                            raw_content = clean_text(page.content)

                            if needs_conversion:
                                final_content = CC.convert(raw_content)
                            else:
                                final_content = raw_content

                            # 5. JSON
                            data_record = {
                                "discipline": discipline,
                                "l1_chapter": l1_chapter,
                                "l2_section": l2_section,
                                "l3_topic": l3_topic,
                                "l4_keyword": keyword,
                                "source_url": page.url,
                                "language": TARGET_LANG,
                                "raw_content": final_content
                            }

                            # 6. JSONL
                            outfile.write(json.dumps(data_record, ensure_ascii=False) + '\n')
                            total_articles += 1

                            print(f"  [成功] 写入文章: '{title}' ({page.url})")

                        except wikipedia.exceptions.DisambiguationError as e:
                            # skipping
                            print(f"  [跳过] 关键词 '{keyword}' 存在歧义: {e.options}")
                        except wikipedia.exceptions.PageError:
                            # Page not found
                            print(f"  [跳过] 关键词 '{keyword}' 页面未找到.")
                        except Exception as e:
                            print(f"  [错误] 关键词 '{keyword}' 发生未知错误: {e}")

                        time.sleep(1)

    print(f"\n--- 抓取完成 ---")
    print(f"总计抓取并保存文章数量: {total_articles}")
    print(f"数据已保存至: {OUTPUT_PATH}")

if __name__ == "__main__":
    fetch_wiki_content()
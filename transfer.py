import json
import os
import datasets
from datasets import Dataset
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tqdm import tqdm
from data_reader import read_general
import re

def extract_answer(s):
    # 使用正则表达式找到所有 \boxed{ 的位置
    boxed_indices = [m.start() for m in re.finditer(r'\\boxed\{', s)]
    
    results = []
    
    for start in boxed_indices:
        stack = []
        content_start = start + len(r'\boxed{')
        content_end = None
        
        for i in range(content_start, len(s)):
            if s[i] == '{':
                stack.append(i)
            elif s[i] == '}':
                if not stack:
                    # 找到了与 \boxed{ 匹配的 }
                    content_end = i
                    break
                else:
                    stack.pop()
        
        if content_end is not None:
            # 提取 \boxed{} 中的内容
            content = s[content_start:content_end]
            results.append(content)
    
    return results

def process_item(item, image_dir_path):
    try:
        question = item['question']
        thinking = item['processed_CoT_reasoning']
        
        # 处理 \boxed{} 的情况
        if r"\boxed{" not in thinking:
            if r"\oxed{" in thinking:
                thinking = thinking.replace("\oxed{", r"\boxed{")
            elif "oxed{" in thinking:
                thinking = thinking.replace("oxed{", r"\boxed{")
        
        # 提取 \boxed{} 中的内容
        extract_gt = extract_answer(thinking)
        if len(extract_gt) != 1:
            # print(thinking)
            return None
        elif " " in extract_gt[0]:
            return None
        # if len(extract_gt) == 0:
        #     # print(thinking)
        #     return None
        
        # 读取图像并转换为 RGB 格式
        image_path = item['image_path']
        image = read_general(image_path)
        image = Image.open(image).convert('RGB')
        width, height = image.size

        extract_gt_0 = extract_gt[0]
        if "pi" in extract_gt_0:
            if "\pi" not in extract_gt_0:
                extract_gt_0 = extract_gt_0.replace("pi", "\pi")
        # if extract_gt_0.startswith("\\"):
        #     extract_gt_0 = extract_gt_0
        # else:
        #     extract_gt_0 = f"\{extract_gt_0}"

        cot_reasoning = r"\boxed{" + f"{str(extract_gt_0)}" + "}"
        
        # 构建新的 item 字典
        new_item = {
            'question': question,
            'image': image,
            'image_width': width,
            'image_height': height,
            'cot_reasoning': cot_reasoning
            # 'extract_gt': extract_gt
        }
        
        return new_item
    
    except Exception as e:
        # 打印错误信息并返回 None，表示该 item 处理失败
        print(f"Error processing item: {item}. Error: {e}")
        return None

def save_parquet_file(data_list, file_index, output_dir):
    dataset_hf = Dataset.from_list(data_list)
    output_path = os.path.join(output_dir, f"train_{file_index}.parquet")
    dataset_hf.to_parquet(output_path)

def chunk_exists(chunk_index, output_dir):
    output_path = os.path.join(output_dir, f"train_{chunk_index}.parquet")
    return os.path.exists(output_path)

# 数据文件路径
data_file_path = "/mnt/petrelfs/zhaoshitian/data/MAVIS-Geometry/processed_masiv_function.jsonl"
f = open(data_file_path, "r", encoding="utf-8")
data_file_list = [json.loads(line) for line in f]

# 图像目录路径
image_dir_path = "/mnt/petrelfs/zhaoshitian/data/AnyWord-3M/imgs"

# 输出目录
output_dir = "/mnt/petrelfs/zhaoshitian/data/MAVIS-Function/mvr-mavis-function"

# 定义每个 chunk 的大小
chunk_size = len(data_file_list) // 16

# 按 chunk 处理并保存数据
for chunk_index in range(16):
    start_index = chunk_index * chunk_size
    end_index = start_index + chunk_size
    if chunk_index == 15:  # 最后一个 chunk 可能更大
        end_index = len(data_file_list)
    
    # 检查 chunk 是否已经处理过
    if chunk_exists(chunk_index + 1, output_dir):
        print(f"Chunk {chunk_index + 1} already processed. Skipping...")
        continue
    
    chunk_data = data_file_list[start_index:end_index]
    
    # 使用线程池并行处理 chunk 中的数据，并显示进度条
    with ThreadPoolExecutor() as executor:
        processed_chunk_data = list(tqdm(executor.map(partial(process_item, image_dir_path=image_dir_path), chunk_data), total=len(chunk_data), desc=f"Processing chunk {chunk_index + 1}"))
    
    # 过滤掉处理失败的 item
    processed_chunk_data = [item for item in processed_chunk_data if item is not None]
    
    # 将处理后的 chunk 保存为 parquet 文件
    save_parquet_file(processed_chunk_data, chunk_index + 1, output_dir)
    print(f"Saved chunk {chunk_index + 1} as parquet file.")

print("All chunks processed and saved.")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量精简“多记录”ECG JSON 文件：
- 支持顶层 {"RECORDS": [...]} 或 顶层是列表 [...]
- 每条记录只保留: diaglabel(重命名为 labels), sample_rate, user_id, ecg_data
- 若 ecg_data 为 "{a,b,c}" 字符串，自动解析为 [a,b,c] 浮点列表
- 结果写入输出目录，保持相对路径结构
"""

from pathlib import Path
import json
import argparse
import re

# 需要保留且（可能）重命名的字段
KEY_MAP = {
    "diaglabel": "labels",
    "sample_rate": "sample_rate",
    "user_id": "user_id",
    "ecg_data": "ecgdata",
}

def parse_ecg_data(value):
    """
    将形如 "{0.1,0.2,...}" 的字符串解析为 float 列表。
    若不是该格式（例如已是列表/数组），原样返回。
    """
    if isinstance(value, str):
        s = value.strip()
        # 简单判断是否为 { ... } 包裹的逗号分隔数字
        if s.startswith("{") and s.endswith("}"):
            s = s[1:-1].strip()
            if not s:
                return []
            # 按逗号切分并转为 float；允许有多余空格
            parts = [p.strip() for p in s.split(",")]
            out = []
            for p in parts:
                # 允许科学计数、正负号
                # 删除可能的多余空白字符
                if p:
                    try:
                        out.append(float(p))
                    except ValueError:
                        # 如果遇到非数值，保底：忽略或抛错均可
                        # 这里选择忽略该点
                        continue
            return out
    return value  # 已是列表或其它类型，直接返回

def transform_record(rec: dict) -> dict:
    """对单条记录做字段筛选与改名。"""
    new_rec = {}
    for src_key, dst_key in KEY_MAP.items():
        if src_key in rec:
            val = rec[src_key]
            if dst_key == "ecg_data":
                val = parse_ecg_data(val)
            new_rec[dst_key] = val
    return new_rec

def transform_payload(data):
    """
    处理不同顶层结构：
    - {"RECORDS": [...]} -> 同结构返回，但每个元素精简
    - [...] -> 同样返回列表
    - 其它（单条 dict）-> 返回单条精简 dict
    """
    if isinstance(data, dict) and "RECORDS" in data and isinstance(data["RECORDS"], list):
        return {"RECORDS": [transform_record(x) for x in data["RECORDS"] if isinstance(x, dict)]}
    elif isinstance(data, list):
        return [transform_record(x) for x in data if isinstance(x, dict)]
    elif isinstance(data, dict):
        return transform_record(data)
    else:
        # 不支持的结构：原样返回，或根据需要抛错
        return data

def process_one_file(src_path: Path, dst_path: Path):
    try:
        with src_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[SKIP] 读取失败: {src_path} -> {e}")
        return

    result = transform_payload(data)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with dst_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[OK] {src_path} -> {dst_path}")
    except Exception as e:
        print(f"[FAIL] 写入失败: {dst_path} -> {e}")

def main():
    ap = argparse.ArgumentParser(description="精简多记录 ECG JSON（重命名 diaglabel->labels）")
    ap.add_argument("input_dir", help="输入目录（含原始 JSON）")
    ap.add_argument("output_dir", help="输出目录（写入精简后的 JSON）")
    args = ap.parse_args()

    in_root = Path(args.input_dir).expanduser().resolve()
    out_root = Path(args.output_dir).expanduser().resolve()

    if not in_root.exists() or not in_root.is_dir():
        raise SystemExit(f"输入目录不存在或不是文件夹: {in_root}")
    if out_root == in_root:
        raise SystemExit("输出目录不能与输入目录相同！请指定新的目录。")

    for src in in_root.rglob("*.json"):
        rel = src.relative_to(in_root)
        dst = out_root / rel
        process_one_file(src, dst)

if __name__ == "__main__":
    main()

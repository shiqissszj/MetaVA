#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量精简 ECG JSON 文件：
- 递归扫描输入目录中的所有 .json
- 仅保留: labels, userid, sample_rate, ecgdata
- 将结果保存到输出目录，保持原有的相对路径结构
"""

from pathlib import Path
import json
import argparse

KEEP_KEYS = ["labels", "userid", "sample_rate", "ecgdata"]

def filter_record(obj):
    """过滤单个记录：只保留指定键；其它丢弃。"""
    if isinstance(obj, dict):
        return {k: obj[k] for k in KEEP_KEYS if k in obj}
    elif isinstance(obj, list):
        # 文件可能是记录列表：逐条过滤
        return [filter_record(x) if isinstance(x, (dict, list)) else x for x in obj]
    else:
        # 既不是 dict 也不是 list，原样返回
        return obj

def process_one_file(src_path: Path, dst_path: Path):
    """读取一个 JSON，过滤后写入到目标路径。"""
    try:
        with src_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[SKIP] 读取失败: {src_path} -> {e}")
        return

    filtered = filter_record(data)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with dst_path.open("w", encoding="utf-8") as f:
            # 使用 indent=2 便于阅读；确保 UTF-8
            json.dump(filtered, f, ensure_ascii=False, indent=2)
        print(f"[OK] {src_path} -> {dst_path}")
    except Exception as e:
        print(f"[FAIL] 写入失败: {dst_path} -> {e}")

def main():
    ap = argparse.ArgumentParser(description="仅保留 JSON 中的 labels/userid/sample_rate/ecgdata")
    ap.add_argument("input_dir", help="输入目录（原始 JSON 所在根目录）")
    ap.add_argument("output_dir", help="输出目录（写入精简后的 JSON）")
    args = ap.parse_args()

    in_root = Path(args.input_dir).expanduser().resolve()
    out_root = Path(args.output_dir).expanduser().resolve()

    if not in_root.exists() or not in_root.is_dir():
        raise SystemExit(f"输入目录不存在或不是文件夹: {in_root}")

    # 避免把结果写回到输入目录
    if out_root == in_root:
        raise SystemExit("输出目录不能与输入目录相同！请指定一个新的目录。")

    # 递归遍历所有 .json
    for src in in_root.rglob("*.json"):
        # 保持相对路径结构
        rel = src.relative_to(in_root)
        dst = out_root / rel
        process_one_file(src, dst)

if __name__ == "__main__":
    main()

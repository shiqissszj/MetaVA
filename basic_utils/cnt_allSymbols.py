import os
import wfdb
import numpy as np
from collections import Counter


def parse_annotation_file(file_path):
    """
    解析单个.atr注释文件，返回符号和辅助注释。
    """
    # 去除扩展名
    base_name = os.path.splitext(file_path)[0]

    # 读取注释文件
    annotation = wfdb.rdann(base_name, 'atr')

    # 提取符号和辅助注释
    symbols = annotation.symbol
    aux_notes = annotation.aux_note

    # 清理辅助信息中的'\x00'字符
    aux_notes_cleaned = [note.strip('\x00') for note in aux_notes]

    return symbols, aux_notes_cleaned


def traverse_directories(directory_list):
    """
    遍历传入的多个目录，查找所有.atr文件，并统计符号和附加注释的出现次数。
    参数:
        directory_list: 包含文件夹路径的列表
    返回:
        symbol_counts: Counter对象，记录所有符号出现次数
        aux_note_counts: Counter对象，记录所有辅助注释出现次数
    """
    all_symbols = []
    all_aux_notes = []

    # 遍历列表中的每个目录
    for directory in directory_list:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".atr"):  # 只处理.atr文件
                    file_path = os.path.join(root, file)
                    symbols, aux_notes = parse_annotation_file(file_path)
                    all_symbols.extend(symbols)
                    all_aux_notes.extend(aux_notes)

    # 统计符号和辅助注释的出现次数
    from collections import Counter
    symbol_counts = Counter(all_symbols)
    aux_note_counts = Counter(all_aux_notes)

    return symbol_counts, aux_note_counts



def main():
    symbol = aux_nots = 0
    directory_path1 = "/Users/shiqissszj/Documents/Datasets/mitdb"  # 替换为你的文件夹路径
    directory_path2 = "/Users/shiqissszj/Documents/Datasets/cudb"  # 替换为你的文件夹路径
    directory_path3 = "/Users/shiqissszj/Documents/Datasets/vfdb"  # 替换为你的文件夹路径

    symbol_counts, aux_note_counts = traverse_directories([directory_path1, directory_path2, directory_path3])

    # 输出符号及其出现次数
    print("符号及其出现次数：")
    for symbol, count in symbol_counts.items():
        print(f"{symbol}: {count}次")

    # 输出辅助注释及其出现次数
    print("\n辅助注释及其出现次数：")
    for aux_note, count in aux_note_counts.items():
        print(f"{aux_note}: {count}次")


if __name__ == "__main__":
    main()

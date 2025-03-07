"""
author: Li Chuanpu
date: 2024_01_26 17:35
"""

import pandas as pd

def save_ra_results(categories, results, save_name):
    """Save results to a CSV file."""
    cats = list(categories.keys())

    # 写入特征名称
    column1 = list(results[list(results.keys())[0]].keys())
    column_dict = {' ': column1}     # 表头为空

    value_dict = {}
    for cat in cats:
        value_dict[cat] = results[cat].values()

        column_dict.update(value_dict)

    df = pd.DataFrame(column_dict)
    df.to_csv(save_name, index=False)
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# 中文字体（支持中文 + ±）
pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))

excel_paths = ["summary_q2.xlsx", "summary_q5.xlsx", "summary_q10.xlsx", "summary_q20.xlsx", "summary_q50.xlsx"]

qs = [2, 5, 10, 20, 50]

styles = getSampleStyleSheet()
styleN = styles['BodyText']

for i in range(5):

    excel_path = excel_paths[i]
    pdf_path = f"table_q{qs[i]}.pdf"



    df = pd.read_excel(excel_path)

    # -----------------------------
    # 解析 mean±std
    # -----------------------------
    import re

    def parse_value(s):
        nums = re.findall(r"-?\d+\.\d+", str(s))
        if len(nums) >= 2:
            return float(nums[0]), float(nums[1])
        elif len(nums) == 1:
            return float(nums[0]), 0.0
        return 0.0, 0.0


    # -----------------------------
    # 保留两位小数
    # -----------------------------
    def format_value(x):

        if pd.isna(x):
            return ""

        s = str(x)

        if "+-" in s:
            m, s2 = s.split("+-")
            try:
                return f"{float(m):.2f}+-{float(s2):.2f}"
            except:
                return s

        try:
            return f"{float(x):.2f}"
        except:
            return s


    df_fmt = df.copy()
    for c in df.columns:
        df_fmt[c] = df_fmt[c].apply(format_value)

    # 数值矩阵
    values = df.applymap(parse_value).applymap(lambda x: x[0]).values

    # 表格数据
    table_data = [df.columns.tolist()] + df_fmt.values.tolist()

    # 创建 PDF

    from reportlab.platypus import SimpleDocTemplate, Table
    from reportlab.lib.pagesizes import landscape

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=(1800, 800)
    )

    page_width = 1600 - 40

    col_width = page_width / len(table_data[0])

    style = [
        ('FONTNAME', (0,0), (-1,-1), 'STSong-Light'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
        ('ALIGN', (1,1), (-1,-1), 'CENTER')
    ]

    # -----------------------------
    # 逐行处理颜色
    # -----------------------------
    for i in range(values.shape[0]):

        row = values[i,1:]   # 第一列通常是dataset
        row_idx = i + 1

        if np.all(pd.isna(row)):
            continue

        valid = np.array([v for v in row if v is not None])

        if len(valid) == 0:
            continue

        vmin = np.min(valid)
        vmax = np.max(valid)

        # -----------------------------
        # 找最大值 和 第二大值
        # -----------------------------
        best_cols = np.where(row == vmax)[0]

        # 找第二大值
        less_than_max = row[row < vmax]

        if len(less_than_max) > 0:
            second_max = np.max(less_than_max)
            second_cols = np.where(row == second_max)[0]
        else:
            second_cols = []

        for j,v in enumerate(row):

            col = j + 1

            if v is None:
                continue

            # 归一化
            if vmax == vmin:
                t = 0.5
            else:
                t = (v - vmin) / (vmax - vmin)

            # 红 → 绿 渐变
            r = 1 - t
            g = t
            b = 0.3

            color = colors.Color(r, g, b, alpha=0.6)

            style.append(
                ('BACKGROUND', (col,row_idx), (col,row_idx), color)
            )

        # -----------------------------
        # 最大值：全部加粗
        # -----------------------------
        for b in best_cols:
            table_data[row_idx][b+1] = Paragraph(
                f"<b>{table_data[row_idx][b+1]}</b>", styleN
            )
        # -----------------------------
        # 第二大值：下划线
        # -----------------------------
        for s in second_cols:
            text = table_data[row_idx][s+1]
            table_data[row_idx][s+1] = Paragraph(f"<u>{text}</u>", styleN)

    
    table = Table(
        table_data,
        colWidths=[col_width]*len(table_data[0])
    )

    table.setStyle(TableStyle(style))

    doc.build([table])

    print("PDF success:", pdf_path)
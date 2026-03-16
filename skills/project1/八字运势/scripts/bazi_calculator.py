#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
八字运势计算器
基于 lunar-python 库进行八字排盘、五行分析、十神计算、大运推算
"""

import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple


def calculate_bazi(birth_date: str, gender: str, year: int) -> Dict:
    """
    计算八字信息并返回结构化数据

    参数:
        birth_date: 出生日期字符串，格式如 "1990-03-15" 或 "1990年3月15日14:30"
        gender: 性别，"男" 或 "女"
        year: 预测年份

    返回:
        包含八字、五行、十神、大运等信息的字典
    """
    try:
        from lunar_python import Solar, Lunar
    except ImportError:
        return {"error": "缺少 lunar-python 库，请安装：pip install lunar-python"}

    # 解析出生日期
    parsed_date = parse_birth_date(birth_date)
    if not parsed_date:
        return {"error": f"无法解析出生日期：{birth_date}"}

    year_val, month_val, day_val, hour_val = parsed_date

    try:
        # 创建 Solar 对象
        solar = Solar.fromYmdHms(year_val, month_val, day_val, hour_val, 0, 0)
        lunar = solar.getLunar()

        # 获取八字四柱
        eight_char = lunar.getEightChar()
        bazi = {
            "年柱": eight_char.getYearGan() + eight_char.getYearZhi(),
            "月柱": eight_char.getMonthGan() + eight_char.getMonthZhi(),
            "日柱": eight_char.getDayGan() + eight_char.getDayZhi(),
            "时柱": eight_char.getTimeGan() + eight_char.getTimeZhi()
        }

        # 五行统计
        wuxing_stats = calculate_wuxing(bazi)

        # 日主（日柱天干）
        ri_zhu = bazi["日柱"][0]

        # 十神定位
        shishen = calculate_shishen(ri_zhu, bazi)

        # 大运计算
        dayun = calculate_dayun(ri_zhu, gender, year_val, lunar)

        # 流年干支
        liu_nian = get_year_ganzhi(year)

        # 流年十神
        liu_nian_shishen = get_shishen_for_gan_zhi(ri_zhu, liu_nian)

        # 当前大运信息
        current_dayun = get_current_dayun(dayun, year)

        return {
            "success": True,
            "八字": bazi,
            "五行统计": wuxing_stats,
            "日主": ri_zhu,
            "十神定位": shishen,
            "大运": dayun,
            "当前大运": current_dayun,
            "流年": liu_nian,
            "流年十神": liu_nian_shishen,
            "出生日期": f"{year_val}年{month_val}月{day_val}日 {hour_val}:00"
        }

    except Exception as e:
        import traceback
        return {"error": f"八字计算失败：{str(e)}", "detail": traceback.format_exc()}


def parse_birth_date(date_str: str) -> Optional[Tuple[int, int, int, int]]:
    """
    解析出生日期字符串

    支持格式：
    - "1990-03-15"
    - "1990-03-15 14:30"
    - "1990年3月15日"
    - "1990年3月15日 14时30分"
    - "1990年3月15日 午时"
    """
    date_str = date_str.strip()

    # 提取数字
    import re
    numbers = re.findall(r'\d+', date_str)

    if len(numbers) < 3:
        return None

    year = int(numbers[0])
    month = int(numbers[1])
    day = int(numbers[2])

    # 处理时辰
    hour = 12  # 默认午时
    if len(numbers) >= 4:
        hour = int(numbers[3])
    else:
        # 尝试从时辰名称解析
        shichen_map = {
            "子": 0, "丑": 2, "寅": 4, "卯": 6,
            "辰": 8, "巳": 10, "午": 12, "未": 14,
            "申": 16, "酉": 18, "戌": 20, "亥": 22
        }
        for shichen, h in shichen_map.items():
            if shichen in date_str:
                hour = h
                break

    return (year, month, day, hour)


def calculate_wuxing(bazi: Dict[str, str]) -> Dict[str, Dict]:
    """
    统计八字中的五行分布
    """
    # 天干五行映射
    tiangan_wuxing = {
        '甲': '木', '乙': '木',
        '丙': '火', '丁': '火',
        '戊': '土', '己': '土',
        '庚': '金', '辛': '金',
        '壬': '水', '癸': '水'
    }

    # 地支五行映射
    dizhi_wuxing = {
        '子': '水', '亥': '水',
        '寅': '木', '卯': '木',
        '辰': '土', '戌': '土', '丑': '土', '未': '土',
        '巳': '火', '午': '火',
        '申': '金', '酉': '金'
    }

    stats = {'木': 0, '火': 0, '土': 0, '金': 0, '水': 0}
    details = {'木': [], '火': [], '土': [], '金': [], '水': []}

    for pillar, ganzhi in bazi.items():
        gan = ganzhi[0]
        zhi = ganzhi[1]

        # 天干五行
        if gan in tiangan_wuxing:
            wx = tiangan_wuxing[gan]
            stats[wx] += 1
            details[wx].append(f"{pillar}天干({gan})")

        # 地支五行
        if zhi in dizhi_wuxing:
            wx = dizhi_wuxing[zhi]
            stats[wx] += 1
            details[wx].append(f"{pillar}地支({zhi})")

    return {
        "统计": stats,
        "详情": details,
        "缺失": [wx for wx, count in stats.items() if count == 0],
        "最旺": [wx for wx, count in stats.items() if count == max(stats.values())]
    }


def calculate_shishen(ri_zhu: str, bazi: Dict[str, str]) -> Dict[str, str]:
    """
    计算十神定位
    """
    # 日主与天干的关系
    wuxing_order = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸']
    wuxing_type = {
        '甲': '木', '乙': '木',
        '丙': '火', '丁': '火',
        '戊': '土', '己': '土',
        '庚': '金', '辛': '金',
        '壬': '水', '癸': '水'
    }

    # 获取日主五行
    ri_zhu_wuxing = wuxing_type.get(ri_zhu, '')

    # 判断阴阳（奇数为阳，偶数为阴）
    ri_zhu_yin_yang = wuxing_order.index(ri_zhu) % 2  # 0:阳, 1:阴

    shishen_map = {}

    # 计算各柱十神
    for pillar, ganzhi in bazi.items():
        gan = ganzhi[0]
        gan_wuxing = wuxing_type.get(gan, '')
        gan_yin_yang = wuxing_order.index(gan) % 2

        shishen = get_shishen_name(ri_zhu_wuxing, ri_zhu_yin_yang,
                                     gan_wuxing, gan_yin_yang)
        shishen_map[pillar] = shishen

    return shishen_map


def get_shishen_name(ri_zhu_wuxing: str, ri_zhu_yin_yang: int,
                      gan_wuxing: str, gan_yin_yang: int) -> str:
    """
    根据五行生克和阴阳关系确定十神
    """
    # 五行相生关系
    sheng_relation = {
        '木': '火', '火': '土', '土': '金', '金': '水', '水': '木'
    }
    # 五行相克关系
    ke_relation = {
        '木': '土', '土': '水', '水': '火', '火': '金', '金': '木'
    }

    # 同五行（比劫）
    if gan_wuxing == ri_zhu_wuxing:
        return "比肩" if gan_yin_yang == ri_zhu_yin_yang else "劫财"

    # 生我（印枭）
    if sheng_relation.get(gan_wuxing) == ri_zhu_wuxing:
        return "正印" if gan_yin_yang == ri_zhu_yin_yang else "偏印"

    # 我生（食伤）
    if sheng_relation.get(ri_zhu_wuxing) == gan_wuxing:
        return "食神" if gan_yin_yang == ri_zhu_yin_yang else "伤官"

    # 克我（官杀）
    if ke_relation.get(gan_wuxing) == ri_zhu_wuxing:
        return "正官" if gan_yin_yang == ri_zhu_yin_yang else "七杀"

    # 我克（财星）
    if ke_relation.get(ri_zhu_wuxing) == gan_wuxing:
        return "正财" if gan_yin_yang == ri_zhu_yin_yang else "偏财"

    return "未知"


def calculate_dayun(ri_zhu: str, gender: str, birth_year: int, lunar) -> List[Dict]:
    """
    计算大运（10年一运）
    """
    try:
        eight_char = lunar.getEightChar()
        yun = eight_char.getYun(gender == '男')
        da_yun_list = yun.getDaYun()

        dayun_list = []
        for i, da_yun in enumerate(da_yun_list):
            ganzhi = da_yun.getGanZhi()
            dayun_list.append({
                "序号": i + 1,
                "干支": ganzhi,
                "开始年龄": da_yun.getStartAge(),
                "结束年龄": da_yun.getEndAge(),
                "开始年份": da_yun.getStartYear(),
                "结束年份": da_yun.getEndYear()
            })

        return dayun_list[:8]  # 返回8步大运

    except Exception as e:
        # 返回简化的大运信息
        return []


def get_current_dayun(dayun_list: List[Dict], year: int) -> Optional[Dict]:
    """
    获取当前年份对应的大运
    """
    for dayun in dayun_list:
        if dayun["开始年份"] <= year <= dayun["结束年份"]:
            return dayun
    return None


def get_year_ganzhi(year: int) -> str:
    """
    获取指定年份的干支
    """
    try:
        from lunar_python import Solar
        solar = Solar.fromYmd(year, 1, 1)
        eight_char = solar.getLunar().getEightChar()
        return eight_char.getYearGan() + eight_char.getYearZhi()
    except:
        # 简化计算
        tiangan = ['庚', '辛', '壬', '癸', '甲', '乙', '丙', '丁', '戊', '己']
        dizhi = ['申', '酉', '戌', '亥', '子', '丑', '寅', '卯', '辰', '巳', '午', '未']
        return tiangan[(year - 4) % 10] + dizhi[(year - 4) % 12]


def get_shishen_for_gan_zhi(ri_zhu: str, ganzhi: str) -> Dict[str, str]:
    """
    获取干支的十神
    """
    gan = ganzhi[0]
    zhi = ganzhi[1]

    # 计算天干十神
    wuxing_order = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸']
    wuxing_type = {
        '甲': '木', '乙': '木',
        '丙': '火', '丁': '火',
        '戊': '土', '己': '土',
        '庚': '金', '辛': '金',
        '壬': '水', '癸': '水'
    }

    ri_zhu_wuxing = wuxing_type.get(ri_zhu, '')
    ri_zhu_yin_yang = wuxing_order.index(ri_zhu) % 2
    gan_wuxing = wuxing_type.get(gan, '')
    gan_yin_yang = wuxing_order.index(gan) % 2

    gan_shishen = get_shishen_name(ri_zhu_wuxing, ri_zhu_yin_yang,
                                    gan_wuxing, gan_yin_yang)

    return {
        "天干": f"{gan}({gan_shishen})",
        "地支": zhi
    }


def main():
    parser = argparse.ArgumentParser(description='八字运势计算器')
    parser.add_argument('--birth-date', type=str, required=True,
                        help='出生日期，如 "1990-03-15" 或 "1990年3月15日14:30"')
    parser.add_argument('--gender', type=str, required=True,
                        help='性别，"男" 或 "女"')
    parser.add_argument('--year', type=int, required=True,
                        help='预测年份，如 2024')

    args = parser.parse_args()

    result = calculate_bazi(args.birth_date, args.gender, args.year)

    if "error" in result:
        print(json.dumps({"success": False, "error": result["error"]},
                        ensure_ascii=False, indent=2))
        sys.exit(1)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

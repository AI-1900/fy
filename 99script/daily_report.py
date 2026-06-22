#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU/CUDA/算子开发任务跟踪型日报生成器 - Markdown Only 版

目标：
  - 只生成 Markdown 日报；
  - 不生成 HTML；
  - 不生成 PDF；
  - 不检查 Git；
  - 不读取代码仓；
  - 不拉取、不提交、不推送、不更新任何代码；
  - 默认统一输出到 ./report/ 文件夹；
  - 模板聚焦：昨日回顾、今日任务、问题跟踪、验证结果、明日计划；
  - 每个表格模块最多 5 条，避免日报膨胀；
  - 使用 emoji、分隔线、引用块、状态符号增强 Markdown 可读性。

依赖：
  - Python3 标准库，无第三方依赖。
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_PROJECT = "SDC200_DeepGEMM"
MAX_ITEMS = 5


@dataclass
class ReportConfig:
    project: str
    out_dir: Path
    report_date: str
    overwrite: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成 GPU/CUDA/算子开发任务跟踪型日报：只输出 Markdown，默认输出到 ./report，不包含任何 Git/代码仓检查。"
    )
    parser.add_argument("--out", default="report", help="输出目录，默认当前目录下的 report 文件夹。")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="项目名称，例如 DeepGEMM_SDC200 / SDC200_Port / CUDA_Kernel。")
    parser.add_argument("--date", default=None, help="日报日期，格式 YYYY-MM-DD。默认当天。")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖同名日报。默认不覆盖，会自动追加编号。")
    parser.add_argument("--install-cron", default=None, metavar="HH:MM", help="安装每日定时生成，例如 --install-cron 23:50。")
    return parser.parse_args()


def normalize_date(date_arg: str | None) -> str:
    if not date_arg:
        return datetime.now().strftime("%Y-%m-%d")
    try:
        return datetime.strptime(date_arg, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError as exc:
        raise SystemExit(f"[ERROR] --date 必须是 YYYY-MM-DD，例如 2026-06-17；当前输入：{date_arg}") from exc


def compact_day(report_date: str) -> str:
    return report_date.replace("-", "")


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def md_escape(s: object) -> str:
    return str(s).replace("|", "\\|").replace("\n", "<br>")


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        padded = row + [""] * (len(headers) - len(row))
        lines.append("| " + " | ".join(md_escape(x) for x in padded[:len(headers)]) + " |")
    return "\n".join(lines)


def divider(label: str = "") -> str:
    if label:
        return f"\n---\n\n### {label}\n"
    return "\n---\n"


def empty_rows(width: int, first_col: list[str]) -> list[list[str]]:
    rows = []
    for i in range(MAX_ITEMS):
        row = [""] * width
        row[0] = first_col[i]
        rows.append(row)
    return rows


def status_strip() -> str:
    return (
        "> **状态标识**："
        "`🔥P0` 必须推进 ｜ "
        "`⚡P1` 重要跟进 ｜ "
        "`🌱P2` 可延后 ｜ "
        "`🟥Open` 待解决 ｜ "
        "`🟨Tracking` 定位中 ｜ "
        "`🟩Fixed` 已修复 ｜ "
        "`✅Pass` 通过 ｜ "
        "`❌Fail` 失败 ｜ "
        "`⚪NA` 暂不适用"
    )


def section_header(config: ReportConfig) -> str:
    return f"""# 🌈 算子任务日报

> **日期**：{config.report_date}  
> **项目**：{config.project}  
> **定位**：记录问题、跟踪状态、控制任务进展、沉淀验证结论  


{status_strip()}

---

## 🎯 今日主线摘要

> 用 1～3 句话描述今天工作的主线


"""


def section_yesterday() -> str:
    headers = ["序号", "⏪ 昨日事项", "✅ 结果 / 证据", "🔁 未闭环点", "➡️ 今日承接动作"]
    rows = empty_rows(len(headers), ["①", "②", "③", "④", "⑤"])
    return "## 一、⏪ 昨日回顾与承接\n\n> 写影响今天工作结论，每条必须能被任务承接。\n\n" + md_table(headers, rows)


def section_today_tasks() -> str:
    headers = ["序号", "优先级", "🎯 今日主要任务", "📦 预期产出", "📏 完成标准", "状态"]
    rows = [
        ["①", "🔥 P0", "", "", "", "⬜ 未开始 / 🟨 进行中 / ✅ 已完成 / ⛔ 阻塞"],
        ["②", "🔥 P0", "", "", "", "⬜ 未开始 / 🟨 进行中 / ✅ 已完成 / ⛔ 阻塞"],
        ["③", "⚡ P1", "", "", "", "⬜ 未开始 / 🟨 进行中 / ✅ 已完成 / ⛔ 阻塞"],
        ["④", "⚡ P1", "", "", "", "⬜ 未开始 / 🟨 进行中 / ✅ 已完成 / ⛔ 阻塞"],
        ["⑤", "🌱 P2", "", "", "", "⬜ 未开始 / 🟨 进行中 / ✅ 已完成 / ⛔ 阻塞"],
    ]
    return "## 二、🎯 今日主要任务\n\n> 每条必须写清楚 **预期产出** 和 **完成标准**。\n\n" + md_table(headers, rows)


def section_issue_tracking() -> str:
    headers = ["编号", "🐞 问题现象", "💥 影响范围", "🧠 当前判断", "🛠️ 下一步动作", "状态"]
    rows = [
        ["🟥 ISSUE-1", "", "🔥 高 / ⚡ 中 / 🌱 低", "", "", "🟥 Open / 🟨 Tracking / 🟩 Fixed / ✅ Closed"],
        ["🟧 ISSUE-2", "", "🔥 高 / ⚡ 中 / 🌱 低", "", "", "🟥 Open / 🟨 Tracking / 🟩 Fixed / ✅ Closed"],
        ["🟨 ISSUE-3", "", "🔥 高 / ⚡ 中 / 🌱 低", "", "", "🟥 Open / 🟨 Tracking / 🟩 Fixed / ✅ Closed"],
        ["🟦 ISSUE-4", "", "🔥 高 / ⚡ 中 / 🌱 低", "", "", "🟥 Open / 🟨 Tracking / 🟩 Fixed / ✅ Closed"],
        ["⬜ ISSUE-5", "", "🔥 高 / ⚡ 中 / 🌱 低", "", "", "🟥 Open / 🟨 Tracking / 🟩 Fixed / ✅ Closed"],
    ]
    return "## 三、🐞 问题记录与状态跟踪\n\n> 不只记录现象，必须补齐 **影响范围、当前判断、下一步动作、状态**。\n\n" + md_table(headers, rows)


def section_validation() -> str:
    headers = ["序号", "🧪 验证对象", "⚙️ Shape / 配置", "🖥️ 平台 / 环境", "📊 关键结果", "结论"]
    rows = [
        ["①", "", "", "", "", "✅ Pass / ❌ Fail / ⚪ NA"],
        ["②", "", "", "", "", "✅ Pass / ❌ Fail / ⚪ NA"],
        ["③", "", "", "", "", "🚀 达标 / 🐢 未达标 / ⏳ 待测"],
        ["④", "", "", "", "", "🟩 正常 / 🟥 异常 / 🟨 待确认"],
        ["⑤", "", "", "", "", "✅ Pass / ❌ Fail / ⚪ NA"],
    ]
    return "## 四、🧪 验证结果与关键数据\n\n> 验证结论必带证据：shape、平台、命令、日志、截图、cycle、mismatch 任选其一。\n\n" + md_table(headers, rows)


def section_decision_log() -> str:
    headers = ["序号", "🧭 今日判断 / 决策", "依据", "影响", "后续动作"]
    rows = [
        ["①", "", "", "", ""],
        ["②", "", "", "", ""],
        ["③", "", "", "", ""],
        ["④", "", "", "", ""],
        ["⑤", "", "", "", ""],
    ]
    return "## 五、🧭 今日判断与决策记录\n\n> 记录关键判断，避免第二天重复讨论。\n\n" + md_table(headers, rows)


def section_tomorrow() -> str:
    headers = ["序号", "📌 明日计划", "🎯 目标结果", "⚠️ 依赖 / 风险", "📏 验收标准", "优先级"]
    rows = [
        ["①", "", "", "", "", "🔥 P0"],
        ["②", "", "", "", "", "🔥 P0"],
        ["③", "", "", "", "", "⚡ P1"],
        ["④", "", "", "", "", "⚡ P1"],
        ["⑤", "", "", "", "", "🌱 P2"],
    ]
    return "## 六、📌 明日预期计划\n\n> 计划不超过 5 条；P0 最多 2 条，保证第二天可执行。\n\n" + md_table(headers, rows)


def section_closure() -> str:
    headers = ["项目", "内容"]
    rows = [
        ["🎯 今日结论", ""],
        ["⛔ 最大阻塞", ""],
        ["🤝 需要协助", ""],
        ["⚠️ 风险等级", "🟢 低 / 🟡 中 / 🔴 高"],
        ["📣 是否升级", "✅ 否 / ⬆️ 是：对象=?，事项=?"],
    ]
    return "## 七、🏁 收敛结论\n\n> 用于日报最后快速扫读。今天是否闭环。\n\n" + md_table(headers, rows)


def section_footer() -> str:
    return f"""---

## 🧾 记录信息

- **生成时间**：{now_text()}
- **作者名称**：{platform.node() or "unknown"}

"""


def build_markdown(config: ReportConfig) -> str:
    parts = [
        section_header(config),
        section_yesterday(),
        section_today_tasks(),
        section_issue_tracking(),
        section_validation(),
        section_decision_log(),
        section_tomorrow(),
        section_closure(),
        section_footer(),
        "",
    ]
    return "\n\n---\n\n".join(parts)


def unique_path(path: Path, overwrite: bool) -> Path:
    if overwrite or not path.exists():
        return path

    parent = path.parent
    stem = path.stem
    suffix = path.suffix

    for i in range(1, 1000):
        candidate = parent / f"{stem}_{i:02d}{suffix}"
        if not candidate.exists():
            return candidate

    raise SystemExit(f"[ERROR] 无法生成唯一文件名：{path}")


def write_report(config: ReportConfig) -> Path:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    day = compact_day(config.report_date)
    md_path = unique_path(config.out_dir / f"TDR_{day}.md", config.overwrite)
    md_path.write_text(build_markdown(config), encoding="utf-8")
    return md_path


def shell_quote(s: str) -> str:
    return "'" + str(s).replace("'", "'\"'\"'") + "'"


def install_cron(time_hhmm: str, config: ReportConfig) -> None:
    try:
        hour_s, minute_s = time_hhmm.split(":")
        hour = int(hour_s)
        minute = int(minute_s)
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError
    except ValueError as exc:
        raise SystemExit(f"[ERROR] --install-cron 格式必须是 HH:MM，例如 23:50；当前输入：{time_hhmm}") from exc

    script_abs = Path(__file__).resolve()
    out_abs = config.out_dir.resolve()
    cmd = (
        f"{minute} {hour} * * * "
        f"{sys.executable} {shell_quote(str(script_abs))} "
        f"--out {shell_quote(str(out_abs))} "
        f"--project {shell_quote(config.project)} "
        f">> {shell_quote(str(out_abs / 'task_daily_report_cron.log'))} 2>&1"
    )

    try:
        current = subprocess.run(["crontab", "-l"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        existing = current.stdout if current.returncode == 0 else ""
        marker = str(script_abs)
        lines = [line for line in existing.splitlines() if marker not in line]
        lines.append(cmd)
        subprocess.run(["crontab", "-"], input="\n".join(lines).strip() + "\n", text=True, check=True)
    except FileNotFoundError as exc:
        raise SystemExit("[ERROR] 当前系统未找到 crontab，无法安装定时任务。") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"[ERROR] 安装 crontab 失败：{exc}") from exc

    print(f"[OK] 已安装每日定时任务：{time_hhmm}")
    print(f"[OK] 输出目录：{out_abs}")
    print("[OK] 定时任务只生成 Markdown 日报，不检查或更新代码仓。")


def main() -> None:
    args = parse_args()
    config = ReportConfig(
        project=args.project,
        out_dir=Path(args.out).expanduser(),
        report_date=normalize_date(args.date),
        overwrite=args.overwrite,
    )

    if args.install_cron:
        install_cron(args.install_cron, config)
        return

    md_path = write_report(config)

    print("[OK]:已生成任务跟踪日报(TASK DAILY REPORT)：")
    print(f"MD_Path: {md_path.resolve()}")



if __name__ == "__main__":
    main()

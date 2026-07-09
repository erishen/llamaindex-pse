"""LlamaIndex PSE — personal-crm 数据质量看门狗（crm-qa 任务）。

流程：
    默认：直接跑确定性扫描（零成本，无需 llamaindex / API key）
    --llm：用通用 PSE Workflow → Planner 提纲 → Specialist 写报告 →
           Evaluator(LLM) 评审 → Verify(程序化核对数字) → (不符则 Fix 重试)

用法:
    python run.py                 # 仅跑确定性扫描
    python run.py --llm          # 额外用 LLM 生成自然语言报告（需 llamaindex + OPENAI_API_KEY）
    python run.py --db <路径>     # 指定 crm.db
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv
    load_dotenv(BASE.parent.parent / ".env")
except Exception:
    pass  # 无 python-dotenv 时退化为直接用环境变量 / 默认路径

sys.path.insert(0, str(BASE))
from qa_scan import scan, DEFAULT_DB  # noqa: E402

# 让通用核心能找到本任务的提示词 (tasks/crm-qa/prompts/*.md)
sys.path.insert(0, str(BASE.parent.parent / "src"))


def _verify_report(report: str, scan_json: dict) -> tuple[list, list]:
    """程序化验证：报告里引用的数字必须能在扫描结果里找到。返回 (不符列表, 符合列表)。

    设计原则：宁可漏报也不误报——只要报告在合理范围内出现了真实值即视为通过，
    避免 LLM 用自然语言表述（如「约 6%」「占 39%」）被误判为幻觉而让 fix 死循环。
    """
    bad: list[str] = []
    ok: list[str] = []
    findings = scan_json.get("findings", [])
    summary = scan_json.get("summary", {})

    # ── findings：定位包含该 check 的「表格行」（行首为 |），只在该行取数字 ──
    for f in findings:
        check = f["check"]
        target = f["count"]
        line = None
        for mpos in re.finditer(re.escape(check), report):
            ls = report.rfind("\n", 0, mpos.start()) + 1
            le = report.find("\n", mpos.start())
            cand = report[ls:le] if le != -1 else report[ls:]
            if cand.lstrip().startswith("|"):
                line = cand
                break
        if line is None:
            bad.append(f"报告未提及 {check}（无表格行）")
            continue
        bm = re.search(r"\*\*([\d,]+)\*\*", line)
        cm = re.search(r"\|?\s*([\d,]{2,})\s*\|", line) if not bm else None
        m = bm or cm
        if not m:
            bad.append(f"{check} 报告未给出数量（真实为 {target}）")
            continue
        val = int(m.group(1).replace(",", ""))
        if val == target:
            ok.append(f"{check} = {target}")
        else:
            bad.append(f"{check} 报告写 {val} 但真实为 {target}")

    # ── summary：报告里出现过该真实值（含千分位）即过 ──
    for k, v in summary.items():
        if k not in report:
            continue
        forms = {str(v), f"{v:,}"}
        if any(form in report for form in forms):
            ok.append(f"概况 {k} = {v}")
        else:
            bad.append(f"概况 {k} 报告提及但未出现真实值 {v}")
    return bad, ok


def _verify_state(state: dict) -> tuple[list, list]:
    return _verify_report(state.get("artifact", ""), state.get("task_data", {}).get("scan_result", {}))


def main():
    ap = argparse.ArgumentParser(description="personal-crm 数据质量看门狗 (llamaindex-pse)")
    ap.add_argument("--db", default=os.getenv("CRM_DB_PATH", DEFAULT_DB))
    ap.add_argument("--llm", action="store_true",
                    help="用 LLM 生成自然语言 QA 报告（需 llamaindex + API key）")
    ap.add_argument("--provider", choices=["deepseek", "agnes"], default="deepseek",
                    help="LLM 网关：deepseek（默认）或 agnes（需配置 AGNES_*）")
    args = ap.parse_args()

    print(f"🔍 运行确定性扫描: {args.db}")
    scan_result = scan(args.db)
    s = scan_result["summary"]
    print(f"   概况: contacts={s['contacts']}  contact_records={s['contact_records']}  "
          f"chat_messages={s['chat_messages']}")
    print(f"   发现 {len(scan_result['findings'])} 项")

    if not args.llm:
        for f in scan_result["findings"]:
            print(f"  [{f['severity'].upper()}] {f['check']} (count={f['count']}) — {f['description']}")
        print("\n结论:", "✅ 无明显高/中危问题" if scan_result["ok"] else "⚠️ 存在需关注的问题")
        print("\n（加 --llm 可用 LlamaIndex PSE 生成自然语言报告并做数字核对）")
        return

    # ── LLM 模式：复用通用 PSE 核心 ──
    try:
        from llamaindex_pse.config import settings
        from llamaindex_pse.workflow import build_workflow
    except Exception as e:
        print(f"❌ 无法加载 llamaindex 运行环境: {e}\n（请先 `uv sync` 并配置对应 provider 的 API key）")
        sys.exit(1)

    scan_json_str = json.dumps(scan_result, ensure_ascii=False, indent=2)
    max_retries = settings.PSE_MAX_RETRIES or 3

    workflow = build_workflow(
        task="crm-qa",
        verify_fn=_verify_state,
        use_planner=True,
        max_retries=max_retries,
        provider=args.provider,
    )

    task_input = (
        "以下是 personal-crm 数据库的自动扫描 JSON 结果，请据此撰写 QA 报告：\n\n"
        f"{scan_json_str}"
    )

    result = asyncio.run(workflow.run(
        task_input=task_input,
        task_data={"scan_result": scan_result},
        max_retries=max_retries,
    ))
    report = result.get("artifact", "")
    out_path = BASE / "qa_report.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"\n✅ 自然语言 QA 报告已保存 → {out_path}")


if __name__ == "__main__":
    main()

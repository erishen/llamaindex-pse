"""personal-crm 数据质量扫描（确定性，只读）。

扫描已知坑类 + 参照完整性问题，输出结构化 findings。
- 可作为独立 CLI 运行：python qa_scan.py [--db 路径] [--json]
- 也可被 llamaindex-pse 的 crm-qa 任务作为工具调用（见 tools.py 的 crm_qa_scan）。

设计原则：只读打开（mode=ro&immutable），绝不修改 crm.db。
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from datetime import date


@dataclass
class Finding:
    check: str
    severity: str  # "high" | "medium" | "low" | "info"
    count: int
    description: str
    samples: list = field(default_factory=list)


# 中性占位默认值——真实库路径通过环境变量 CRM_DB_PATH 或 --db 指定，
# 避免在源码中硬编码个人绝对路径（隐私考量）。
DEFAULT_DB = "crm.db"


def _open(db_path: str) -> sqlite3.Connection:
    # 只读 + immutable：即使库处于 WAL 运行态也能安全打开，不阻塞写入方
    con = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    con.row_factory = sqlite3.Row
    return con


def scan(db_path: str = DEFAULT_DB) -> dict:
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"找不到 CRM 数据库: {db_path}\n"
            f"请通过环境变量 CRM_DB_PATH 指定，或在运行时加 --db <绝对路径>；\n"
            f"例如在 .env 中设置 CRM_DB_PATH=/path/to/crm.db。"
        )
    con = _open(db_path)
    cur = con.cursor()
    findings: list[Finding] = []

    def q1(sql: str, params=()) -> list[sqlite3.Row]:
        cur.execute(sql, params)
        return cur.fetchall()

    # 0. 库概况
    n_contacts = cur.execute("SELECT COUNT(*) FROM contacts").fetchone()[0]
    n_records = cur.execute("SELECT COUNT(*) FROM contact_records").fetchone()[0]
    n_chat = cur.execute("SELECT COUNT(*) FROM chat_messages").fetchone()[0]
    summary = {
        "contacts": n_contacts,
        "contact_records": n_records,
        "chat_messages": n_chat,
    }

    # 1. 重复 wechat_id / wechat_id_clean（import 未去重）
    dups = q1(
        """SELECT wechat_id, COUNT(*) AS cnt FROM contacts
           WHERE wechat_id IS NOT NULL AND wechat_id != ''
           GROUP BY wechat_id HAVING cnt > 1 ORDER BY cnt DESC"""
    )
    if dups:
        findings.append(Finding(
            "duplicate_wechat_id", "high", sum(d["cnt"] for d in dups),
            "存在重复 wechat_id（同一微信账号多条联系人），通常是 import 未去重导致",
            [f"{d['wechat_id']} x{d['cnt']}" for d in dups[:10]],
        ))
    dups2 = q1(
        """SELECT wechat_id_clean, COUNT(*) AS cnt FROM contacts
           WHERE wechat_id_clean IS NOT NULL AND wechat_id_clean != ''
           GROUP BY wechat_id_clean HAVING cnt > 1 ORDER BY cnt DESC"""
    )
    if dups2:
        findings.append(Finding(
            "duplicate_wechat_id_clean", "high", sum(d["cnt"] for d in dups2),
            "wechat_id_clean 重复（清洗后仍有重复）",
            [f"{d['wechat_id_clean']} x{d['cnt']}" for d in dups2[:10]],
        ))

    # 2. name 仍是 wxid / 含 @ / 为空（未改名）
    bad = q1(
        """SELECT id, name, wechat_id FROM contacts
           WHERE name = wechat_id
              OR name LIKE 'wxid_%'
              OR name LIKE '%@%'
              OR name IS NULL OR TRIM(name) = ''"""
    )
    if bad:
        findings.append(Finding(
            "name_not_renamed", "medium", len(bad),
            "name 仍是 wxid / 含@ / 为空（未从微信原始 ID 改名为可读名）",
            [f"id={b['id']} name={b['name']!r}" for b in bad[:10]],
        ))

    # 3. 有 wechat_id 但 wechat_id_clean 为空（清洗遗漏）
    miss_clean = q1(
        """SELECT COUNT(*) AS cnt FROM contacts
           WHERE wechat_id IS NOT NULL AND wechat_id != ''
             AND (wechat_id_clean IS NULL OR wechat_id_clean = '')"""
    )
    if miss_clean and miss_clean[0]["cnt"]:
        findings.append(Finding(
            "wechat_id_clean_missing", "low", miss_clean[0]["cnt"],
            "有 wechat_id 但 wechat_id_clean 为空，清洗可能遗漏",
        ))

    # 4. 孤儿 chat_messages：contact_id 为 NULL
    null_c = q1("SELECT COUNT(*) AS cnt FROM chat_messages WHERE contact_id IS NULL")
    if null_c and null_c[0]["cnt"]:
        findings.append(Finding(
            "chat_contact_id_null", "medium", null_c[0]["cnt"],
            "chat_messages.contact_id 为 NULL（未能匹配到联系人，可能是群/非好友或匹配失败）",
        ))
    # 悬空外键：contact_id 非空但联系人不存在
    orphan = q1(
        """SELECT COUNT(*) AS cnt FROM chat_messages cm
           LEFT JOIN contacts c ON cm.contact_id = c.id
           WHERE cm.contact_id IS NOT NULL AND c.id IS NULL"""
    )
    if orphan and orphan[0]["cnt"]:
        findings.append(Finding(
            "chat_orphan_fk", "high", orphan[0]["cnt"],
            "chat_messages.contact_id 指向不存在的联系人（外键悬空）",
        ))

    # 5. contact_records 外键悬空
    rec_orphan = q1(
        """SELECT COUNT(*) AS cnt FROM contact_records r
           LEFT JOIN contacts c ON r.contact_id = c.id
           WHERE c.id IS NULL"""
    )
    if rec_orphan and rec_orphan[0]["cnt"]:
        findings.append(Finding(
            "record_orphan_fk", "high", rec_orphan[0]["cnt"],
            "contact_records.contact_id 指向不存在的联系人",
        ))

    # 6. contact_date 异常
    fut = q1(
        "SELECT COUNT(*) AS cnt FROM contact_records WHERE contact_date > ?",
        (date.today().isoformat(),),
    )
    if fut and fut[0]["cnt"]:
        findings.append(Finding(
            "contact_date_future", "high", fut[0]["cnt"],
            "contact_date 在未来（明显错误，可能是时区/写入 bug）",
        ))
    old = q1("SELECT COUNT(*) AS cnt FROM contact_records WHERE contact_date < '2000-01-01'")
    if old and old[0]["cnt"]:
        findings.append(Finding(
            "contact_date_implausible", "medium", old[0]["cnt"],
            "contact_date 早于 2000 年（异常）",
        ))

    # 7. UTC/local 错位候选：contact_date != DATE(created_at)
    tz = q1(
        """SELECT COUNT(*) AS cnt FROM contact_records
           WHERE created_at IS NOT NULL AND contact_date != DATE(created_at)"""
    )
    if tz and tz[0]["cnt"]:
        findings.append(Finding(
            "contact_date_tz_mismatch_candidate", "info", tz[0]["cnt"],
            "contact_date 与 created_at 的 UTC 日期不一致（可能是已知 UTC/local 跨日 bug 残留；含合理改期，仅供参考）",
        ))

    # 8. 空内容
    empty_rec = q1(
        "SELECT COUNT(*) AS cnt FROM contact_records WHERE content IS NULL OR TRIM(content) = ''"
    )
    if empty_rec and empty_rec[0]["cnt"]:
        findings.append(Finding(
            "empty_record_content", "low", empty_rec[0]["cnt"],
            "contact_records.content 为空",
        ))
    empty_chat = q1(
        "SELECT COUNT(*) AS cnt FROM chat_messages WHERE content IS NULL OR TRIM(content) = ''"
    )
    if empty_chat and empty_chat[0]["cnt"]:
        findings.append(Finding(
            "empty_chat_content", "low", empty_chat[0]["cnt"],
            "chat_messages.content 为空",
        ))

    # 9. 有聊天但 0 条 contact_record 的联系人
    no_rec = q1(
        """SELECT COUNT(*) AS cnt FROM contacts c
           WHERE EXISTS (SELECT 1 FROM chat_messages cm WHERE cm.contact_id = c.id)
             AND NOT EXISTS (SELECT 1 FROM contact_records r WHERE r.contact_id = c.id)"""
    )
    if no_rec and no_rec[0]["cnt"]:
        findings.append(Finding(
            "chat_no_record", "info", no_rec[0]["cnt"],
            "有聊天记录但没有任何 contact_record（往来未沉淀为记录）",
        ))

    con.close()
    high_med = [f for f in findings if f.severity in ("high", "medium")]
    return {
        "db_path": db_path,
        "summary": summary,
        "findings": [asdict(f) for f in findings],
        "ok": len(high_med) == 0,
    }


def main():
    ap = argparse.ArgumentParser(description="personal-crm 数据质量扫描")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    try:
        result = scan(args.db)
    except FileNotFoundError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(f"数据库: {result['db_path']}")
    s = result["summary"]
    print(f"概况: contacts={s['contacts']}  contact_records={s['contact_records']}  chat_messages={s['chat_messages']}")
    print(f"\n发现 {len(result['findings'])} 项:")
    for f in result["findings"]:
        print(f"  [{f['severity'].upper()}] {f['check']} (count={f['count']})")
        print(f"      {f['description']}")
        if f["samples"]:
            print(f"      样本: {', '.join(f['samples'][:5])}")
    print("\n结论:", "✅ 无明显高/中危问题" if result["ok"] else "⚠️ 存在需关注的问题")


if __name__ == "__main__":
    main()

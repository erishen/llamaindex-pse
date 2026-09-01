"""hot-news 任务测试入口：将 tasks/hot-news 加入 sys.path 以便 import run.py。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

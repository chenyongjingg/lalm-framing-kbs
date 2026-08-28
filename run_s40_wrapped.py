# -*- coding: utf-8 -*-
"""run_s40_wrapped.py — 运行时包装运行 s40_benign_control.py（不改对方源码）。

对方脚本 s40_benign_control.py:235 把普通函数 log 传给 common_utils.ModelManager
（签名要求 logging.Logger）→ mm.load 内 self.log.info 崩溃（AttributeError）。
此处 monkey-patch ModelManager.__init__：把函数 logger 包成带 .info/.warning/...
的 shim，转发到原函数。与 S12 gpu1_common._FnLogger 同族修复，但这里不改任何
源码文件。用法： python run_s40_wrapped.py --n-queries 40 --templates 0,1,2
"""
import sys
import time
import threading
import common_utils

NQ = 40
TPL = "0,1,2"
MODEL = "gemma_4_e2b"
for i, a in enumerate(sys.argv[1:]):
    if a == "--n-queries" and i + 2 <= len(sys.argv[1:]):
        NQ = int(sys.argv[i + 2])
    elif a == "--templates" and i + 2 <= len(sys.argv[1:]):
        TPL = sys.argv[i + 2]
    elif a == "--model" and i + 2 <= len(sys.argv[1:]):
        MODEL = sys.argv[i + 2]


def _wrap(fn):
    class _Shim:
        def _emit(self, level, msg):
            try:
                fn("[%s] %s" % (level, msg))
            except Exception:
                pass

        def info(self, m, *a):
            self._emit("INFO", m % a if a else m)

        def warning(self, m, *a):
            self._emit("WARNING", m % a if a else m)

        def warn(self, m, *a):
            self._emit("WARNING", m % a if a else m)

        def error(self, m, *a):
            self._emit("ERROR", m % a if a else m)

        def debug(self, m, *a):
            self._emit("DEBUG", m % a if a else m)

        def exception(self, m, *a):
            self._emit("EXCEPTION", m % a if a else m)

        def setLevel(self, lvl):
            pass

        def addHandler(self, h):
            pass
    return _Shim()


_orig_init = common_utils.ModelManager.__init__


def _patched_init(self, logger, *a, **k):
    if logger is not None and not hasattr(logger, "info"):
        logger = _wrap(logger)
    return _orig_init(self, logger, *a, **k)


common_utils.ModelManager.__init__ = _patched_init

# 心跳（60s 一次），供监控判断存活
HB = "/root/lalm_framing_revision_v6/logs/benign_control.hb"


def _hb_loop():
    while True:
        try:
            with open(HB, "w") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S UTC") + "\n")
        except Exception:
            pass
        time.sleep(60)


threading.Thread(target=_hb_loop, daemon=True).start()

# 以 __main__ 运行对方脚本
import runpy  # noqa: E402
sys.argv = ["s40_benign_control.py",
            "--n-queries", str(NQ),
            "--templates", TPL,
            "--model", MODEL]
runpy.run_path("/root/lalm_framing_revision_v6/s40_benign_control.py",
               run_name="__main__")

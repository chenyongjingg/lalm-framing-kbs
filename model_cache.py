"""
model_cache.py — 模型快速存储预置与后台预取模块

针对 overlayfs I/O 瓶颈（~5MB/s 读取、随机挂起 8h）的核心缓解方案：

  1. FastStorage      — 自动检测快速存储（/dev/shm RAM 盘 / 本地 NVMe scratch），
                        测量实际读写速度，决策是否值得预置
  2. copy_with_watchdog — 目录复制 + 停滞看门狗（N 秒无字节进展即中止，
                        防止 overlayfs 挂起导致无限等待）
  3. prestage_model   — 将模型从 HF 缓存（overlayfs）复制到快速存储，
                        加载路径切换到快速存储，读取速度提升 10-100 倍
  4. ModelPrefetcher  — 后台线程：当前模型推理期间预取下一个模型，
                        GPU 计算与 I/O 预取并行，消除串行等待

可移植性：自动检测，无硬编码路径；无快速存储时优雅降级为直接加载。
"""

import logging
import os
import shutil
import threading
import time
from pathlib import Path

log = logging.getLogger("model_cache")


# ---------------------------------------------------------------------------
# 快速存储检测与测速
# ---------------------------------------------------------------------------

class FastStorage:
    """检测并基准测试可用的快速存储。"""

    def __init__(self, preferred: str = "auto", max_gb: float = 24.0,
                 logger: logging.Logger = None):
        self.log = logger or log
        self.max_bytes = int(max_gb * 1e9)
        self.path, self.kind, self.speed_mbps = self._select(preferred)

    def _candidates(self) -> list:
        cands = []
        # /dev/shm（tmpfs，RAM 速度，Linux）
        if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK):
            cands.append(("/dev/shm", "ramdisk"))
        # 常见 HPC 本地 scratch
        for p in ["/scratch", "/local_scratch", "/local", "/fast"]:
            if os.path.isdir(p) and os.access(p, os.W_OK):
                cands.append((p, "local_nvme"))
        # TMPDIR / /tmp 兜底（可能也是 overlayfs，需测速判断）
        for env in ["TMPDIR"]:
            p = os.environ.get(env)
            if p and os.path.isdir(p) and os.access(p, os.W_OK):
                cands.append((p, "tmpdir"))
        if os.path.isdir("/tmp") and os.access("/tmp", os.W_OK):
            cands.append(("/tmp", "tmp"))
        return cands

    def _select(self, preferred: str):
        cands = self._candidates()
        if not cands:
            self.log.warning("未找到可写快速存储，预置功能禁用")
            return None, "none", 0.0
        if preferred != "auto":
            for p, kind in cands:
                if p == preferred:
                    speed = benchmark_read_write(p, 32)
                    return p, kind, speed[1]
            self.log.warning("指定快速存储 %s 不可用，回退自动检测", preferred)
        # 测速选最快
        best, best_speed = None, 0.0
        for p, kind in cands:
            free = shutil.disk_usage(p).free
            if free < self.max_bytes // 2:
                self.log.info("跳过 %s（可用 %.1fGB 不足）", p, free / 1e9)
                continue
            try:
                _, r_speed = benchmark_read_write(p, 32)
            except Exception as e:  # noqa: BLE001
                self.log.warning("测速失败 %s: %s", p, str(e)[:100])
                continue
            self.log.info("快速存储候选: %s (%s) 读取 %.0f MB/s 可用 %.1fGB",
                          p, kind, r_speed, free / 1e9)
            if r_speed > best_speed:
                best, best_speed = (p, kind), r_speed
        if best is None:
            return None, "none", 0.0
        return best[0], best[1], best_speed

    @property
    def available(self) -> bool:
        return self.path is not None

    def worth_prestage(self, source_speed_mbps: float) -> bool:
        """快速存储读取速度显著高于源（≥2 倍）才值得预置。"""
        return self.available and self.speed_mbps >= max(source_speed_mbps * 2, 50)

    def reserve_dir(self, name: str) -> Path:
        d = Path(self.path) / "lalm_model_cache" / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def free_gb(self) -> float:
        return shutil.disk_usage(self.path).free / 1e9 if self.path else 0.0


def benchmark_read_write(path: str, size_mb: int = 32) -> tuple:
    """写入+读取测速。返回 (write_MBps, read_MBps)。"""
    test_file = Path(path) / f".lalm_speedtest_{os.getpid()}"
    data = os.urandom(size_mb * 1024 * 1024)
    try:
        t0 = time.time()
        with open(test_file, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        w_time = time.time() - t0
        # 清页缓存不可行（需 root），改读文件不同偏移区域近似
        t0 = time.time()
        with open(test_file, "rb") as f:
            while f.read(4 * 1024 * 1024):
                pass
        r_time = max(time.time() - t0, 1e-6)
        return size_mb / max(w_time, 1e-6), size_mb / r_time
    finally:
        test_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 带停滞看门狗的目录复制
# ---------------------------------------------------------------------------

def dir_size(path: Path) -> int:
    path = Path(path)
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def copy_with_watchdog(src: Path, dst: Path, stall_timeout: int = 120,
                       progress_interval: int = 30,
                       logger: logging.Logger = None) -> bool:
    """逐文件复制目录；看门狗监测停滞（stall_timeout 秒无字节进展则中止）。

    返回 True=完成, False=停滞中止（dst 残留由调用方清理）。
    """
    logger = logger or log
    src = Path(src); dst = Path(dst)
    files = [f for f in src.rglob("*") if f.is_file()]
    total_bytes = sum(f.stat().st_size for f in files)
    state = {"copied": 0, "done": False, "stalled": False}

    def _copy():
        for f in files:
            rel = f.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(f, "rb") as fin, open(target, "wb") as fout:
                    while True:
                        chunk = fin.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        fout.write(chunk)
                        state["copied"] += len(chunk)
            except OSError as e:
                logger.error("复制失败 %s: %s", f, str(e)[:150])
                state["stalled"] = True
                return
        state["done"] = True

    th = threading.Thread(target=_copy, daemon=True)
    t0 = time.time()
    th.start()
    last_copied, last_change = 0, time.time()
    last_report = time.time()
    while th.is_alive():
        time.sleep(2)
        if state["copied"] != last_copied:
            last_copied = state["copied"]
            last_change = time.time()
        if time.time() - last_change > stall_timeout:
            state["stalled"] = True
            logger.error("复制停滞 %ds 无进展（已复制 %.1f/%.1fGB），中止"
                         "（疑似 overlayfs I/O 挂起）",
                         stall_timeout, last_copied / 1e9, total_bytes / 1e9)
            return False
        if time.time() - last_report > progress_interval:
            elapsed = time.time() - t0
            speed = last_copied / 1e9 / max(elapsed, 1)
            logger.info("预置进度 %.1f/%.1fGB (%.0f%%，%.1f GB/s)",
                        last_copied / 1e9, total_bytes / 1e9,
                        100 * last_copied / max(total_bytes, 1), speed)
            last_report = time.time()
    ok = state["done"] and not state["stalled"]
    if ok:
        elapsed = time.time() - t0
        logger.info("预置复制完成: %.1fGB 耗时 %.0fs（%.1f MB/s）",
                    total_bytes / 1e9, elapsed,
                    total_bytes / 1e6 / max(elapsed, 1))
    return ok


def verify_copy(src: Path, dst: Path) -> bool:
    """校验复制完整性（文件数与总字节数一致）。"""
    src = Path(src); dst = Path(dst)
    src_files = {f.relative_to(src): f.stat().st_size
                 for f in src.rglob("*") if f.is_file()}
    for rel, size in src_files.items():
        target = dst / rel
        if not target.exists() or target.stat().st_size != size:
            return False
    return True


# ---------------------------------------------------------------------------
# 模型预置
# ---------------------------------------------------------------------------

def resolve_hf_snapshot(model_id_or_path: str, hf_home: str = None) -> Path:
    """将 HF 模型 id 解析为本地快照目录（不触发下载）。"""
    p = Path(model_id_or_path).expanduser()
    if p.is_dir():
        return p
    # 扫描 HF 缓存
    homes = [hf_home] if hf_home else []
    homes += [os.environ.get("HF_HOME"), os.environ.get("HUGGINGFACE_HUB_CACHE"),
              "~/.cache/huggingface"]
    safe_name = "models--" + model_id_or_path.replace("/", "--")
    for h in homes:
        if not h:
            continue
        snap_root = Path(h).expanduser() / "hub" / safe_name / "snapshots"
        if snap_root.is_dir():
            snaps = sorted(snap_root.iterdir(), key=lambda d: d.stat().st_mtime)
            if snaps:
                return snaps[-1]
    raise FileNotFoundError(
        f"模型 {model_id_or_path} 不在本地缓存（HF_HOME={hf_home or os.environ.get('HF_HOME') or '~/.cache/huggingface'}）。"
        f"请先运行 prestage_models.py 下载。")


def prestage_model(model_key: str, model_id_or_path: str, fast: FastStorage,
                   hf_home: str = None, stall_timeout: int = 120,
                   logger: logging.Logger = None) -> str:
    """将模型预置到快速存储。返回用于加载的路径（快速存储路径或原路径）。

    决策逻辑：
      - 快速存储不可用 / 容量不足 → 返回原路径（直接加载）
      - 复制停滞中止 → 清理残留，返回原路径
      - 复制完成但校验失败 → 清理，返回原路径
    """
    logger = logger or log
    src = resolve_hf_snapshot(model_id_or_path, hf_home)
    if not fast.available:
        logger.info("[%s] 无快速存储，直接从源加载: %s", model_key, src)
        return str(src)
    need = dir_size(src)
    if need > fast.max_bytes or fast.free_gb() * 1e9 < need * 1.1:
        logger.warning("[%s] 模型 %.1fGB 超出快速存储预算/余量，直接加载",
                       model_key, need / 1e9)
        return str(src)
    dst = fast.reserve_dir(model_key)
    marker = dst / ".prestage_complete"
    if marker.exists() and verify_copy(src, dst):
        logger.info("[%s] 预置缓存命中: %s", model_key, dst)
        return str(dst)
    # 清理旧残留后重新复制
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True, exist_ok=True)
    logger.info("[%s] 预置到快速存储: %s → %s（%.1fGB）",
                model_key, src, dst, need / 1e9)
    ok = copy_with_watchdog(src, dst, stall_timeout=stall_timeout, logger=logger)
    if ok and verify_copy(src, dst):
        marker.write_text(str(time.time()))
        return str(dst)
    shutil.rmtree(dst, ignore_errors=True)
    logger.warning("[%s] 预置失败，回退直接加载: %s", model_key, src)
    return str(src)


# ---------------------------------------------------------------------------
# 后台预取器
# ---------------------------------------------------------------------------

class ModelPrefetcher:
    """后台预取：当前模型推理期间，将后续模型复制到快速存储。

    用法:
        pf = ModelPrefetcher(fast, hf_home)
        pf.submit("qwen3b", "Qwen/Qwen2.5-3B-Instruct")   # 提交后立即返回
        ...
        path = pf.wait_ready("qwen3b")                    # 需要时等待完成
        pf.shutdown()
    """

    def __init__(self, fast: FastStorage, hf_home: str = None,
                 stall_timeout: int = 120, logger: logging.Logger = None):
        self.fast = fast
        self.hf_home = hf_home
        self.stall_timeout = stall_timeout
        self.log = logger or log
        self._results = {}   # key -> prestaged path 或 None(失败)
        self._events = {}    # key -> threading.Event
        self._lock = threading.Lock()
        self._threads = []
        self._shutdown = False

    def submit(self, model_key: str, model_id_or_path: str):
        if not self.fast.available:
            return
        with self._lock:
            if model_key in self._events:
                return
            ev = threading.Event()
            self._events[model_key] = ev

        def _job():
            try:
                path = prestage_model(model_key, model_id_or_path, self.fast,
                                      self.hf_home, self.stall_timeout, self.log)
                self._results[model_key] = path
            except Exception as e:  # noqa: BLE001
                self.log.error("[prefetch/%s] 预取失败: %s",
                               model_key, str(e)[:200])
                self._results[model_key] = None
            finally:
                ev.set()

        th = threading.Thread(target=_job, daemon=True,
                              name=f"prefetch-{model_key}")
        th.start()
        self._threads.append(th)
        self.log.info("[prefetch] 已提交后台预取: %s", model_key)

    def wait_ready(self, model_key: str, timeout: float = None):
        """等待预取完成。返回预置路径；未提交/失败返回 None。"""
        ev = self._events.get(model_key)
        if ev is None:
            return None
        ev.wait(timeout)
        return self._results.get(model_key)

    def shutdown(self):
        self._shutdown = True
        for th in self._threads:
            th.join(timeout=5)

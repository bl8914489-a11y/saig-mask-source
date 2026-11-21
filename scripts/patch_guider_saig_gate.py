"""给 FADE 的 guider 加一个「环境变量门控」的身份闸门注入点（第 4 步用）。

为什么要打这个补丁：
  mask/<seq>.pt 实测是 (49,480,720) bool；FADE 走
    max_pool2d(8×8) -> 每 4 帧取 max -> 13×60×90 -> dilate -> mask.bool()
  最后那句 .bool() 会把 (1 − 0.5·g↑) 的软权重全抹成 1，公式等于没实现。
  所以注入点必须在 .bool() 这一步，并且只在设置了 SAIG_GID 时生效。

安全性：
  - 不设 SAIG_GID 时走原分支 `mask = mask.bool()`，与 B0 / go3 完全同路径；
  - 打补丁前自动备份 <file>.bak_saig，并校验 `mask = mask.bool()` 恰好出现 1 次，
    不满足就中止，绝不盲改。
"""
import argparse
import os
import shutil

TARGET = "/root/saig/FADE/diffusion_core/guiders/guidance_editing_cogvideo.py"
OLD = "            mask = mask.bool()"
NEW = '''            _saig_gid = __import__('os').environ.get('SAIG_GID', '')
            if _saig_gid:
                import numpy as _snp
                _g = _snp.load(_saig_gid).astype('float32').clip(0.0, 1.0)
                _gt = torch.from_numpy(_g).to(device=mask.device, dtype=mask.dtype)
                _gt = _gt.unsqueeze(0).expand(mask.shape[0], -1, -1)
                mask = mask * (1.0 - 0.5 * _gt)
                print("SAIG_GATE_APPLIED", _saig_gid, "mean=%.4f" % float(mask.mean()),
                      flush=True)
            else:
                mask = mask.bool()'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=TARGET)
    ap.add_argument("--check", action="store_true", help="只检查，不改")
    args = ap.parse_args()

    if not os.path.exists(args.file):
        print("NO_FILE", args.file)
        return 2
    src = open(args.file, encoding="utf-8").read()
    n_old = src.count(OLD)
    n_gate = src.count("SAIG_GATE_APPLIED")
    print("TARGET", args.file)
    print("COUNT_MASK_BOOL", n_old)
    print("COUNT_GATE_APPLIED", n_gate)

    if n_gate == 1:
        print("ALREADY_PATCHED")
        return 0
    if n_old != 1:
        print("ABORT: `mask = mask.bool()` 出现 %d 次，不是 1 次，不做修改" % n_old)
        return 3
    if "import os" not in src.split("mask = mask.bool()")[0]:
        print("NOTE: 该文件未导入 os，注入块用 __import__('os') 自取，不依赖文件头部导入")
    if args.check:
        print("CHECK_OK (未修改)")
        return 0

    bak = args.file + ".bak_saig"
    if not os.path.exists(bak):
        shutil.copy2(args.file, bak)
        print("BACKUP", bak)
    out = src.replace(OLD, NEW, 1)
    with open(args.file, "w", encoding="utf-8") as f:
        f.write(out)
    chk = open(args.file, encoding="utf-8").read()
    print("WROTE", args.file)
    print("VERIFY_GATE", chk.count("SAIG_GATE_APPLIED"))
    print("VERIFY_ORIG_PRINT", chk.count("mask = mask.bool()"))
    print("RESULT: PATCH_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

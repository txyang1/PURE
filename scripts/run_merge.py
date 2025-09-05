#!/usr/bin/env python3
import os
import argparse
import subprocess

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True,
                    help="形如 /.../math_grpo_phi_advprm_... ，其下直接有 global_step_*/actor")
    ap.add_argument("--start", type=int, default=10)
    ap.add_argument("--end", type=int, default=100)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--hf_upload_path", default="", help="可选：上传到 HF 的 repo id")
    ap.add_argument("--dry_run", action="store_true", help="仅打印命令不执行")
    args = ap.parse_args()

    if not os.path.isdir(args.root_dir):
        raise SystemExit(f"根目录不存在: {args.root_dir}")

    for s in range(args.start, args.end + 1, args.step):
        local_dir = os.path.join(args.root_dir, f"global_step_{s}", "actor")
        if not os.path.isdir(local_dir):
            print(f"⚠️ 跳过：{local_dir} 不存在")
            continue

        cmd = ["python", "model_merger.py", "--local_dir", local_dir]
        if args.hf_upload_path:
            cmd += ["--hf_upload_path", args.hf_upload_path]

        print("🔧 运行：", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)

    print("✅ 全部完成")

if __name__ == "__main__":
    main()


'''python run_merge.py --root_dir /home/vault/b273dd/b273dd15/checkpoints/verl_grpo_phi_advprm_lookahead_prmreward_rawgain_0904/math_grpo_phi_advprm_lookahead_prmreward_rawgain_0904 --start 10 --end 100 --step 10'''
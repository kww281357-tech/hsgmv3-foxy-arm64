# HSGM v3 — Foxy ARM64 / Go2 EDU staging

> New independent port from jkffj54/hsgmv3 @ 3a0a6558a5a485bca7e0793461b9822060b392eb. Original repository unchanged.
> **Read-only deployment foundation, not completed real-robot navigation.** Docker build, ARM64 runtime and robot communication are not yet verified.

See [Foxy ARM64 Go2 build, network and limitations](docs/FOXY_ARM64_GO2.md).

# HSGM v3

HSGM v3 独立代码仓库，用于分支开发和 agent 协作。初始源码来自本机现有 v3 工作目录；不继承上层仓库历史。原始文件哈希记录于 `docs/SOURCE_SNAPSHOT.json`。

## 仓库内容

- `src/`：v3 导航、映射、记忆、分割与实验入口。
- `config/`：运行配置。
- `tests/`：现有 v3 回归测试，已适配独立仓库路径。
- `AGENTS.md`：分支开发、验证、自动提交与推送规则。
- `scripts/check_repo.py`：语法和提交内容检查。
- `requirements.txt`、`LICENSE`：沿用原项目依赖清单及 Apache-2.0 许可证。

本仓库不包含数据集、权重、实验结果、日志、密钥和旧版本。私有仓库设置不会改变原项目许可证义务。

## 本地运行

需要准备兼容的 Python、CUDA、Habitat-Sim / Habitat-Lab 和 requirements.txt 中的依赖。依赖清单沿用原项目，未通过全新环境安装验证。

1. 在本机准备 `data/`，包含 `scene_datasets/mp3d`、R2R-CE / RxR-CE 等数据。
2. 准备 `ckpt/yoloe-26l-seg.pt`。当前分割初始化还可能使用 `mobileclip2_b.ts`，请预先准备本地缓存；缺失时上游库可能联网下载。
3. 数据和权重目录可以使用本地软链接；这些路径已加入 .gitignore。
4. 复制 `.env.example` 为 `.env` 并填入本地凭据，然后显式加载环境变量。

```bash
set -a
source .env
set +a
cd src
python run_experiments.py --task r2r --config ../config/vlnce_test.yaml \
  --model_name "$OPENAI_MODEL" --begin_idx 0 --end_idx 1 --max_steps 100
```

上面是运行示例，上传仓库过程中不启动仿真实验。

## 验证

```bash
python scripts/check_repo.py
python scripts/check_repo.py --offline-tests
git config --local core.hooksPath .githooks
```

离线回归需要 NumPy；缺少 OpenCV 时部分渲染检查会跳过。完整 mapper 缓存测试需要额外依赖和权重，需单独运行。提交钩子检查整个 Git 暂存快照的文件类型、大小、语法和常见凭据格式；它不能识别所有形式的密钥，提交前仍需检查 diff。

初始快照的离线回归存在已知失败，原目录与独立仓库结果一致；详情见 `docs/INITIAL_VALIDATION.md`。代码上传成功不表示导航功能已通过全部测试。

## Agent 与并行开发

后续直接在这个仓库或其 worktree 中开发。每项任务使用自己的分支，完成相关验证后自动 commit、push 并创建或更新 PR。`main` 用来保存已合并的稳定版本。

```bash
git fetch origin
git worktree add -b feature/navigation ../hsgmv3-navigation origin/main
git -C ../hsgmv3-navigation config --local core.hooksPath .githooks
```

新克隆的仓库应重新配置提交钩子，并确保 `gh auth status` 正常。GitHub 网页登录与本机 Git 授权是分开的，本机 Git 通过 GitHub CLI 登录后才能持续推送。

在新聊天里可以说：“在 hsgmv3 仓库里完成这个任务，遵守 AGENTS.md，验证后提交并推送任务分支。”

## 访问范围

远程仓库使用个人账号下的 Private 仓库，不邀请其他协作者。账号所有者及其授权的工具可按授予的权限访问；不要将仓库改为 Public 或授予不需要的访问权限。


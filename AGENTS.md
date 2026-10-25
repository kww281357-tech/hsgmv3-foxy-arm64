# HSGM v3 开发规则

## 仓库与数据
- 这是 HSGM v3 独立代码仓库。直接在本仓库或其 worktree 修改代码。
- GitHub 仓库必须保持 Private，不添加协作者，不扩大访问权限。
- 只提交源码、必要配置、测试、依赖清单和文档。遵守 .gitignore。
- 数据、模型权重、实验结果、日志、备份、.env 和密钥只保存在本机。
- 不把本机绝对路径或凭据硬编码进代码。数据路径由配置或本地软链接提供。

## 并行开发
- 每个任务使用独立分支（如 feature/navigation、fix/recovery）。
- 多个 agent 同时工作时，每个任务使用独立 git worktree。
- 开始前检查 git status、当前分支与远程状态；保护用户及其他任务的未提交修改。
- 不重置、覆盖其他任务的工作，不强制推送，不自动合并 main。
- 并行改代码不代表可以并行运行 GPU/仿真实验；不要停止现有实验或抢占其资源。

## 完成任务后同步 GitHub
- 用户已授权：每次完成代码任务和相关验证后，自动 commit 并 push 到该任务分支，无需再次询问。
- 先检查 diff，运行 python scripts/check_repo.py；根据改动执行必要测试。
- 用明确的文件路径 git add，只提交本任务改动；写清晰的 commit message。
- 若没有改动，不创建空提交。已有本任务未推送提交应正常推送。
- 推送前核实 origin 指向本项目的私有仓库。使用 git push -u origin HEAD。
- 创建或更新该任务的 Pull Request；只有用户明确要求时才合并 main。
- 验证失败时先定位；不得引入新的失败。初始快照已有失败见 docs/INITIAL_VALIDATION.md；不得隐藏、跳过或把它们表述为通过，也不要为无关任务擅自修改导航语义。
- 身份验证、网络或权限阻止推送时，保留本地工作并明确报告，不能声称已同步。
- 最终回复包含改动摘要、验证结果、分支、commit ID、推送结果及 PR 链接（如有）。

## 验证
- python scripts/check_repo.py：源码语法检查，不导入模型或启动实验。
- python scripts/check_repo.py --staged：额外检查 Git 暂存内容中的非代码文件和常见凭据。
- python scripts/check_repo.py --offline-tests：运行四组轻量离线回归，需要 NumPy。
- tests/test_v3_path_planning_cache.py 会导入完整 mapper 及分割模型；只在依赖和本地权重准备好后单独运行，避免自动下载或占用 GPU。
- 初始化本地提交检查：git config --local core.hooksPath .githooks。


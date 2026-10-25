# 游戏人物生产代码：采用 v7，源代码专用交付

此目录对应已存在的 `adult_neutral / adult_smile / adult_cry` 三个游戏人物静态 GLB。**不包含模型、照片、权重，也不会自动下载或生成。** 新人物优化已暂停。不要把研究环境依赖装入 Go2 / ROS / Python 3.8 环境。

## 可复现范围与缺失输入

采用 Blender **4.3.2 CPU**；脚本使用其内置 `bpy`、`mathutils`、`numpy`。纯 Python 拓扑检查需要独立 Python + NumPy。没有新GPU/API要求。

最短重建链是既有 `v6/editable_character_v6.blend` + v6 object config → `pipeline/refine_v7.py` → 三表情静态 GLB。外部文件及 SHA256 见 `external_assets.json`。它们在此前项目的 v7 editable-and-code 交付包内，但没有假称存在的公共下载地址；本 GitHub 源码提交不携带这些大文件。缺少它们时准备器明确失败。

历史中保留 v4→v5→v6 代码，但 v1–v3 初始生成代码未在恢复的工作区发现。因此不能声称仅下载官方 MPFB 就能逐字节重建这个已定制角色。官方底座来源、具体组件和授权见 `licenses/ASSET_PROVENANCE.txt`。

```bash
# 只验证输入，不生成、不复制模型
python prepare_rebuild.py --asset-root /path/to/existing/game_character_refine_v5
# 在全新目录建立隔离工作区，拒绝覆盖旧数据；仍然不会运行 Blender
python prepare_rebuild.py --asset-root /path/to/existing/game_character_refine_v5 --work /path/to/NEW_workdir
# 以下为将来明确需要重建时手动执行；此次上传任务未执行
blender -b --disable-autoexec --threads 4 --python /path/to/NEW_workdir/refine_v7.py
```

重建后在该工作区分别运行 `validate_v7_glbs.py`（Blender GLB 回读）、`validate_v7_scope.py`（原网格/形状键/材质未变）、`check_hand_geometry_v7.py`（限定手部表面检查）、`validate_exported_topology_v7.py`（Python实际导出索引一致）、`validate_v7_lower_body.py`（Python三表情下半身位置一致）。渲染脚本仅供将来显式审查，非默认流程。

## 采用版的行为

- 同一人物、同一双臂下垂姿态；v7调整手腕/指骨、拇指与食指内收。
- smile联动嘴角、脸颊、眼睑；cry是悲伤表情，不生成眼泪。并不声称照片级逼真。
- 每个表情先求值再固定三角化，避免表情间四边形自动分割不同。输出静态mesh，不是GLB运行时动画。
- `config/`保留 meters、Y-up、Habitat import front=-Z、feet-origin。朝向用actor yaw，不能再翻转几何。
- `is_collidable=false`。历史报告不证明导航安全；此源代码交付没有运行新的Habitat验证。运行时是否采用由导航模块独立确认。
- `evidence/`是旧资产的历史检查结果及hash，不是此次重新生成的结果。不能将有限BVH表面相交检查当成全身/体积碰撞认证。

## 许可证与隐私

官方 MPFB 程序代码 **GPL-3.0-or-later**，它的指定官方底座/rig/target/texture数据 **CC0-1.0**；不能统称整个MPFB为CC0。这里的已署名自写适配/检查脚本贡献沿用 **CC0-1.0**。没有复制MPFB程序逻辑到这些脚本；保留GPL文本用于说明上游边界，不改变自写脚本授权。

不上传用户参考照片、SMPL/FLAME文件、HumanNOVA权重、登录信息、个人路径的私有资料或大模型。官方资产包里的其他第三方资源不得自动继承授权。

## 历史实验，不是默认路线

`history/v4/`保存早期二次开发代码；`history/experiments/`保留v5/v6构建和v8–v10局部实验。它们依赖各自未包含的中间blend/资产和目录，**不是一键完整重建入口**。v8笑容、v9材质仅有限改善；v9 C灰蒙角膜、E发壳和v10 F假发壳被否决。v10 H睫毛+牙齿内部间隙修复只局部改善，不替代采用的v7三表情。未附源照片或未验证资产，不默认运行实验。

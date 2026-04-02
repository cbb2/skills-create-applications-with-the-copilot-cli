# 在 VSCode 中配置 Claude Code Skills 与 MCP

> 适用版本: VSCode 1.90+，Claude Code CLI 最新版，Node.js ≥ 18

---

## 目录

1. [前置条件](#1-前置条件)
2. [安装 Claude Code CLI](#2-安装-claude-code-cli)
3. [在 VSCode 中打开集成终端](#3-在-vscode-中打开集成终端)
4. [配置 MCP 服务器](#4-配置-mcp-服务器)
   - 4.1 [项目级 MCP 配置（推荐）](#41-项目级-mcp-配置推荐)
   - 4.2 [全局 MCP 配置](#42-全局-mcp-配置)
   - 4.3 [验证 MCP 服务器状态](#43-验证-mcp-服务器状态)
5. [配置 Claude Code Skills](#5-配置-claude-code-skills)
   - 5.1 [Skills 文件结构](#51-skills-文件结构)
   - 5.2 [本项目已有的三个 Skills](#52-本项目已有的三个-skills)
   - 5.3 [手动加载 / 重新加载 Skills](#53-手动加载--重新加载-skills)
   - 5.4 [编写自定义 Skill](#54-编写自定义-skill)
6. [VSCode 推荐插件与工作区设置](#6-vscode-推荐插件与工作区设置)
7. [完整使用示例](#7-完整使用示例)
8. [常见问题排查](#8-常见问题排查)

---

## 1. 前置条件

在开始之前，请确保本机已安装以下软件：

| 软件 | 最低版本 | 检查命令 | 下载链接 |
|------|---------|---------|---------|
| Node.js | 18.0 | `node -v` | https://nodejs.org |
| npm | 9.0 | `npm -v` | 随 Node.js 一起安装 |
| Git | 2.30 | `git --version` | https://git-scm.com |
| Python | 3.9 | `python --version` | https://python.org |
| VSCode | 1.90 | 帮助 → 关于 | https://code.visualstudio.com |

---

## 2. 安装 Claude Code CLI

Claude Code 以 npm 包形式分发，全局安装后即可在任意项目中使用：

```bash
# 全局安装 Claude Code CLI
npm install -g @anthropic-ai/claude-code

# 验证安装
claude --version

# 首次登录（需要 Anthropic API Key 或 Claude.ai 账号）
claude login
```

> **提示**：若在公司网络下安装失败，尝试设置 npm 镜像：
> ```bash
> npm config set registry https://registry.npmmirror.com
> ```

---

## 3. 在 VSCode 中打开集成终端

所有 Claude Code 操作都通过终端完成，推荐在 VSCode 内使用集成终端：

1. 用 VSCode 打开本项目根目录：
   ```
   File → Open Folder → 选择 skills-create-applications-with-the-copilot-cli/
   ```

2. 打开集成终端：
   - 快捷键：`` Ctrl+` `` (Windows/Linux) 或 `` Cmd+` `` (macOS)
   - 菜单：Terminal → New Terminal

3. 在终端中启动 Claude Code（交互模式）：
   ```bash
   claude
   ```
   看到 `>` 提示符即代表 Claude Code 已就绪。

---

## 4. 配置 MCP 服务器

MCP（Model Context Protocol）服务器扩展了 Claude Code 的能力边界，使其能够读写文件、执行网络请求、进行结构化推理等。

### 4.1 项目级 MCP 配置（推荐）

**项目级配置** 写在项目根目录的 `.mcp.json` 文件中，只对该项目生效，方便团队共享。

本项目的 `.mcp.json` 已预配置三个服务器，路径：`.mcp.json`

```jsonc
{
  "mcpServers": {

    // ① 文件系统服务器：安全读写项目内文件
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem@2026.1.14",
        "."          // ← "." 表示只允许访问当前目录（安全沙箱）
      ]
    },

    // ② 顺序思维服务器：结构化多步推理
    "sequential-thinking": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-sequential-thinking@2025.12.18"
      ]
    },

    // ③ 网络获取服务器：抓取 arXiv 论文、PyTorch 文档
    "fetch": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-fetch"
      ]
    }

  }
}
```

**Claude Code 自动发现规则**：
- 启动时自动扫描当前工作目录及其所有父目录中的 `.mcp.json`
- 距离当前目录最近的 `.mcp.json` 优先级最高
- **无需任何额外命令**，打开项目目录后启动 `claude` 即自动加载

### 4.2 全局 MCP 配置

若需要对所有项目都生效的 MCP 服务器（如个人常用工具），编辑全局配置文件：

| 操作系统 | 全局配置文件路径 |
|---------|----------------|
| macOS / Linux | `~/.claude/mcp.json` |
| Windows | `%APPDATA%\Claude\mcp.json` |

```bash
# 快速打开全局配置（macOS/Linux）
code ~/.claude/mcp.json

# 或通过 Claude Code 命令行管理
claude mcp add my-server --command "npx -y @my/server"
claude mcp list
claude mcp remove my-server
```

也可以在 VSCode 集成终端中直接用 `/mcp` 命令管理：

```
# 在 claude 交互模式中输入:
/mcp show             # 查看当前所有 MCP 服务器及状态
/mcp add server-name  # 添加新服务器
/mcp disable fetch    # 临时禁用某个服务器
/mcp enable fetch     # 重新启用
```

### 4.3 验证 MCP 服务器状态

启动 Claude Code 后，用以下命令验证服务器是否正常运行：

```
# 在 claude 交互模式中：
/mcp show
```

正常输出示例：
```
MCP Servers (3 connected):
  ● filesystem        [connected]  项目文件读写
  ● sequential-thinking [connected] 结构化推理
  ● fetch             [connected]  网络资源获取
```

若某服务器显示 `[error]`，通常是 Node.js 版本过低或网络问题：
```bash
# 检查 Node.js 版本（需 >= 18）
node -v

# 手动测试 MCP 服务器是否能启动
npx -y @modelcontextprotocol/server-filesystem .
```

---

## 5. 配置 Claude Code Skills

Skills 是存放在项目 `.claude/skills/` 目录下的 Markdown 文件，为 Claude Code 提供领域专项知识和行为规范。

### 5.1 Skills 文件结构

每个 Skill 是一个独立目录，包含一个 `SKILL.md` 文件：

```
.claude/
└── skills/
    ├── deep-learning-debugger/
    │   └── SKILL.md          ← Skill 定义文件
    ├── pytorch-code-reviewer/
    │   └── SKILL.md
    └── vit-explainer/
        └── SKILL.md
```

`SKILL.md` 文件格式（YAML frontmatter + Markdown 正文）：

```markdown
---
name: skill-name              # 技能唯一标识符（kebab-case）
description: |                # 触发描述：什么情况下自动激活该技能
  一段描述此技能的文字，
  Claude Code 根据此描述决定何时调用该技能。
---

# 技能正文

这里写详细的指令、决策树、示例等内容。
Claude Code 激活此技能后会完整阅读这些内容并据此行动。
```

### 5.2 本项目已有的三个 Skills

| Skill 名称 | 目录 | 自动触发场景 |
|-----------|------|------------|
| `deep-learning-debugger` | `.claude/skills/deep-learning-debugger/` | 遇到 loss 不收敛、梯度爆炸/消失、维度不匹配、CUDA OOM 等训练问题 |
| `pytorch-code-reviewer` | `.claude/skills/pytorch-code-reviewer/` | 提交新的 PyTorch/ViT 代码片段需要审查时 |
| `vit-explainer` | `.claude/skills/vit-explainer/` | 询问 ViT 架构原理、注意力机制数学、模块实现细节时 |

在 Claude Code 中**手动调用** Skill 的方式：

```
# 在 claude 交互模式中，用 /skills 命令管理：
/skills list                    # 列出所有已加载的 Skills
/skills info vit-explainer      # 查看某个 Skill 的详细信息
/skills reload                  # 重新加载所有 Skills（修改后刷新）

# 直接在对话中触发（Claude 会自动识别并调用对应 Skill）：
> 帮我解释一下 MultiHeadSelfAttention 的注意力缩放因子为什么是 1/√d
  ↳ Claude 自动激活 vit-explainer Skill

> 我的训练 loss 在第 3 个 epoch 变成了 NaN，怎么排查？
  ↳ Claude 自动激活 deep-learning-debugger Skill
```

### 5.3 手动加载 / 重新加载 Skills

Claude Code **每次启动时自动扫描** `.claude/skills/` 目录，无需手动导入。  
修改 `SKILL.md` 文件后，在交互模式中输入：

```
/skills reload
```

即可热重载，无需重启 Claude Code。

> **注意**：Skills 只在 Claude Code **交互模式**（`claude` 命令）中生效，  
> 不适用于 `claude --print` 单次查询模式。

### 5.4 编写自定义 Skill

以下是一个为本项目定制的 **训练监控 Skill** 示例：

```bash
# 在终端中创建新 Skill 目录
mkdir -p .claude/skills/training-monitor
```

创建文件 `.claude/skills/training-monitor/SKILL.md`：

```markdown
---
name: training-monitor
description: |
  当用户询问如何监控 ViT 模型训练过程、如何解读 loss/accuracy 曲线、
  如何使用 TensorBoard 可视化训练日志时自动激活。
---

# 训练监控专家

## 核心能力

1. **实时监控**: 读取 `runs/` 目录下的 TensorBoard 日志
2. **曲线分析**: 分析 train/val loss 走势，判断过拟合/欠拟合
3. **超参建议**: 根据曲线形态给出学习率、batch size 调整建议

## 常用命令

```bash
# 启动 TensorBoard 可视化
tensorboard --logdir runs/

# 在浏览器中打开: http://localhost:6006
```

## 判断训练状态

| 现象 | 诊断 | 建议 |
|------|------|------|
| train_loss 下降，val_loss 上升 | 过拟合 | 增大 dropout，使用 DropPath |
| 两者都不下降 | 欠拟合/lr 过小 | 增大 lr，检查数据管道 |
| loss 突然变 NaN | 梯度爆炸 | 降低 lr，添加 gradient clipping |
```

---

## 6. VSCode 推荐插件与工作区设置

### 推荐安装的 VSCode 插件

在 VSCode 中按 `Ctrl+Shift+X` 打开插件市场，搜索并安装：

| 插件名 | 插件 ID | 用途 |
|--------|---------|------|
| Python | `ms-python.python` | Python 语法支持、调试 |
| Pylance | `ms-python.vscode-pylance` | 类型检查、智能补全 |
| Jupyter | `ms-toolsai.jupyter` | 交互式 Notebook 支持 |
| GitLens | `eamodio.gitlens` | Git 历史、blame 等 |
| YAML | `redhat.vscode-yaml` | YAML/frontmatter 语法高亮 |
| Markdown All in One | `yzhang.markdown-all-in-one` | Markdown 预览和编辑 |

### 一键安装插件（命令行）

```bash
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-toolsai.jupyter
code --install-extension eamodio.gitlens
code --install-extension yzhang.markdown-all-in-one
```

### 推荐的工作区设置

在项目根目录创建或编辑 `.vscode/settings.json`，添加以下配置：

```jsonc
{
  // Python 解释器（根据你的虚拟环境路径修改）
  "python.defaultInterpreterPath": ".venv/bin/python",

  // 保存时自动格式化
  "editor.formatOnSave": true,
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter"
  },

  // 终端默认 Shell
  "terminal.integrated.defaultProfile.linux": "bash",
  "terminal.integrated.defaultProfile.osx": "zsh",

  // 文件关联（让 SKILL.md 获得正确的 Markdown 语法高亮）
  "files.associations": {
    "SKILL.md": "markdown",
    "CLAUDE.md": "markdown",
    ".mcp.json": "jsonc"
  },

  // 资源管理器中隐藏无关文件
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    "**/runs": false,        // 保留 TensorBoard 日志目录
    "**/checkpoints": false  // 保留训练 checkpoint 目录
  }
}
```

### 配置 Python 虚拟环境（推荐）

```bash
# 在项目根目录创建虚拟环境
python -m venv .venv

# 激活虚拟环境
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows PowerShell

# 安装项目依赖
pip install -r vision_transformer/requirements.txt

# 在 VSCode 中选择该解释器：
# Ctrl+Shift+P → "Python: Select Interpreter" → 选择 .venv/bin/python
```

---

## 7. 完整使用示例

以下是在配置好环境后，一次完整的 Claude Code + Skills + MCP 工作流：

```bash
# Step 1: 进入项目目录
cd skills-create-applications-with-the-copilot-cli

# Step 2: 启动 Claude Code（自动加载 .mcp.json 和 Skills）
claude

# Step 3: 在交互模式中工作
> /mcp show              # 确认 MCP 服务器已连接
> /skills list           # 确认 3 个 Skills 已加载

# Step 4: 提问或请求任务（Claude 会自动调用合适的 Skill 和 MCP）
> 帮我分析 vit.py 中 MultiHeadSelfAttention 的实现，和原论文有哪些不同？
  ↳ Claude 调用 filesystem MCP 读取 vit.py
  ↳ 激活 vit-explainer Skill 进行深度解析

> 我在训练 ViT-Tiny 时 loss 变成了 NaN，log 如下：[粘贴日志]
  ↳ 激活 deep-learning-debugger Skill
  ↳ 调用 sequential-thinking MCP 进行结构化诊断

> 帮我审查这段新写的 PatchEmbedding 代码：[粘贴代码]
  ↳ 激活 pytorch-code-reviewer Skill

> 帮我获取 ViT 原论文的摘要
  ↳ 调用 fetch MCP 获取 https://arxiv.org/abs/2010.11929
```

---

## 8. 常见问题排查

### Q1: Claude Code 启动时显示 "MCP server failed to start"

**原因**: Node.js 版本过低，或 npx 无法访问 npm registry。

```bash
# 检查 Node.js 版本（需要 >= 18）
node -v

# 升级 Node.js（使用 nvm）
nvm install 20
nvm use 20

# 测试 npm 连通性
npx -y @modelcontextprotocol/server-filesystem --version
```

### Q2: Skills 没有被自动激活

**原因**: `.claude/skills/` 目录结构不正确，或 `SKILL.md` 的 frontmatter 格式有误。

```bash
# 检查目录结构
ls -la .claude/skills/

# 验证 SKILL.md 的 YAML frontmatter
# 确保 --- 前后没有多余空格，name 字段不能为空
head -5 .claude/skills/vit-explainer/SKILL.md

# 手动重新加载
# 在 claude 交互模式中输入：
/skills reload
```

### Q3: `filesystem` MCP 无法读取某些文件

**原因**: 文件不在 `.mcp.json` 中配置的根目录 `"."` 之内，或文件被 `.gitignore` 排除（MCP 遵守访问限制）。

```bash
# 确认文件路径在项目目录内
realpath <file>

# 临时扩展文件系统访问范围（编辑 .mcp.json）
# 将 "." 改为绝对路径或上层目录
```

### Q4: 在 Windows 上 Claude Code 运行异常

推荐使用 **WSL 2 (Windows Subsystem for Linux)**：

```powershell
# 安装 WSL 2（管理员 PowerShell）
wsl --install

# 在 VSCode 中安装 WSL 插件
code --install-extension ms-vscode-remote.remote-wsl

# 通过 WSL 打开项目
wsl
cd /home/<your-user>/projects/skills-create-applications-with-the-copilot-cli
claude
```

### Q5: 如何确认某个 MCP 工具被调用了

在 Claude Code 交互模式下，每次 MCP 工具被调用时会显示：

```
[Tool: filesystem/read_file] vision_transformer/vit.py
[Tool: fetch/fetch] https://arxiv.org/abs/2010.11929
```

若没有看到这些行，说明 Claude 没有调用 MCP（可能直接用了训练数据回答）。  
可以明确要求：`"请使用 filesystem MCP 读取 vit.py 文件后再回答"`。

---

## 参考资料

- [Claude Code 官方文档](https://docs.anthropic.com/en/docs/claude-code)
- [Claude Code Skills 指南](https://docs.anthropic.com/en/docs/claude-code/skills)
- [MCP 协议规范](https://modelcontextprotocol.io)
- [可用 MCP 服务器列表](https://github.com/modelcontextprotocol/servers)
- [本项目 .mcp.json 配置](./../.mcp.json)
- [本项目 Skills 目录](./../.claude/skills/)

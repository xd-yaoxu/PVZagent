# PVZagent — 只读实时建议助手 v1.0

面向原版《植物大战僵尸》（Plants vs. Zombies，重点适配 3-7 泳池十卡槽场景）的 **纯视觉、严格只读** 实时战术伴随侧栏。

> **核心安全原则**：0 游戏键鼠输入 · 0 游戏内存读写 · 0 截图像素落盘 · 默认纯本地 0 外网请求 · API 密钥仅本机 Windows DPAPI 加密保存。

---

## 核心特性

- **原生 WGC 纯内存采集**：基于 Windows Graphics Capture（C++ 原生 DLL + 独立 MTA 线程）直接从显卡缓冲读取窗口客户区，不读取游戏内存，使用后立即 `fill(0)` 清零内存，磁盘零图像残留。
- **单帧 `< 0.8s` 快速视觉感知**：第 1 帧使用防遮挡 SIFT 完成窗口几何对齐，后续帧使用缓存仿射 + 7 块非数字 HUD/棋盘 NCC 锚点校验（~8.7 ms）配合单遍冻结模型感知（~760 ms/帧），输出【卡片】/【威胁】/【位置】三通道真实证据，看不清即如实标为未知。
- **Fail-Closed 失效应撤与防跳动单侧栏**：
  - 独立 2.0s 操作系统看门狗线程：切窗、游戏暂停、选卡界面或超时立即撤下当前建议并释放 WGC，回到游戏自动重连，禁止旧代结果复活；
  - 固定高度卡片布局（建议历史 **0 px 跳动**）+ 3.5s 可读停留稳定器（遇靠左紧急威胁 `x <= 200` 或关注行清空立即抢占切换）；
  - 同一侧栏内置「建议」「时间轴（四阶段动态目标）」「设置」三页，支持一键折叠为窄条。
- **默认纯本地 + 可选异步 LLM 教练**：
  - 默认关闭外网请求；可在「设置」页自主填写 OpenAI 兼容 `Base URL`（强制 HTTPS / 本机回环）与 `API Key`，填完后自动拉取 `GET /models` 模型列表并支持文本连通测试；
  - API 密钥通过 Windows DPAPI（`CryptProtectData`）加密存储在仓库外的 `%LOCALAPPDATA%\PVZagent\readonly_advisor_v1_settings.json`，绝不明文落盘或写入仓库；
  - LLM 返回的战术建议必须通过当前 `<= 2.0s` 新鲜帧复核才会显示，且后续每帧持续复验、局势变化即撤下。

---

## 运行环境

- **操作系统**：Windows 10 (1903+) / Windows 11（64 位）
- **Python**：Python 3.11+（含自带 `tkinter`）
- **目标游戏**：原版窗口化《植物大战僵尸》（支持 `800×600` 或 `1200×900` 客户区，进程名匹配 `PVZ-Windowed.exe` / `PlantsVsZombies.exe` / `popcapgame1.exe`；也可通过环境变量 `PVZ_GAME_EXE` 精确指定任意路径）

---

## 快速上手

### 1. 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 运行离线自检（验证冻结基线与原生 DLL）

```powershell
python scripts/pvz_readonly_advisor.py --self-test
```

预期返回 `"status": "SELF_TEST_PASS"`（`input_sent: false`、`stop_latched: true`）。

### 3. 启动只读实时建议侧栏

双击根目录的 **`启动只读建议助手.bat`**，或运行：

```powershell
pythonw scripts/pvz_readonly_advisor.py
```

1. 点击侧栏顶部 **「选窗(5秒)」**，在倒计时内切换到原版 PVZ 游戏窗口；
2. 点击 **「开始监测」** 后切回游戏即可开始实时建议与时间轴追踪；
3. 如需启用异步 LLM，切到 **「设置」** 填写 `Base URL` 与 `API Key`（自动拉取模型），再勾选「启用后台异步 LLM 战术建议」。

> 可选：如需自定义游戏主程序精确路径，启动前设置环境变量 `PVZ_GAME_EXE=你的完整路径\PlantsVsZombies.exe`。

---

## 逻辑架构

```text
原版 PVZ 窗口 (MainWindow, 800x600 / 1200x900)
   │  [L1] OriginalPvzWindowSelector + pvz_wgc_continuous.dll (WGC 内存直采, 边缘 Alpha 门禁)
   ▼
800x600 标准 RGB 内存帧（用完即 fill(0) 清零）
   │  [L2] FastGeometryNormalizer (缓存仿射 + 7 块 NCC) → FastSinglePassPerception (单遍冻结感知)
   ▼
结构化观测状态（阳光 / 10 卡槽 / 僵尸头 / 车辆 / 54 格 / 地面阳光）
   ├──► [L3 主通道] 本地即时策略 + 三通道真实证据 + PhaseTracker (四阶段时间轴)
   └──► [L3 可选]   AsyncLlmCoach (默认关闭, 后台非阻塞请求, 逐帧 <=2.0s 复验撤回)
   │  [L4] 2.0s OS 看门狗 + 3.5s 停留稳定器 + 切窗/暂停自动撤回与重连
   ▼
[L5] ReadonlyAdvisorSidebarApp (单右侧可折叠固定高度侧栏 + Windows DPAPI 加密配置)
```

---

## 从源码重新编译原生 WGC DLL（可选）

仓库内已附带 64 位 `src/pvz_agent/readonly_advisor_v1/native/build/pvz_wgc_continuous.dll`。如需自行验证并从 C++ 源码重编，在 **x64 Native Tools Command Prompt for VS 2022+** 中运行：

```bat
cl /nologo /utf-8 /std:c++20 /EHsc /O2 /LD ^
  /I src\pvz_agent\readonly_advisor_v1\native ^
  src\pvz_agent\readonly_advisor_v1\native\pvz_wgc_continuous.cpp ^
  /Fo:src\pvz_agent\readonly_advisor_v1\native\build\pvz_wgc_continuous.obj ^
  /Fe:src\pvz_agent\readonly_advisor_v1\native\build\pvz_wgc_continuous.dll ^
  d3d11.lib dxgi.lib dwmapi.lib runtimeobject.lib user32.lib windowsapp.lib
```

---

## 运行测试

```powershell
python -m unittest tests/test_readonly_advisor_v1.py -v
```

---

## 开源打包导出

如在包含大量历史实验数据的开发工作区中准备开源发布，可运行：

```powershell
python scripts/export_opensource_release.py
```

脚本会在 `deliverables/PVZagent-opensource-v1.0/` 生成一份仅包含核心代码、运行必需冻结基线与文档的干净发布目录（约 ~19 MB，不含历史录屏、快捷方式、第三方二进制与私人路径）。

---

## 免责声明与许可

- 本项目仅供计算机视觉、实时人机协作与工程安全边界研究学习使用，**不包含任何游戏本体或第三方商业素材**。
- 本版本严格只读，不读取或修改游戏进程内存，不向游戏发送任何模拟输入。
- 如需配合窗口缩放工具（如 Magpie），请从对应开源项目官方页面自行下载。
- 本项目基于 [MIT License](LICENSE) 开源。

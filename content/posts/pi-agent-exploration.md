+++
date = '2026-03-01T16:30:00+08:00'
draft = true
title = '探秘：OpenClaw起源—Pi Agent'
description = '深入探索 Pi Agent 源码，揭开 OpenClaw 的神秘面纱'
tags = ['OpenClaw', 'Pi Agent', 'AI Agent', '源码解析']
categories = ['技术探索']
+++

# 探秘：OpenClaw 起源 — Pi Agent

> 这是一次充满好奇心的源码探索之旅。本文记录了我对 Pi Agent 源码的完整分析过程，希望能给同样对 AI Agent 感兴趣的朋友一些启发。

## 缘起

一切源于一个问题：**OpenClaw 是如何工作的？**

在深入使用 OpenClaw 的过程中，我发现它与一个叫 **Pi Agent**（现在叫 `pi-coding-agent`）的项目有着千丝万缕的联系。README 中明确提到：

> See [openclaw/openclaw](https://github.com/openclaw/openclaw) for a real-world SDK integration.

这勾起了我的好奇心——OpenClaw 究竟是如何集成 Pi Agent 的？Pi Agent 本身又有着怎样的架构设计？让我们一探究竟。

## 初识 Pi Monorepo

Pi 是一个开源的编程 Agent 项目，由 badlogic 开发维护（也就是 Mario Zechner，libgdx 的作者）。

```bash
git clone https://github.com/badlogic/pi-mono.git
```

克隆下来后，我发现这是一个典型的 **Monorepo** 结构：

| 包 | 描述 |
|---|---|
| **coding-agent** | 交互式编程 Agent CLI（主包） |
| **agent-core** | Agent 运行时，支持 tool calling 和状态管理 |
| **ai** | 统一的多 provider LLM API（OpenAI/Anthropic/Google 等） |
| **mom** | Slack bot，把消息委托给 coding agent |
| **tui** | 终端 UI 库（差分渲染） |
| **web-ui** | AI 聊天界面的 Web 组件 |
| **pods** | 管理 vLLM 部署的 CLI |

每个包各司其职，通过 npm 依赖串联起整个系统。

## 四种运行模式

Pi Agent 支持四种运行模式，每种模式面向不同的使用场景：

### 1. Interactive Mode（交互模式）

日常对话式编程的主要入口。启动命令就是简单的 `pi`，会进入一个完整的 TUI（终端用户界面）：

- 消息列表 + 输入编辑器 + 底部状态栏
- 支持键盘快捷键、文件引用（@）、图片粘贴
- 可以实时查看工具执行、切换模型

### 2. Print Mode（单次模式）

用于一次性 prompt + 输出结果 → 退出：

```bash
pi -p "prompt"           # 输出纯文本结果
pi --mode json "prompt"  # 输出 JSON 事件流
```

适合脚本/CI 场景，不维护会话状态。

### 3. RPC Mode（远程过程调用模式）

通过 stdin/stdout 进行 JSON 协议通信：

- 接收 JSON 命令，返回事件和响应
- 支持扩展 UI 交互

**OpenClaw 就是基于这个模式集成 Pi 的**（或者是 SDK 模式）。

### 4. SDK Mode（编程接口）

在代码中直接调用：

```typescript
const { session } = await createAgentSession(options);
```

用于构建自己的 UI 或 Agent 包装器。

## 核心组件解析

### 1. SessionManager — 会话管理的艺术

SessionManager 是 Pi 最吸引我的组件之一。它设计了一套优雅的**树形会话管理系统**：

```
SessionManager
├── 存储：JSONL 文件（每行一个 JSON entry）
├── 结构：树形结构（id + parentId）
├── 当前指针：leafId（指向最新 entry）
└── 持久化在 ~/.pi/agent/sessions/
```

**Entry 类型丰富多样**：

| 类型 | 作用 |
|---|---|
| `message` | 用户/助手消息 |
| `thinking_level_change` | 思考级别变更 |
| `model_change` | 模型变更 |
| `compaction` | 上下文压缩摘要 |
| `branch_summary` | 分支摘要 |
| `label` | 标签（书签） |
| `custom` | 扩展自定义数据 |

**核心操作**：

- **append**：在 leaf 下追加新 entry，leaf 前移
- **branch(entryId)**：移动 leaf 指针到指定 entry，从那里继续
- **resetLeaf()**：回到根之前（重新编辑第一条消息）

这套设计支持：
- ✅ 多分支会话（像 Git 一样）
- ✅ 上下文压缩（防止 token 溢出）
- ✅ 会话持久化（JSONL 格式）

### 2. InteractiveMode — TUI 的实现

InteractiveMode 是交互模式的核心，负责：

- 用户输入处理（键盘、文件引用、图片）
- 事件驱动的 UI 渲染
- 与 AgentSession 的交互

核心流程：

```typescript
async run() {
    await this.init();
    
    // 处理初始消息
    if (initialMessage) {
        await this.session.prompt(initialMessage);
    }
    
    // 主循环
    while (true) {
        const userInput = await this.getUserInput();
        await this.session.prompt(userInput);
    }
}
```

### 3. AgentSession — 业务逻辑层

AgentSession 是**所有运行模式共享的核心类**（interactive/print/RPC），封装了：

- Agent 生命周期管理
- 会话持久化
- 压缩（compaction）
- 分支（fork/tree navigation）
- 扩展系统

核心方法：

| 方法 | 作用 |
|---|---|
| `prompt(text, options)` | 发送用户消息给 Agent |
| `newSession(options)` | 创建新会话 |
| `fork(entryId)` | 从历史消息分支创建新会话 |
| `navigateTree(targetId)` | 在当前会话内切换分支 |
| `compact(customInstructions)` | 手动压缩上下文 |

## 事件驱动架构

Pi Agent 采用**事件驱动架构**，核心是 AgentCore 发出的事件流：

```
agent_start → turn_start → message_start → message_update 
→ tool_execution_* → message_end → turn_end → agent_end
```

每一步都会发出事件，UI 层（InteractiveMode）监听这些事件并实时更新渲染：

- `agent_start` → 显示加载动画
- `message_update` → 流式更新内容
- `tool_execution_*` → 渲染工具执行过程
- `message_end` → 写入 SessionManager

## 深入 agent-core

在继续深入探索后，我把目光投向了 **agent-core** 包——这是 Pi Agent 的核心心脏。它位于 `packages/agent/src/` 目录下，仅仅用了约 1500 行代码就实现了一个完整的 Agent 运行时。

### 核心文件结构

| 文件 | 行数 | 核心功能 |
|------|------|----------|
| `types.ts` | 194 | 类型定义（AgentMessage、AgentTool、AgentEvent 等） |
| `agent.ts` | 559 | Agent 类（状态管理 + 对外 API） |
| `agent-loop.ts` | 417 | **Agent 循环逻辑（LLM 调用 + 工具执行）** |
| `proxy.ts` | 340 | 代理工具（浏览器端使用） |

---

### AgentLoopConfig — Agent 的配置核心

`AgentLoopConfig` 是配置 Agent 循环的核心接口，它定义了 Agent 与外部世界交互的所有关键点：

```typescript
interface AgentLoopConfig extends SimpleStreamOptions {
    model: Model<any>;                    // LLM 模型
    convertToLlm: (msgs) => Message[];   // 消息转换（核心！）
    
    // 可选字段
    transformContext?: (msgs, signal) => Promise<AgentMessage[]>;  // 上下文预处理
    getApiKey?: (provider) => string;     // 动态 API Key
    getSteeringMessages?: () => AgentMessage[];  // 打断机制
    getFollowUpMessages?: () => AgentMessage[];  // 排队消息
}
```

**核心字段解析**：

1. **`convertToLlm`**：每次 LLM 调用前，把 AgentMessage 转成 LLM 认识的 Message。这是**最核心的字段**！

2. **`transformContext`**：在 `convertToLlm` 之前调用，可以做上下文压缩、注入外部上下文等。

3. **`getApiKey`**：解决 OAuth token 过期问题，每次 LLM 调用前刷新。

4. **`getSteeringMessages`**：用户"打断"Agent 工作时注入新消息。

5. **`getFollowUpMessages`**：Agent 本该停止时，检查是否有排队消息让它继续。

---

### agent-loop — 主循环逻辑

`agent-loop.ts` 是整个 Agent 的**动力引擎**。它包含四个核心函数：

| 函数 | 作用 |
|------|------|
| `agentLoop()` | 带新 prompt 启动循环 |
| `agentLoopContinue()` | 从当前状态继续（用于重试） |
| `runLoop()` | **主循环逻辑**（内外两层循环） |
| `streamAssistantResponse()` | 调用 LLM 并流式返回 |
| `executeToolCalls()` | 执行工具调用 |

#### 双层循环设计

`runLoop()` 采用**双层循环**设计，完美处理了打断和排队的边界情况：

```typescript
while (true) {  // 外层：处理 follow-up
    while (hasMoreToolCalls || pendingMessages.length > 0) {  // 内层：处理 tool calls
        // 1. 处理 pending messages（steering）
        // 2. 调用 LLM
        // 3. 检查 tool_calls
        // 4. 执行工具
        // 5. 检查 steering
    }
    
    // 检查 follow-up
    const followUp = await config.getFollowUpMessages?.();
    if (followUp.length > 0) {
        continue;  // 继续外层
    }
    
    break;  // 退出
}
```

**关键设计点**：
- **Steering（打断）**：工具执行过程中可以检查 steering，有则跳过剩余工具
- **Follow-up（排队）**：Agent 本该停止时，检查是否有排队消息让它继续

---

### 流式处理 — Agent 的"感官系统"

`streamAssistantResponse()` 是 LLM 流式处理的核心，它负责把 LLM 的流式输出转换为 Agent 事件。

#### LLM 流式事件类型

来自 `pi-ai` 的 `AssistantMessageEvent`：

| 事件类型 | 阶段 | 说明 |
|----------|------|------|
| `start` | 开始 | LLM 开始响应，创建空消息 |
| `text_start/delta/end` | 文本 | 文本块的流式输出 |
| `thinking_start/delta/end` | 思考 | 思考（reasoning）流式输出 |
| `toolcall_start/delta/end` | 工具 | 工具调用参数的流式解析 |
| `done` | 完成 | 正常完成 |
| `error` | 错误 | 出错或中断 |

#### 流式处理核心逻辑

```typescript
for await (const event of response) {
    switch (event.type) {
        case "start":
            // 创建空消息，push 到 context
            partialMessage = event.partial;
            context.messages.push(partialMessage);
            stream.push({ type: "message_start", message: {...partialMessage} });
            break;
        
        case "text_delta":
        case "thinking_delta":
        case "toolcall_delta":
            // 流式更新：更新 partialMessage
            partialMessage = event.partial;
            context.messages[最后一条] = partialMessage;
            
            // 发出 message_update 事件
            stream.push({
                type: "message_update",
                assistantMessageEvent: event,
                message: {...partialMessage},
            });
            break;
        
        case "done":
        case "error":
            // 获取最终消息，发送 message_end
            const finalMessage = await response.result();
            stream.push({ type: "message_end", message: finalMessage });
            return finalMessage;
    }
}
```

#### 设计亮点

1. **简洁的快照模式**：每次 delta 事件携带完整状态（`event.partial`），直接替换即可，无需追踪增量。

2. **双向事件流**：原始 LLM 事件 + 包装后的 Agent 事件，UI 层可以看到原始事件（如 `thinking_delta`）。

3. **实时 context 更新**：流式过程中 `context.messages` 始终保持最新，下一次 LLM 调用时已有完整历史。

4. **统一的错误处理**：`done` 和 `error` 共用处理逻辑。

---

## 与 OpenClaw 的关系

经过探索，我发现：

1. **OpenClaw 没有直接使用 Pi 的多分支会话能力** — 它有自己独立的会话管理系统
2. **OpenClaw 可能是通过 SDK 模式或 RPC 模式集成的 Pi**
3. 两者的设计理念相似：事件驱动、分层架构、扩展性

## 总结

这次源码探索让我对 Pi Agent 有了深入的理解：

1. **分层架构清晰**：UI 层 → 业务层 → 核心层 → 存储层
2. **树形会话设计**很有创意，像 Git 一样支持分支
3. **事件驱动**让流式输出和工具执行都能实时渲染
4. **扩展性强**：支持 skills、extensions、prompt templates

如果你对 AI Agent 感兴趣，Pi Agent 是一个很好的学习对象。它的代码质量高，架构设计值得借鉴。

---

*最后，感谢 badlogic 的开源贡献，让我们能有机会深入学习一个生产级 Agent 的实现。*

## 参考资料

- [pi-mono GitHub](https://github.com/badlogic/pi-mono)
- [pi.dev](https://pi.dev)
- [shittycodingagent.ai](https://shittycodingagent.ai)

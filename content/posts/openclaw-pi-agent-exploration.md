---
title: "探秘：OpenClaw集成Pi Agent的方式"
date: 2026-03-05
categories: ["产品探索"]
tags: ["OpenClaw", "Pi Agent", "AI Gateway"]
keywords: ["OpenClaw", "Pi Agent", "集成", "源码分析"]
description: "深入解析OpenClaw如何集成Pi Agent，从消息接收到执行的完整调用链路"
---

# 探秘：OpenClaw集成Pi Agent的方式

## 序言

OpenClaw是一个多通道AI网关，支持多种消息平台的集成。在其架构中，Pi Agent作为核心的AI执行引擎，承担着处理用户对话的重要职责。本文将深入分析OpenClaw如何集成Pi Agent，揭示从消息接收到AI响应的完整技术细节。

## OpenClaw调用链路

OpenClaw采用统一的架构设计，所有消息通道（Discord、Telegram、Signal等）共享同一套`auto-reply`核心模块。这种设计确保了代码复用和行为一致性。

### 整体架构

```
用户消息到达各Channel
         ↓
message-handler (Discord/Signal/Slack/iMessage...)
         ↓
dispatchInboundMessage()  ← 统一入口
         ↓
dispatchReplyFromConfig()
         ↓
runReplyAgent()  ← src/auto-reply/reply/agent-runner.ts
         ↓
runAgentTurnWithFallback()  ← 执行入口
         ↓
    ┌───────────────────────────────────────┐
    │  if isCliProvider                     │
    │      → runCliAgent()                  │  ← CLI模式
    │  else                                │
    │      → runEmbeddedPiAgent()  ← ⚡️ 核心│
    └───────────────────────────────────────┘
```

### 关键分支逻辑

OpenClaw根据不同的Provider类型选择不同的执行路径：

| 模式 | 使用场景 | 实际调用 |
|------|---------|---------|
| **runCliAgent** | 使用外部CLI工具 | `claude` / `codex` 命令行 |
| **runEmbeddedPiAgent** | 使用内置嵌入式Pi Agent | 直接调用LLM API |

`isCliProvider`判断逻辑：
- `claude-cli` - 调用本地`claude`命令
- `codex-cli` - 调用本地`codex`命令
- 用户自定义CLI Backend

## RunEmbeddedPiAgent

`runEmbeddedPiAgent`是OpenClaw集成Pi Agent的核心函数，位于`src/agents/pi-embedded-runner/run.ts`。

### 核心调用链

```
runEmbeddedPiAgent()
         ↓
runEmbeddedAttempt()  ← attempt.ts
         ↓
┌─────────────────────────────────────────────────────────────┐
│  1. 创建Pi Session                                         │
│     createAgentSession({...})                              │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│  2. 配置Stream函数                                         │
│     activeSession.agent.streamFn = ...                       │
│     - streamSimple (默认)                                    │
│     - createOllamaStreamFn (Ollama)                         │
│     - createOpenAIWebSocketStreamFn (OpenAI WS)            │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│  3. 订阅事件流                                              │
│     subscribeEmbeddedPiSession({ session, ... })            │
│     - 处理tool调用                                           │
│     - 处理消息块                                             │
│     - 流式输出响应                                           │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│  4. 执行Prompt  ← ⚡️ 核心！                                │
│     activeSession.prompt(effectivePrompt)                  │
└─────────────────────────────────────────────────────────────┘
```

### 关键参数

在创建`createAgentSession`时，OpenClaw传入以下关键参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| **`cwd`** | `resolvedWorkspace` | Agent工作目录 - 用户文件存放处 |
| **`agentDir`** | `agentDir` | Agent配置目录 - 存放认证、模型配置 |

### Pi SDK的使用

OpenClaw使用`@mariozechner/pi-coding-agent`作为Pi的SDK：

```typescript
import {
  createAgentSession,
  SessionManager,
} from "@mariozechner/pi-coding-agent";
```

## OpenClaw的Session持久化

OpenClaw的Session数据持久化存储在特定目录，确保对话历史的连续性。

### 存储目录结构

```
~/.openclaw/
└── agents/
    └── main/                    ← Agent ID
        └── sessions/             ← Session存储目录
            ├── sessions.json     ← Session元数据
            │
            ├── abc123.jsonl      ← Session对话记录
            │
            └── ...
```

### 目录层级说明

```
~/.openclaw/
│
├── 📁 workspace/          ← cwd (工作目录)
│   ├── AGENTS.md          ← Agent定义
│   ├── SOUL.md            ← AI人格
│   ├── USER.md            ← 用户信息
│   └── ...
│
└── 📁 agents/
    └── 📁 main/
        └── 📁 agent/      ← agentDir
            ├── models.json  ← API Key等认证配置
            └── ...
        └── 📁 sessions/    ← Session持久化目录
            ├── sessions.json
            └── <sessionId>.jsonl
```

### Session存储内容

| 文件 | 内容 |
|------|------|
| `sessions.json` | Session元数据（ID、创建时间、最后活跃时间） |
| `<sessionId>.jsonl` | 对话历史（用户消息+AI回复+Tool调用结果） |

### Pi配置方式

Pi Agent的配置通过以下方式指定：

1. **代码默认值**：
```typescript
// src/agents/defaults.ts
export const DEFAULT_PROVIDER = "anthropic";
export const DEFAULT_MODEL = "claude-opus-4-6";
```

2. **配置文件（config.yaml）**：
```yaml
agents:
  defaults:
    provider: anthropic
    model: claude-sonnet-4-6
```

支持的Provider包括：`anthropic`、`openai`、`google`、`minimax`、`ollama`、`claude-cli`、`codex-cli`等。

## 结语

通过本文的分析，我们可以看到OpenClaw在集成Pi Agent方面采用了清晰的分层架构：

1. **统一入口**：`auto-reply`模块为所有消息通道提供一致的回复处理逻辑
2. **灵活执行**：根据Provider类型选择CLI模式或嵌入式模式
3. **清晰目录**：workspace与agents目录分离，Session数据独立存储

这种设计使得OpenClaw能够：
- 轻松支持多个消息平台
- 灵活切换不同的AI Provider
- 保持对话历史的持久化
- 支持多Agent并行运行

对于希望深入了解AI网关架构或扩展OpenClaw功能的开发者来说，理解这一集成方式将非常有帮助。

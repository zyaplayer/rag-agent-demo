# RAG Agent 启动与关闭指南（Windows + PowerShell）

本文档适用于当前项目目录：`E:\rag-agent-demo`。
# RAG Agent System

## 项目简介
基于 Milvus + LangChain + FastAPI 构建的智能问答系统

## 技术栈
- FastAPI
- Milvus
- Sentence-Transformers
- LangChain

## 1. 前置条件

- 已安装 `Anaconda/Miniconda`
- 已有环境：`rag-agent`
- 已安装 `Docker Desktop`
- 在项目根目录存在 `docker-compose.yml`

---

## 2. 手动启动（推荐先理解）

### 2.1 启动 Milvus

```powershell
cd E:\rag-agent-demo
conda activate rag-agent
docker compose up -d
Test-NetConnection localhost -Port 19530
```

看到 `TcpTestSucceeded : True` 代表 Milvus 可用。

### 2.2 初始化向量库（首次或重建时）

```powershell
cd E:\rag-agent-demo
conda activate rag-agent
python -m backend.rag.init_db
```

### 2.3 启动后端 FastAPI

```powershell
cd E:\rag-agent-demo
conda activate rag-agent
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

- 健康检查：`http://127.0.0.1:8000/api/`
- 文档地址：`http://127.0.0.1:8000/docs`

### 2.4 启动前端 Streamlit

```powershell
cd E:\rag-agent-demo
conda activate rag-agent
streamlit run frontend/app.py
```

- 前端地址通常为：`http://localhost:8501`

---

## 3. 手动关闭

1) 在前端终端按 `Ctrl + C`  
2) 在后端终端按 `Ctrl + C`  
3) 关闭 Milvus：

```powershell
cd E:\rag-agent-demo
docker compose down
```

如需清空 Milvus 数据（谨慎）：

```powershell
docker compose down -v
```

---

## 4. 一键脚本方案

项目根目录已提供：

- `start.ps1`：启动 Milvus + 可选初始化 + 启动后端/前端
- `stop.ps1`：停止后端/前端并关闭 Milvus（可选清库）

### 4.1 首次运行脚本（如果遇到执行策略限制）

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 4.2 使用方式

```powershell
# 启动（默认会初始化数据库）
.\start.ps1

# 启动但跳过 init_db
.\start.ps1 -SkipInitDB

# 关闭后端/前端 + 关闭 Milvus
.\stop.ps1

# 关闭并清空 Milvus 卷数据
.\stop.ps1 -RemoveVolumes
```

---

## 5. 常见问题

- `POST /chat/query 404`：前端用了旧接口，改为 `POST /api/chat`
- `MilvusException localhost:19530`：Milvus 未启动或端口不通
- `collection 不存在`：先执行 `python -m backend.rag.init_db`
- PowerShell `curl` 参数报错：优先使用 `Invoke-RestMethod`


import json
import requests
import streamlit as st

st.set_page_config(page_title="RAG Agent 知识库助理", layout="centered")

st.title("📚 企业知识库 RAG Agent 助理")
st.markdown("---")

# 与 backend/api/router.py 对齐
FASTAPI_URL = "http://127.0.0.1:8000/api/chat"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("references"):
            with st.expander("相关知识来源"):
                st.json(message["references"])

if prompt := st.chat_input("向知识库提问..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Agent 思考中..."):
        try:
            payload = {
                "query": prompt,
                "top_k": 3,
            }
            resp = requests.post(FASTAPI_URL, json=payload, timeout=60)
            resp.raise_for_status()

            data = resp.json()
            agent_response = data.get("answer", "未能获取 Agent 响应。")
            references = data.get("references", [])

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": agent_response,
                    "references": references,
                }
            )

            with st.chat_message("assistant"):
                st.markdown(agent_response)
                if references:
                    with st.expander("相关知识来源"):
                        st.json(references)

        except requests.exceptions.ConnectionError:
            st.error(f"无法连接后端，请确认服务已启动：{FASTAPI_URL}")
        except requests.exceptions.Timeout:
            st.error("请求超时，请稍后重试（可增大 timeout 或降低 top_k）。")
        except requests.exceptions.RequestException as e:
            try:
                err_json = e.response.json() if e.response is not None else {}
            except Exception:
                err_json = {}
            detail = err_json.get("detail") or str(e)
            st.error(f"请求后端 API 失败：{detail}")
        except json.JSONDecodeError:
            st.error("后端返回了无效 JSON。")
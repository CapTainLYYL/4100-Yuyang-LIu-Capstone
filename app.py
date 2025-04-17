import streamlit as st
from openai import OpenAI
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 设置页面标题
st.set_page_config(page_title="WRTT Assistant")
st.title("💬 WRTT Assistant")

# 初始化系统提示词模板
DEFAULT_SYSTEM_PROMPT = """Backgounds:1.You are an AI documentation assistant for WRTT company. 
2.Regardless of the user's language, you must answer in English.
3.Your thinking process must always be in English."""

# 在侧边栏添加配置选项
with st.sidebar:
    # API密钥输入
    openai_api_key = st.text_input("API Key", key="chatbot_api_key", type="password")
    "[Get API key](https://platform.siliconflow.cn/api_keys)"
    
    # 用户设置区域
    with st.expander("Settings", expanded=False):
        # 提示词模板配置
        st.subheader("Prompt Template")
        system_prompt = st.text_area(
            "System Prompt",
            value=DEFAULT_SYSTEM_PROMPT,
            key="system_prompt"
        )
    
    # 对话管理
    if st.button("New Chat"):
        # 保存当前对话到历史记录
        if "messages" in st.session_state:
            if "history_conversations" not in st.session_state:
                st.session_state.history_conversations = []
            
            # 如果当前对话已存在则更新，否则添加新对话
            if st.session_state.current_chat_index < len(st.session_state.history_conversations):
                st.session_state.history_conversations[st.session_state.current_chat_index] = st.session_state.messages.copy()
            else:
                st.session_state.history_conversations.append(st.session_state.messages.copy())
        
        # 创建新对话并添加到历史记录
        new_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": "How can I assist you with WRTT documentation today?"}
        ]
        st.session_state.history_conversations.append(new_messages.copy())
        st.session_state.current_chat_index = len(st.session_state.history_conversations) - 1
        st.session_state.messages = new_messages.copy()

    # 历史对话列表
    st.subheader("Chat History")
    if "history_conversations" in st.session_state:
        for idx in range(len(st.session_state.history_conversations)):
            if st.button(f"Chat {idx + 1}", key=f"load_conv_{idx}"):
                st.session_state.current_chat_index = idx
                st.session_state.messages = st.session_state.history_conversations[idx].copy()

# 初始化API客户端
if not openai_api_key:
    try:
        openai_api_key = st.secrets.api_key
    except AttributeError:
        st.info("Please enter your API key")
        st.stop()

client = OpenAI(
    api_key=openai_api_key,
    base_url="https://api.siliconflow.cn/v1/"
)
knowledge_base = [
    {"question": "What services does WRTT provide in wealth management?", "answer": "WRTT provides personalized investment strategies, portfolio management, and financial planning services."},
    {"question": "How do I open a retirement account?", "answer": "You can open a retirement account by contacting a WRTT advisor or visiting the Wells Fargo website and selecting the retirement account that fits your needs."},
    {"question": "What is a trust and how does it work?", "answer": "A trust is a legal arrangement where a trustee manages assets on behalf of beneficiaries. It helps in estate planning and asset protection."},
    {"question": "How can I check my 401k balance?", "answer": "Log into your Wells Fargo online account and navigate to the Retirement section to view your 401(k) balance."},
    {"question": "What are the fees associated with wealth management services?", "answer": "Fees vary based on services used, typically including management fees, fund expenses, and sometimes transaction fees."},
    {"question": "Can I rollover my old 401k into an IRA?", "answer": "Yes, you can rollover a 401(k) into a Wells Fargo IRA. This typically involves contacting a retirement advisor and filling out a rollover form."},
    {"question": "What is the 401k contribution limit for 2024?", "answer": "For 2024, the IRS contribution limit is $22,500, or $30,000 if you're 50 or older (including catch-up contributions)."},
    {"question": "How secure are my investments with WRTT?", "answer": "WRTT employs industry-standard security protocols and is FDIC-insured where applicable, ensuring your investments are safe."},
    {"question": "What is the difference between a Roth and Traditional IRA?", "answer": "Traditional IRAs offer tax-deferred growth while Roth IRAs provide tax-free withdrawals in retirement."},
    {"question": "How do I contact a trust advisor?", "answer": "You can reach a trust advisor by calling the WRTT customer service or visiting a local branch for a scheduled appointment."}
]

knowledge_context = "Knowledge Base:\n"
for qa in knowledge_base:
    knowledge_context += f"Q: {qa['question']}\nA: {qa['answer']}\n\n"
# 初始化对话历史
if "history_conversations" not in st.session_state:
    initial_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": "How can I assist you with WRTT documentation today?"}
    ]
    st.session_state.history_conversations = [initial_messages.copy()]
    st.session_state.current_chat_index = 0

if "messages" not in st.session_state:
    st.session_state.messages = st.session_state.history_conversations[st.session_state.current_chat_index].copy()

# 显示对话历史（过滤系统消息）
for msg in st.session_state.messages:
    if msg["role"] != "system":
        st.chat_message(msg["role"]).write(msg["content"])


# 处理用户输入
if prompt := st.chat_input():
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 加载知识库
    knowledge_base = []
    try:
        with open("knowledge_base.json", "r", encoding="utf-8") as f:
            knowledge_base = json.load(f)
    except Exception as e:
        st.error(f"Failed to load knowledge base: {str(e)}")

    # RAG处理流程
    context = ""
    if knowledge_base:
        # 获取Embeddings
        def get_embedding(text):
            response = client.embeddings.create(
                model="BAAI/bge-large-en-v1.5",
                input=text
            )
            return np.array(response.data[0].embedding)

        try:
            # 计算用户输入向量
            user_embedding = get_embedding(prompt)
            
            # 计算知识库文档向量
            doc_embeddings = []
            for doc in knowledge_base:
                content = f"Q: {doc.get('question','')}\nA: {doc.get('answer','')}"
                doc_embeddings.append(get_embedding(content))
            
            # 计算相似度
            similarities = cosine_similarity([user_embedding], doc_embeddings)[0]
            top_indices = np.argsort(similarities)[-3:][::-1]

            # 构建上下文
            context = "Your answer must strictly follow the given knowledge base answers, first give the content in the knowledge base, then start a new paragraph to supplement: \n"
            context = "Relevant knowledge base content:\n"
            for idx in top_indices:
                context += f"Q: {knowledge_base[idx].get('question','')}\nA: {knowledge_base[idx].get('answer','')}\n"

        except Exception as e:
            st.error(f"Knowledge retrieval failed: {str(e)}")

    # 构建API请求消息
    messages_for_api = [
        {"role": "system", "content": f"{system_prompt}\n{knowledge_context}\n{context}"},
        *[msg for msg in st.session_state.messages if msg["role"] != "system"]
    ]
    print(messages_for_api)
    # 调用API
    try:
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=messages_for_api,
            stream=True,
            extra_body={
                "reasoning_config": {
                    "enabled": True,
                    "type": "compact"
                }
            }
        )
    except Exception as e:
        st.error(f"API call failed: {str(e)}")
        st.stop()

    # 流式处理响应
    assistant_reply = ""
    reasoning_process = ""
    with st.chat_message("assistant"):
        reply_placeholder = st.empty()
        reasoning_expander = st.expander("Thinking Process", expanded=False)
        reasoning_placeholder = reasoning_expander.empty()

        # 初始化临时显示变量
        current_display = ""
        current_reasoning = ""

        for chunk in response:
            # 安全校验层1：确保有有效choices
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            
            # 安全校验层2：获取delta对象
            delta = getattr(choice, 'delta', None)
            if not delta:
                continue

            # 处理主回复内容
            content = getattr(delta, 'content', None) or ""
            if content:
                assistant_reply += content
                current_display = assistant_reply + "▌"  # 带光标
                reply_placeholder.markdown(current_display)

            # 处理思考过程
            reasoning_content = getattr(delta, 'reasoning_content', None) or ""
            if reasoning_content:
                reasoning_process += reasoning_content
                current_reasoning = reasoning_process + "▌"
                with reasoning_expander:
                    reasoning_placeholder.markdown(current_reasoning)

        # 最终清理光标符号
        final_display = assistant_reply.replace("▌", "")
        final_reasoning = reasoning_process.replace("▌", "")
        
        reply_placeholder.markdown(final_display)
        with reasoning_expander:
            reasoning_placeholder.markdown(final_reasoning)

    # 更新对话历史
    st.session_state.messages.append({
        "role": "assistant",
        "content": assistant_reply,
        "reasoning": reasoning_process  # 可选存储思考过程
    })
    st.session_state.history_conversations[st.session_state.current_chat_index] = st.session_state.messages.copy()
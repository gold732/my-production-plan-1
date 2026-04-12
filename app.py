import streamlit as st
import pandas as pd
from pyomo.environ import *
import plotly.graph_objects as go
import plotly.express as px
import google.generativeai as genai
import random

# 페이지 설정
st.set_page_config(page_title="Horticultural S&OP Control Tower", layout="wide")
st.title("🚀 AI 기반 생산계획(APP) 최적화 관제 시스템")

# --- AI 설정 및 API 로테이션 ---
def get_ai_consultant(prompt, context):
    try:
        keys = st.secrets.get("GEMINI_KEYS", [])
        if not keys: return "⚠️ Secrets에 API 키를 설정해주세요."
        
        genai.configure(api_key=random.choice(keys))
        # 모델명을 가장 안정적인 'gemini-1.5-flash'로 고정합니다.
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        system_msg = f"""당신은 스마트제조 생산관리 전문가입니다. 
        사용자가 수립한 총괄생산계획(APP) 데이터를 바탕으로 분석을 제공하세요.
        [데이터]: {context}"""
        
        response = model.generate_content(system_msg + "\n\n질문: " + prompt)
        return response.text
    except Exception as e:
        return f"❌ AI 오류: {str(e)}"

# --- 사이드바: 모든 운영 변수 풀세트 [cite: 123-128] ---
with st.sidebar:
    st.header("🎮 시스템 제어")
    opt_mode = st.radio("알고리즘 선택", ["정수계획법(IP)", "선형계획법(LP)"])
    domain_type = NonNegativeIntegers if "IP" in opt_mode else NonNegativeReals

    st.markdown("---")
    st.header("💰 단위 비용 (천원)")
    c_reg = st.number_input("정규 임금 (월)", value=640) [cite: 126]
    c_ot  = st.number_input("초과 근무 (시간)", value=6) [cite: 126]
    c_h   = st.number_input("고용 비용", value=300) [cite: 127]
    c_l   = st.number_input("해고 비용", value=500) [cite: 127]
    c_inv = st.number_input("재고 유지비", value=2) [cite: 124]
    c_back= st.number_input("부재고 비용", value=5) [cite: 124]
    c_mat = st.number_input("원자재비", value=10) [cite: 123]
    c_sub = st.number_input("외주 하청비", value=30) [cite: 139]

    st.markdown("---")
    st.header("📈 수요 및 초기값")
    demand_raw = st.text_input("6개월 수요", "1600, 3000, 3200, 3800, 2200, 2200") [cite: 122]
    demand = [float(d.strip()) for d in demand_raw.split(",")]
    w_init = st.number_input("초기 인원", value=80) [cite: 126]
    i_init = st.number_input("초기 재고", value=1000) [cite: 125]
    i_final = st.number_input("목표 기말재고", value=500) [cite: 125]

# --- 최적화 로직 [cite: 141-160] ---
def solve_app(D, domain):
    m = ConcreteModel()
    T = range(1, len(D) + 1)
    TIME = range(0, len(D) + 1)
    
    m.W = Var(TIME, domain=domain); m.H = Var(TIME, domain=domain); m.L = Var(TIME, domain=domain)
    m.P = Var(TIME, domain=domain); m.I = Var(TIME, domain=domain); m.S = Var(TIME, domain=domain)
    m.C = Var(TIME, domain=domain); m.O = Var(TIME, domain=domain)

    # 목적함수 수식 [cite: 144]
    m.cost = Objective(expr=sum(
        c_reg*m.W[t] + c_ot*m.O[t] + c_h*m.H[t] + c_l*m.L[t] + 
        c_inv*m.I[t] + c_back*m.S[t] + c_mat*m.P[t] + c_sub*m.C[t] 
        for t in T), sense=minimize)
    
    m.c = ConstraintList()
    m.c.add(m.W[0] == w_init); m.c.add(m.I[0] == i_init); m.c.add(m.S[0] == 0)
    for t in T:
        m.c.add(m.W[t] == m.W[t-1] + m.H[t] - m.L[t]) # [cite: 149]
        m.c.add(m.P[t] <= 40*m.W[t] + 0.25*m.O[t])     # [cite: 154]
        m.c.add(m.I[t] == m.I[t-1] + m.P[t] + m.C[t] - D[t-1] - m.S[t-1] + m.S[t]) # [cite: 157]
        m.c.add(m.O[t] <= 10*m.W[t])                   # [cite: 160]
    m.c.add(m.I[len(D)] >= i_final); m.c.add(m.S[len(D)] == 0)
    
    SolverFactory('glpk').solve(m)
    return m

# --- 화면 탭 구성 ---
tab1, tab2 = st.tabs(["📊 통합 대시보드", "💬 AI 전문가 상담"])

with tab1:
    if st.button("🚀 최적화 실행"):
        model = solve_app(demand, domain_type)
        st.session_state['res'] = model
        
        # 1. KPI 지표
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("총 비용", f"{model.cost():,.0f}k")
        k2.metric("평균 재고", f"{sum(model.I[t]() for t in range(1,7))/6:,.0f}ea")
        k3.metric("인력 변동", f"{sum(model.H[t]() + model.L[t]() for t in range(1,7)):.0f}명")
        k4.metric("외주량", f"{sum(model.C[t]() for t in range(1,7)):,.0f}ea")

        # 2. 복합 차트
        st.subheader("📈 공급망 흐름 진단")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(range(1,7)), y=[model.P[t]() for t in range(1,7)], name="자체생산"))
        fig.add_trace(go.Scatter(x=list(range(1,7)), y=demand, name="수요", line=dict(color='red', dash='dash')))
        fig.add_trace(go.Scatter(x=list(range(1,7)), y=[model.I[t]() for t in range(1,7)], name="재고", yaxis="y2"))
        fig.update_layout(yaxis2=dict(overlaying='y', side='right'), barmode='stack')
        st.plotly_chart(fig, use_container_width=True)

        # 3. 비용 분석 (문제의 괄호 부분 수정 완료)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("💰 비용 세부 구성")
            costs = {
                "노무비": sum(c_reg * model.W[t]() for t in range(1, 7)),
                "재고비": sum(c_inv * model.I[t]() for t in range(1, 7)),
                "재료비": sum(c_mat * model.P[t]() for t in range(1, 7)),
                "기타": model.cost() - sum((c_reg*model.W[t]() + c_inv*model.I[t]() + c_mat*model.P[t]()) for t in range(1,7))
            }
            st.plotly_chart(px.pie(names=list(costs.keys()), values=list(costs.values()), hole=0.4))
        with c2:
            st.subheader("👷 인력 운영 현황")
            st.line_chart(pd.DataFrame({"인원": [model.W[t]() for t in range(1, 7)]}))

with tab2:
    st.subheader("💬 AI S&OP 컨설턴트")
    if 'msgs' not in st.session_state: st.session_state.msgs = []
    for m in st.session_state.msgs:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if p := st.chat_input("계획에 대해 물어보세요"):
        st.session_state.msgs.append({"role": "user", "content": p})
        with st.chat_message("user"): st.markdown(p)
        
        ctx = "최적화 전"
        if 'res' in st.session_state:
            res = st.session_state['res']
            ctx = f"총비용: {res.cost():,.0f}, 기말재고: {res.I[6]()}, 외주: {sum(res.C[t]() for t in range(1,7))}"
            
        with st.chat_message("assistant"):
            ans = get_ai_consultant(p, ctx)
            st.markdown(ans)
            st.session_state.msgs.append({"role": "assistant", "content": ans})

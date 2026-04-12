import streamlit as st
import pandas as pd
from pyomo.environ import *
import plotly.graph_objects as go
import plotly.express as px
import google.generativeai as genai
import random

# 1. 페이지 설정 및 디자인
st.set_page_config(page_title="Horticultural S&OP Master", layout="wide")
st.title("🛡️ 스마트제조 AI 생산전략 관제탑 (S&OP Control Tower)")

# 2. AI 컨설턴트 로직 (Gemini 2.5-Flash-Lite + API 로테이션)
def get_ai_consultant(prompt, context_summary):
    try:
        keys = st.secrets.get("GEMINI_KEYS", [])
        if not keys:
            return "⚠️ Streamlit Secrets에 'GEMINI_KEYS'를 설정해주세요."
        
        # API 키 순환 선택
        selected_key = random.choice(keys)
        genai.configure(api_key=selected_key)
        
        # 요청하신 최신 모델 반영
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        system_instruction = f"""
        당신은 스마트제조 및 생산관리 전문 컨설턴트입니다. 
        사용자가 수립한 총괄생산계획(APP) 최적화 데이터를 바탕으로 경영적 통찰을 제공하세요.
        
        [현재 최적화 결과 데이터 요약]
        {context_summary}
        
        위 데이터를 근거로 사용자의 질문에 답변하고, 비용 절감이나 리스크 관리 방안을 제안하세요.
        """
        
        response = model.generate_content(system_instruction + "\n\n사용자 질문: " + prompt)
        return response.text
    except Exception as e:
        return f"❌ AI 서비스 오류: {str(e)}"

# 3. 사이드바: 모든 제어 파라미터 풀세트 [cite: 121-129, 171-176]
with st.sidebar:
    st.header("🎮 시스템 제어판")
    opt_mode = st.radio("알고리즘 선택", ["정수계획법(IP)", "선형계획법(LP)"])
    domain_type = NonNegativeIntegers if "IP" in opt_mode else NonNegativeReals

    st.markdown("---")
    st.subheader("⏱️ 공정 효율 및 제약")
    std_time = st.slider("제품당 표준 작업 시간 (Hr)", 1.0, 10.0, 4.0) # [cite: 129]
    working_days = st.slider("월간 가동 일수", 0, 30, 20) # [cite: 128]
    ot_limit = st.slider("인당 월간 초과근무 제한 (Hr)", 0, 30, 10) # [cite: 128]

    st.markdown("---")
    st.subheader("💰 운영 비용 설정 (천원)")
    c_reg = st.number_input("정규 임금 (인/월)", value=640) # [cite: 144]
    c_ot  = st.number_input("초과 근무 수당 (Hr)", value=6) # [cite: 126]
    c_h   = st.number_input("신규 고용 비용 (인)", value=300) # [cite: 127]
    c_l   = st.number_input("해고 비용 (인)", value=500) # [cite: 127]
    c_inv = st.number_input("재고 유지비 (개/월)", value=2) # [cite: 124]
    c_back= st.number_input("부재고 비용 (개/월)", value=5) # [cite: 124]
    c_mat = st.number_input("재료비 (개당)", value=10) # [cite: 123]
    c_sub = st.number_input("외주 하청 비용 (개당)", value=30) # [cite: 144]

    st.markdown("---")
    st.subheader("📈 초기값 및 수요")
    demand_raw = st.text_input("6개월 수요 예측 (쉼표 구분)", "1600, 3000, 3200, 3800, 2200, 2200")
    demand = [float(d.strip()) for d in demand_raw.split(",")]
    w_init = st.number_input("현재 근로자 수", value=80) # [cite: 126]
    i_init = st.number_input("현재고 수준", value=1000) # [cite: 125]
    i_final = st.number_input("기말 목표 재고", value=500) # [cite: 125]

# 4. 최적화 엔진 (Pyomo)
def solve_production_plan(D, domain):
    m = ConcreteModel()
    T = range(1, len(D) + 1)
    TIME = range(0, len(D) + 1)

    # 결정변수 [cite: 133-140]
    m.W = Var(TIME, domain=domain); m.H = Var(TIME, domain=domain); m.L = Var(TIME, domain=domain)
    m.P = Var(TIME, domain=domain); m.I = Var(TIME, domain=domain); m.S = Var(TIME, domain=domain)
    m.C = Var(TIME, domain=domain); m.O = Var(TIME, domain=domain)

    # 목적함수 [cite: 141-144]
    m.cost = Objective(expr=sum(
        c_reg*m.W[t] + c_ot*m.O[t] + c_h*m.H[t] + c_l*m.L[t] + 
        c_inv*m.I[t] + c_back*m.S[t] + c_mat*m.P[t] + c_sub*m.C[t] 
        for t in T), sense=minimize)
    
    # 제약조건 [cite: 145-160]
    m.cons = ConstraintList()
    m.cons.add(m.W[0] == w_init)
    m.cons.add(m.I[0] == i_init)
    m.cons.add(m.S[0] == 0)
    
    for t in T:
        m.cons.add(m.W[t] == m.W[t-1] + m.H[t] - m.L[t]) # [cite: 149]
        # 공정 효율 변수 적용된 생산능력 제약 [cite: 154]
        cap_reg = (1/std_time) * 8 * working_days * m.W[t]
        m.cons.add(m.P[t] <= cap_reg + (1/std_time)*m.O[t]) 
        m.cons.add(m.I[t] == m.I[t-1] + m.P[t] + m.C[t] - D[t-1] - m.S[t-1] + m.S[t]) # [cite: 157]
        m.cons.add(m.O[t] <= ot_limit * m.W[t]) # [cite: 160]
        
    m.cons.add(m.I[len(D)] >= i_final) # [cite: 173]
    m.cons.add(m.S[len(D)] == 0) # [cite: 176]

    result = SolverFactory('glpk').solve(m)
    return m, result

# 5. 세션 상태 초기화
if 'res' not in st.session_state: st.session_state['res'] = None
if 'success' not in st.session_state: st.session_state['success'] = False
if 'messages' not in st.session_state: st.session_state.messages = []

# 6. 메인 화면 탭 구성
tab1, tab2, tab3 = st.tabs(["📊 운영 대시보드", "📉 리스크/효율 분석", "💬 AI 전략 상담방"])

with tab1:
    if st.button("🚀 최적 생산계획 수립 실행"):
        model, sol = solve_production_plan(demand, domain_type)
        if sol.solver.termination_condition == TerminationCondition.optimal:
            st.session_state['res'] = model
            st.session_state['success'] = True
        else:
            st.session_state['success'] = False
            st.error("❌ 최적해를 찾을 수 없습니다. 제약 조건이나 수요를 조정해주세요.")

    if st.session_state['success']:
        m = st.session_state['res']
        
        # 안전한 가동률 계산 (ZeroDivision 방지)
        utils = []
        for t in range(1, 7):
            denom = 8 * working_days * m.W[t]()
            utils.append((m.P[t]() * std_time / denom * 100) if denom > 0 else 0)

        # KPI 메트릭
        kpis = st.columns(4)
        kpis[0].metric("총 운영 비용", f"{m.cost():,.0f}k")
        kpis[1].metric("평균 가동률", f"{sum(utils)/6:.1f}%")
        kpis[2].metric("총 인력 변동", f"{sum(m.H[t]() + m.L[t]() for t in range(1,7)):.0f}명")
        kpis[3].metric("외주 처리량", f"{sum(m.C[t]() for t in range(1,7)):,.0f}ea")

        # 복합 흐름 차트
        st.subheader("📈 월별 공급망 흐름 (수요/생산/재고)")
        fig_main = go.Figure()
        fig_main.add_trace(go.Bar(x=list(range(1,7)), y=[m.P[t]() for t in range(1,7)], name="자체 생산", marker_color='royalblue'))
        fig_main.add_trace(go.Bar(x=list(range(1,7)), y=[m.C[t]() for t in range(1,7)], name="외주 하청", marker_color='lightslategray'))
        fig_main.add_trace(go.Scatter(x=list(range(1,7)), y=demand, name="예상 수요", line=dict(color='crimson', width=3, dash='dash')))
        fig_main.add_trace(go.Scatter(x=list(range(1,7)), y=[m.I[t]() for t in range(1,7)], name="재고 수준", yaxis="y2", line=dict(color='orange', width=2)))
        fig_main.update_layout(yaxis2=dict(overlaying='y', side='right'), barmode='stack', hovermode="x unified")
        st.plotly_chart(fig_main, use_container_width=True)

        # 하단 분석 차트
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("💰 비용 세부 구성")
            costs_breakdown = {
                "노무비": sum(c_reg*m.W[t]() for t in range(1,7)),
                "재고비": sum(c_inv*m.I[t]() for t in range(1,7)),
                "재료비": sum(c_mat*m.P[t]() for t in range(1,7)),
                "기타(외주/채용 등)": m.cost() - sum((c_reg*m.W[t]() + c_inv*m.I[t]() + c_mat*m.P[t]()) for t in range(1,7))
            }
            st.plotly_chart(px.pie(names=list(costs_breakdown.keys()), values=list(costs_breakdown.values()), hole=0.4), use_container_width=True)
        with c2:
            st.subheader("👷 인력 운영 안정성")
            st.line_chart(pd.DataFrame({"작업자 수": [m.W[t]() for t in range(1,7)]}))

with tab2:
    st.subheader("⚠️ 운영 리스크 및 가동률 진단")
    if st.session_state['success']:
        m = st.session_state['res']
        # 가동률 리포트
        fig_risk = px.area(x=list(range(1,7)), y=utils, title="생산 설비 가동률 추이 (%)", markers=True)
        fig_risk.add_hline(y=100, line_dash="dot", line_color="red", annotation_text="풀 가동 한계")
        st.plotly_chart(fig_risk, use_container_width=True)
        
        if max(utils) > 100:
            st.error(f"🚨 경고: 특정 달에 가동률이 {max(utils):.1f}%까지 치솟아 과부하가 예상됩니다.")
        else:
            st.success("✅ 현재 계획은 모든 공정 능력 범위 내에 안정적으로 분포되어 있습니다.")

with tab3:
    st.subheader("💬 AI 전략 컨설턴트 (Gemini 2.5-Flash-Lite)")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("계획에 대해 질문하세요."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        ctx = "최적화 데이터 없음"
        if st.session_state['success']:
            m = st.session_state['res']
            ctx = f"총비용:{m.cost():,.0f}, 알고리즘:{opt_mode}, 평균가동률:{sum(utils)/6:.1f}%, 기말재고:{m.I[6]()}"
            
        with st.chat_message("assistant"):
            ai_res = get_ai_consultant(prompt, ctx)
            st.markdown(ai_res)
            st.session_state.messages.append({"role": "assistant", "content": ai_res})

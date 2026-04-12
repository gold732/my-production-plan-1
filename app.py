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

# 2. AI 컨설턴트 로직 (API 키 자동 재시도 및 로테이션 강화)
def get_ai_consultant(prompt, context_summary):
    keys = st.secrets.get("GEMINI_KEYS", [])
    if not keys:
        return "⚠️ Streamlit Secrets에 'GEMINI_KEYS'를 설정해주세요."
    
    # 가용 키 목록을 랜덤하게 섞어서 순차적으로 시도
    available_keys = list(keys)
    random.shuffle(available_keys)
    
    last_error = ""
    for key in available_keys:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-2.5-flash-lite')
            
            system_instruction = f"""
            당신은 스마트제조 및 생산관리 전문 컨설턴트입니다. 
            사용자가 수립한 총괄생산계획(APP) 최적화 데이터를 바탕으로 통찰력을 제공하세요.
            [현재 최적화 결과 데이터 요약]
            {context_summary}
            위 데이터를 근거로 사용자의 질문에 답변하고, 경영적 제언을 하세요.
            """
            
            response = model.generate_content(system_instruction + "\n\n사용자 질문: " + prompt)
            return response.text
        except Exception as e:
            last_error = str(e)
            continue  # 에러 발생 시 다음 키로 이동
            
    return f"❌ 모든 API 키가 유효하지 않거나 오류가 발생했습니다: {last_error}"

# 3. 사이드바: 모든 제어 파라미터 풀세트 [cite: 121-129, 171-176]
with st.sidebar:
    st.header("🎮 시스템 제어판")
    opt_mode = st.radio("알고리즘 선택", ["정수계획법(IP)", "선형계획법(LP)"])
    domain_type = NonNegativeIntegers if "IP" in opt_mode else NonNegativeReals

    st.markdown("---")
    st.subheader("⏱️ 공정 효율 및 제약")
    std_time = st.slider("제품당 표준 작업 시간 (Hr)", 1.0, 10.0, 4.0)
    working_days = st.slider("월간 가동 일수", 0, 30, 20)
    ot_limit = st.slider("인당 월간 초과근무 제한 (Hr)", 0, 30, 10)

    st.markdown("---")
    st.subheader("💰 운영 비용 설정 (천원)")
    c_reg = st.number_input("정규 임금 (인/월)", value=640)
    c_ot  = st.number_input("초과 근무 수당 (Hr)", value=6)
    c_h   = st.number_input("신규 고용 비용 (인)", value=300)
    c_l   = st.number_input("해고 비용 (인)", value=500)
    c_inv = st.number_input("재고 유지비 (개/월)", value=2)
    c_back= st.number_input("부재고 비용 (개/월)", value=5)
    c_mat = st.number_input("재료비 (개당)", value=10)
    c_sub = st.number_input("외주 하청 비용 (개당)", value=30)

    st.markdown("---")
    st.subheader("📈 초기값 및 수요")
    demand_raw = st.text_input("6개월 수요 예측 (쉼표 구분)", "1600, 3000, 3200, 3800, 2200, 2200")
    demand = [float(d.strip()) for d in demand_raw.split(",")]
    w_init = st.number_input("현재 근로자 수", value=80)
    i_init = st.number_input("현재고 수준", value=1000)
    i_final = st.number_input("기말 목표 재고", value=500)

# 4. 최적화 엔진 (Pyomo) [cite: 130-160]
def solve_production_plan(D, domain):
    m = ConcreteModel()
    T = range(1, len(D) + 1)
    TIME = range(0, len(D) + 1)

    m.W = Var(TIME, domain=domain); m.H = Var(TIME, domain=domain); m.L = Var(TIME, domain=domain)
    m.P = Var(TIME, domain=domain); m.I = Var(TIME, domain=domain); m.S = Var(TIME, domain=domain)
    m.C = Var(TIME, domain=domain); m.O = Var(TIME, domain=domain)

    m.cost = Objective(expr=sum(
        c_reg*m.W[t] + c_ot*m.O[t] + c_h*m.H[t] + c_l*m.L[t] + 
        c_inv*m.I[t] + c_back*m.S[t] + c_mat*m.P[t] + c_sub*m.C[t] 
        for t in T), sense=minimize)
    
    m.cons = ConstraintList()
    m.cons.add(m.W[0] == w_init)
    m.cons.add(m.I[0] == i_init)
    m.cons.add(m.S[0] == 0)
    
    for t in T:
        m.cons.add(m.W[t] == m.W[t-1] + m.H[t] - m.L[t])
        cap_reg = (1/std_time) * 8 * working_days * m.W[t]
        m.cons.add(m.P[t] <= cap_reg + (1/std_time)*m.O[t]) 
        m.cons.add(m.I[t] == m.I[t-1] + m.P[t] + m.C[t] - D[t-1] - m.S[t-1] + m.S[t]) 
        m.cons.add(m.O[t] <= ot_limit * m.W[t])
        
    m.cons.add(m.I[len(D)] >= i_final)
    m.cons.add(m.S[len(D)] == 0)

    result = SolverFactory('glpk').solve(m)
    return m, result

# 5. 세션 상태 초기화 및 데이터 갱신 보장
if 'res' not in st.session_state: st.session_state['res'] = None
if 'success' not in st.session_state: st.session_state['success'] = False
if 'messages' not in st.session_state: st.session_state.messages = []

# 6. 메인 화면 탭 구성
tab1, tab2, tab3 = st.tabs(["📊 운영 대시보드", "📉 리스크/효율 분석", "💬 AI 전략 상담방"])

with tab1:
    # 실행 버튼 클릭 시 이전 데이터를 명확히 밀어내고 새로 계산
    if st.button("🚀 최적 생산계획 수립 실행"):
        st.session_state['success'] = False  # 상태 초기화
        with st.spinner('최신 파라미터로 계산 중...'):
            model, sol = solve_production_plan(demand, domain_type)
            if sol.solver.termination_condition == TerminationCondition.optimal:
                st.session_state['res'] = model
                st.session_state['success'] = True
            else:
                st.error("❌ 최적해를 찾을 수 없습니다. 제약 조건을 조정해주세요.")

    if st.session_state['success']:
        m = st.session_state['res']
        
        # 안전한 가동률 계산 (ZeroDivision 방지)
        utils = []
        for t in range(1, len(demand) + 1):
            denom = 8 * working_days * m.W[t]()
            utils.append((m.P[t]() * std_time / denom * 100) if denom > 0 else 0)

        kpis = st.columns(4)
        kpis[0].metric("총 운영 비용", f"{m.cost():,.0f}k")
        kpis[1].metric("평균 가동률", f"{sum(utils)/len(utils):.1f}%")
        kpis[2].metric("총 인력 변동", f"{sum(m.H[t]() + m.L[t]() for t in range(1,len(demand)+1)):.0f}명")
        kpis[3].metric("기말 재고 현황", f"{m.I[len(demand)]():,.0f}ea")

        st.subheader("📈 월별 공급망 흐름 (수요/생산/재고)")
        fig_main = go.Figure()
        fig_main.add_trace(go.Bar(x=list(range(1,len(demand)+1)), y=[m.P[t]() for t in range(1,len(demand)+1)], name="자체 생산", marker_color='royalblue'))
        fig_main.add_trace(go.Bar(x=list(range(1,len(demand)+1)), y=[m.C[t]() for t in range(1,len(demand)+1)], name="외주 하청", marker_color='lightslategray'))
        fig_main.add_trace(go.Scatter(x=list(range(1,len(demand)+1)), y=demand, name="예상 수요", line=dict(color='crimson', width=3, dash='dash')))
        fig_main.add_trace(go.Scatter(x=list(range(1,len(demand)+1)), y=[m.I[t]() for t in range(1,len(demand)+1)], name="재고 수준", yaxis="y2", line=dict(color='orange', width=2)))
        fig_main.update_layout(yaxis2=dict(overlaying='y', side='right'), barmode='stack', hovermode="x unified")
        st.plotly_chart(fig_main, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("💰 비용 세부 구성")
            costs_breakdown = {
                "노무비": sum(c_reg*m.W[t]() for t in range(1,len(demand)+1)),
                "재고비": sum(c_inv*m.I[t]() for t in range(1,len(demand)+1)),
                "재료비": sum(c_mat*m.P[t]() for t in range(1,len(demand)+1)),
                "기타비용": m.cost() - sum((c_reg*m.W[t]() + c_inv*m.I[t]() + c_mat*m.P[t]()) for t in range(1,len(demand)+1))
            }
            st.plotly_chart(px.pie(names=list(costs_breakdown.keys()), values=list(costs_breakdown.values()), hole=0.4), use_container_width=True)
        with c2:
            st.subheader("👷 인력 운영 안정성")
            st.line_chart(pd.DataFrame({"작업자 수": [m.W[t]() for t in range(1,len(demand)+1)]}))

with tab2:
    st.subheader("⚠️ 운영 리스크 및 가동률 진단")
    if st.session_state['success']:
        m = st.session_state['res']
        fig_risk = px.area(x=list(range(1,len(demand)+1)), y=utils, title="생산 설비 가동률 추이 (%)", markers=True)
        fig_risk.add_hline(y=100, line_dash="dot", line_color="red", annotation_text="가동 한계선")
        st.plotly_chart(fig_risk, use_container_width=True)
        
        if max(utils) > 100:
            st.error(f"🚨 경고: 가동률이 {max(utils):.1f}%로 한계를 초과했습니다. 외주 혹은 추가 채용이 필요합니다.")
        else:
            st.success("✅ 생산 능력이 수요를 안정적으로 소화하고 있습니다.")

with tab3:
    st.subheader("💬 AI 전략 컨설턴트")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("계획의 타당성이나 개선 방안을 물어보세요."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        summary = "최적화 미수행"
        if st.session_state['success']:
            m = st.session_state['res']
            summary = f"총비용:{m.cost():,.0f}, 알고리즘:{opt_mode}, 평균가동률:{sum(utils)/len(utils):.1f}%"
            
        with st.chat_message("assistant"):
            ai_res = get_ai_consultant(prompt, summary)
            st.markdown(ai_res)
            st.session_state.messages.append({"role": "assistant", "content": ai_res})

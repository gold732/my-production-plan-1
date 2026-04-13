import streamlit as st
import pandas as pd
from pyomo.environ import *
import plotly.graph_objects as go
import plotly.express as px
import google.generativeai as genai
import random

# 1. 페이지 설정 및 디자인
st.set_page_config(page_title="AI S&OP Control Tower", layout="wide")
st.title("🛡️ 스마트제조 AI 생산전략 관제탑 (S&OP Master)")

# 2. AI 컨설턴트 로직
def get_ai_consultant(prompt, context_summary):
    keys = st.secrets.get("GEMINI_KEYS", [])
    if not keys: return "⚠️ Secrets에 'GEMINI_KEYS'를 설정해주세요."
    
    available_keys = list(keys)
    random.shuffle(available_keys)
    
    for key in available_keys:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-2.5-flash-lite')
            
            system_instruction = f"""
            당신은 스마트제조 및 생산관리 전문 컨설턴트입니다. 
            아래 최적화 결과(자체 생산비, 외주비 포함)를 분석하여 경영적 통찰을 제공하세요.
            특히 가동률이 100%를 넘는 달의 리스크와 외주 비중이 높을 경우의 수익성 저하를 경고하세요.
            
            [현재 최적화 결과 데이터]
            {context_summary}
            """
            
            response = model.generate_content(system_instruction + "\n\n사용자 질문: " + prompt)
            return response.text
        except Exception:
            continue 
    return "❌ AI 연결 오류 (모든 키 시도 실패)"

# 3. 사이드바 제어판 (기존 유지)
with st.sidebar:
    st.header("🎮 시스템 제어판")
    opt_mode = st.radio("알고리즘 선택", ["정수계획법(IP)", "선형계획법(LP)"])
    domain_type = NonNegativeIntegers if "IP" in opt_mode else NonNegativeReals

    st.markdown("---")
    st.subheader("⏱️ 공정 효율 및 제약")
    std_time = st.slider("제품당 표준 작업 시간 (Hr)", 1.0, 10.0, 4.0)
    working_days = st.slider("월간 가동 일수", 1, 30, 20)
    ot_limit = st.slider("인당 월간 초과근무 제한 (Hr)", 0, 30, 10)

    st.markdown("---")
    st.subheader("💰 운영 비용 설정 (천원)")
    v_c_reg = st.number_input("정규 임금 (인/월)", value=640)
    v_c_ot  = st.number_input("초과 근무 수당 (Hr)", value=6)
    v_c_h   = st.number_input("신규 고용 비용 (인)", value=300)
    v_c_l   = st.number_input("해고 비용 (인)", value=500)
    v_c_inv = st.number_input("재고 유지비 (개/월)", value=2)
    v_c_back= st.number_input("부재고 비용 (개/월)", value=5)
    v_c_mat = st.number_input("재료비 (개당)", value=10)
    v_c_sub = st.number_input("외주 하청 비용 (개당)", value=30)

    st.markdown("---")
    st.subheader("📈 초기값 및 수요")
    demand_raw = st.text_input("6개월 수요 예측 (쉼표 구분)", "1600, 3000, 3200, 3800, 2200, 2200")
    demand = [float(d.strip()) for d in demand_raw.split(",")]
    v_w_init = st.number_input("현재 근로자 수", value=80)
    v_i_init = st.number_input("현재고 수준", value=1000)
    v_i_final = st.number_input("기말 목표 재고", value=500)

# 4. [span_2](start_span)[span_3](start_span)최적화 엔진 (강의록 수식 준수[span_2](end_span)[span_3](end_span))
def solve_production_plan(D, domain, reg, ot, h, l, inv, back, mat, sub, stime, wdays, ot_lim, w0, i0, ifinal):
    m = ConcreteModel()
    T = range(1, len(D) + 1); TIME = range(0, len(D) + 1)
    m.W = Var(TIME, domain=domain); m.H = Var(TIME, domain=domain); m.L = Var(TIME, domain=domain)
    m.P = Var(TIME, domain=domain); m.I = Var(TIME, domain=domain); m.S = Var(TIME, domain=domain)
    m.C = Var(TIME, domain=domain); m.O = Var(TIME, domain=domain)

    # 목적함수: 모든 비용 항목 포함
    m.cost = Objective(expr=sum(reg*m.W[t] + ot*m.O[t] + h*m.H[t] + l*m.L[t] + 
                                inv*m.I[t] + back*m.S[t] + mat*m.P[t] + sub*m.C[t] for t in T), sense=minimize)
    
    m.c = ConstraintList()
    m.c.add(m.W[0] == w0); m.c.add(m.I[0] == i0); m.c.add(m.S[0] == 0)
    for t in T:
        m.c.add(m.W[t] == m.W[t-1] + m.H[t] - m.L[t])
        cap_reg = (1/stime) * 8 * wdays * m.W[t]
        m.c.add(m.P[t] <= cap_reg + (1/stime)*m.O[t]) 
        m.c.add(m.I[t] == m.I[t-1] + m.P[t] + m.C[t] - D[t-1] - m.S[t-1] + m.S[t]) 
        m.c.add(m.O[t] <= ot_lim * m.W[t])
    m.c.add(m.I[len(D)] >= ifinal); m.c.add(m.S[len(D)] == 0)

    SolverFactory('glpk').solve(m)
    return m

# 5. 세션 상태 및 탭 구성
if 'messages' not in st.session_state: st.session_state.messages = []
if 'success' not in st.session_state: st.session_state['success'] = False

tab1, tab2, tab3 = st.tabs(["📊 운영 대시보드", "📉 리스크/효율 분석", "💬 AI 전략 상담방"])

with tab1:
    if st.button("🚀 최적 생산계획 수립 실행"):
        try:
            m = solve_production_plan(demand, domain_type, v_c_reg, v_c_ot, v_c_h, v_c_l, v_c_inv, v_c_back, v_c_mat, v_c_sub, std_time, working_days, ot_limit, v_w_init, v_i_init, v_i_final)
            st.session_state['res'] = m
            st.session_state['success'] = True
            
            # 가동률 계산
            temp_utils = []
            for t in range(1, len(demand) + 1):
                denom = 8 * working_days * m.W[t]()
                temp_utils.append((m.P[t]() * std_time / denom * 100) if denom > 0 else 0)
            st.session_state['utils'] = temp_utils
            st.toast("✅ 최적화 성공!")
        except Exception as e:
            st.error(f"⚠️ 연산 오류: {str(e)}")

    if st.session_state.get('success'):
        m = st.session_state['res']
        utils = st.session_state['utils']
        T_range = range(1, len(demand) + 1)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("총 운영 비용", f"{m.cost():,.0f}k")
        k2.metric("평균 가동률", f"{sum(utils)/len(utils):.1f}%")
        k3.metric("총 외주 수량", f"{sum(m.C[t]() for t in T_range):,.0f}ea")
        k4.metric("기말 재고량", f"{m.I[len(demand)]():,.0f}ea")

        # 메인 차트 (사용자 요청: 외주 그래프 제외)
        st.subheader("📈 월별 생산/수요/재고 흐름")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(T_range), y=[m.P[t]() for t in T_range], name="자체 생산량", marker_color='royalblue'))
        fig.add_trace(go.Scatter(x=list(T_range), y=demand, name="예상 수요", line=dict(color='crimson', dash='dash')))
        fig.add_trace(go.Scatter(x=list(T_range), y=[m.I[t]() for t in T_range], name="재고 수준", yaxis="y2", line=dict(color='orange')))
        fig.update_layout(yaxis2=dict(overlaying='y', side='right'), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("💰 비용 세부 구성 (외주비 포함)")
            # 비용 상세 계산
            c_labor = sum(v_c_reg*m.W[t]() + v_c_ot*m.O[t]() for t in T_range)
            c_inventory = sum(v_c_inv*m.I[t]() + v_c_back*m.S[t]() for t in T_range)
            c_material = sum(v_c_mat*m.P[t]() for t in T_range)
            c_sub = sum(v_c_sub*m.C[t]() for t in T_range) # 외주 비용 명시
            c_hr = sum(v_c_h*m.H[t]() + v_c_l*m.L[t]() for t in T_range)
            
            cost_map = {"노무비": c_labor, "재고/부재고비": c_inventory, "재료비": c_material, "외주비": c_sub, "채용/해고비": c_hr}
            st.plotly_chart(px.pie(names=list(cost_map.keys()), values=list(cost_map.values()), hole=0.4), use_container_width=True)
        with col_r:
            st.subheader("👷 월별 인력 운영 현황")
            st.line_chart(pd.DataFrame({"인원": [m.W[t]() for t in T_range]}))

# 탭 2, 탭 3 로직은 기존 수정본 유지 (생략 가능하나 문맥을 위해 ai_res 부분만 업데이트)
with tab3:
    st.subheader("💬 AI 전략 상담방")
    if st.button("🧹 대화 내용 초기화"):
        st.session_state.messages = []; st.rerun()
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("계획에 대해 질문하세요."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        if st.session_state['success']:
            m = st.session_state['res']
            sub_total = sum(m.C[t]() for t in range(1, len(demand)+1))
            sub_cost = sum(v_c_sub*m.C[t]() for t in range(1, len(demand)+1))
            ctx = f"총비용:{m.cost():,.0f}, 외주비용:{sub_cost:,.0f}, 외주수량:{sub_total}, 가동률:{st.session_state['utils']}"
        else: ctx = "데이터 없음"

        with st.chat_message("assistant"):
            ai_res = get_ai_consultant(prompt, ctx)
            st.markdown(ai_res)
            st.session_state.messages.append({"role": "assistant", "content": ai_res})

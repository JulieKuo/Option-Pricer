#!/usr/bin/env python
# coding: utf-8


import numpy as np
import QuantLib as ql
import random
from tqdm import tqdm
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import create_engine


class basic_option():
    '''parameter'''
    ''' startdate: 定價日
        maturity: 到期日
        S: 即期價格
        K: 履約價
        p: 本金

        direction: 1.Buy  2.Sell
        CallPut: 1.Call  2.Put
        option_type: 1.European  2.American 
        barrier_type: 1.Vanilla  2.Up In  3.Down In  4.Up out  5.Down out

        barrier: 障礙價格

        model: 1.Black-scholes  2.Heston

        r: 本國(交易幣別)利率 - 外國(交易幣別)利率
        q: 股利殖利率
        sigma: 波動率

        xi: 波動率的波動率
        rho: 相關性 (模型中兩個布朗運動之間的相關性)
        v0: 變異數的起始值
        theta: 長期價格變異數
        kappa: 均數復歸 (回歸到長期價格變異數的反轉速度)

        steps: time step (切幾次時間間隔)
        N: 模擬情境次數
        
        npv: 淨現值
        price: 價格
        value: 權利金'''


    '''選擇障礙條件'''
    def Barrier_type(self, barrier_type1, barrier_type2):
        if barrier_type1 == 1:#單純
            barrier_type = 1
        elif barrier_type1 == 2:#觸及生效
            if barrier_type2 == 1:#上漲生效
                barrier_type = 2
            else:#下跌生效
                barrier_type = 3
        else:#觸及失效
            if barrier_type2 == 1:#上漲失效
                barrier_type = 4
            else:#下跌失效
                barrier_type = 5
        
        return barrier_type


    '''Black-scholes'''
    #Monte Carlo模擬未來可能的價格，其中利率、殖利率、波動率不變
    def BS(self, S, T, r, q, sigma, steps, N):
        dt = T / steps#每一步的時間
        
        #np.random.normal(): [N(情境) x step(時間)] 的隨機常態分配值
        #np.cumsum(): 累加 (axis = 1，按row累加)
        St = S *  np.exp(np.cumsum(((r - q - sigma**2 / 2) * dt +
                                    sigma * np.sqrt(dt) * np.random.normal(size=(N, steps))),
                                    axis=1))
        
        return St


    '''Heston'''
    #Monte Carlo模擬未來可能的價格，其中利率、殖利率不變，波動率隨時間改變
    def Heston(self, S, T, r, q, vt, rho, kappa, theta, xi, steps, N):
        dt = T / steps
        St = S
        prices = np.zeros((N, steps))
        for t in range(steps):
            #各種情境在t時的價格變化隨機值
            WT = np.random.multivariate_normal(np.array([0,0]), 
                                            cov = np.array([[1,rho],
                                                            [rho,1]]),
                                            size = N)
            #在t時的價格
            St = St *  np.exp((r - q - vt / 2) * dt + np.sqrt(vt) * np.sqrt(dt) * WT[:, 0])
            #在t時的波動率
            vt = np.abs(vt + (kappa * (theta - vt) * dt + xi * np.sqrt(vt) * np.sqrt(dt) * WT[:,1]))

            prices[:, t] = St
            
        return prices


    '''選擇模擬股價路徑的模型'''
    def simulation_model(self, model, S, T, r, q, sigma, vt, rho, kappa, theta, xi, steps, N):
        if model == 1:#Black-scholes
            prices = self.BS(S, T, r, q, sigma, steps, N)
        else:#heston
            prices = self.Heston(S, T, r, q, vt, rho, kappa, theta, xi, steps, N)
            
        return prices


    '''歐式普通期權的損益'''
    '''ST: 到期日的股價'''
    def European_Payoff(self, CallPut, ST, K):
        if CallPut == 1:#Call
            payoff = np.maximum(ST - K, 0)#逐位比較取其大者
        else:#Put
            payoff = np.maximum(K - ST, 0)
            
        return payoff


    '''歐式障礙期權的損益'''
    '''ST: 到期日的股價'''
    def European_barrier_Payoff(self, barrier_type, ST, barrier, payoff):
        if (barrier_type == 2):#上漲生效
            flag = (ST >= barrier).astype(int)#大於等於barrier才計算payoff
        elif (barrier_type == 3):#下跌生效
            flag = (ST <= barrier).astype(int)#小於等於barrier才計算payoff
        elif (barrier_type == 4):#上漲失效
            flag = (ST < barrier).astype(int)#小於barrier才計算payoff
        elif (barrier_type == 5):#下跌失效
            flag = (ST > barrier).astype(int)#大於barrier才計算payoff
            
        payoff = payoff * flag
        
        return payoff


    '''買/賣選擇權'''
    def BuySell(self, npv, direction):
        if direction == 1:#Buy
            npv = -npv
        else:#Sell
            npv = npv
            
        return npv


    '''可能的情境結果'''
    def Scenario(self, barrier_type, CallPut, direction):
        barrer_id = {
            # VanillaOption
            '1': {'cond1':{'condi':None, 'discribe': None},
                'cond2':{'condi':None, 'discribe': None}},
            # Barrier_UpIn
            '2': {'cond1':{'condi':'1', 'discribe': '1'},
                'cond2':{'condi':'3', 'discribe': '2'}},
            # Barrier_DownIn
            '3': {'cond1':{'condi':'2', 'discribe': '1'},
                'cond2':{'condi':'4', 'discribe': '2'}},
            # Barrier_UpOut
            '4': {'cond1':{'condi':'1', 'discribe': '3'},
                'cond2':{'condi':'3', 'discribe': '4'}},
            # Barrier_DownOut
            '5': {'cond1':{'condi':'2', 'discribe': '3'},
                'cond2':{'condi':'4', 'discribe': '4'}}
            }

        CallPut_id = { 
        '1': {'1': {'case1':'1', 'case2':'2'}, # Call # Buy
            '2':{'case1':'3', 'case2':'4'}}, # Put
        '2': {'1': {'case1':'5', 'case2':'6'}, # Call # Sell
            '2':{'case1':'7', 'case2':'8'}} # Put
        }

        condition         = barrer_id[barrier_type]
        condition['case'] = CallPut_id[direction][CallPut]

                
        return condition

        
    '''歐式定價'''
    def European_Pricing(self, startdate, maturity, S, K, p, direction, CallPut, barrier_type,
                        barrier, model, steps, N, r, q, sigma, xi, rho, v0, theta, kappa):
        T = (maturity - startdate).days / 365# 到期時間(年)
        
        St = self.simulation_model(model, S, T, r, q, sigma, v0, rho, kappa, theta, xi, steps, N)
        payoff = self.European_Payoff(CallPut, St[:, -1], K)
        if barrier_type != 1:#障礙期權
            payoff = self.European_barrier_Payoff(barrier_type, St[:, -1], barrier, payoff)
        npv = np.mean(payoff) * np.exp(-r * T)#payoff折現
        npv = self.BuySell(npv, direction)
        price = npv / S
        value = price * p
        
        return round(npv, 4), round(price * 100, 4), round(value, 2)


    '''設定美式買權或賣權'''
    def set_ql_CallPut(self, CallPut):
        if CallPut == 1:#買權
            CallPut = ql.Option.Call
        else:#賣權
            CallPut = ql.Option.Put
            
        return CallPut


    '''設定美式界線條件'''
    def set_ql_barrier(self, barrier_type):
        if barrier_type == 1:
            return
        if barrier_type == 2:#上漲生效
            barrier_type1 = ql.Barrier.UpIn
        elif  barrier_type == 3:#下跌生效
            barrier_type1 = ql.Barrier.DownIn
        elif  barrier_type == 4:#上漲失效
            barrier_type1 = ql.Barrier.UpOut
        else:#下跌失效
            barrier_type1 = ql.Barrier.DownOut
        
        return barrier_type1


    '''設定美式選擇權類型'''
    def set_ql_option(self, barrier_type, payoff, exercise_type, barrier_type1, barrier, rebate):
        if barrier_type == 1: #Vanilla
            option = ql.VanillaOption(payoff, exercise_type)
        else: #Barrier
            option = ql.BarrierOption(barrier_type1, barrier, rebate, payoff, exercise_type)
            
        return option


    '''設定美式模型'''
    def set_ql_model(self, initialValue, dividendTS, riskFreeTS, volatilityTS, v0, kappa, theta, xi, rho):
        process = ql.BlackScholesMertonProcess(initialValue, dividendTS, riskFreeTS, volatilityTS)
            
        return process


    '''設定美式計算引擎'''
    def set_ql_engine(self, barrier_type, process, steps, N):
        if barrier_type == 1: #Vanilla
            rng = "pseudorandom"
            engine = ql.MCAmericanEngine(process, rng, steps, requiredSamples = N)
                
        else: #Barrier
            engine = ql.AnalyticBarrierEngine(process)
        
        return engine

    '''美式定價'''
    def American_pricing(self, startdate, maturity, S, K, p, direction, CallPut, barrier_type,
                        barrier, steps, N, r, q, sigma, xi, rho, v0, theta, kappa):
        startdate = ql.Date(startdate.day, startdate.month, startdate.year) #定價日
        maturity = ql.Date(maturity.day, maturity.month, maturity.year) #到期日
        rebate = 0

        CallPut = self.set_ql_CallPut(CallPut)#設定買權或賣權
        barrier_type1 = self.set_ql_barrier(barrier_type)#設定界線條件
        exercise_type = ql.AmericanExercise(startdate, maturity) #設定美式選擇權

        ql.Settings.instance().evaluationDate = startdate#設定定價日
        payoff = ql.PlainVanillaPayoff(CallPut, K)#算payoff
        #設定選擇權類型
        option = self.set_ql_option(barrier_type, payoff, exercise_type, barrier_type1, barrier, rebate)

        # 假定無風險利率、股利殖利率和波動率曲線是平的(不隨時間改變)
        riskFreeTS = ql.YieldTermStructureHandle(ql.FlatForward(startdate, r, ql.Actual365Fixed()))
        dividendTS = ql.YieldTermStructureHandle(ql.FlatForward(startdate, q, ql.Actual365Fixed()))
        volatilityTS = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(startdate, ql.NullCalendar(), sigma, ql.Actual365Fixed()))

        #初始化Black scholese過程
        initialValue = ql.QuoteHandle(ql.SimpleQuote(S))
        process = self.set_ql_model(initialValue, dividendTS, riskFreeTS, volatilityTS, 
                            v0, kappa, theta, xi, rho)#設定模型
        engine = self.set_ql_engine(barrier_type, process, steps, N)#設定計算引擎
        option.setPricingEngine(engine)#對選擇權權設定計算引擎

        npv = option.NPV()
        npv = self.BuySell(npv, direction)
        price = npv / S
        value = price * p
        
        return round(npv, 4), round(price * 100, 4), round(value, 2)



class TARF():
    '''Parameter'''
    '''
        N: 模擬情境次數

        ***交易參數***
        trade_date: 交易日
        S: 即期價格

        ***全球界線***
        boundary_type1: 界線  (1. 無  2. 單一界線  3. 雙重界線)
        boundary_type2: 界線類型  (boundary_type1 = 2 -> 1. 上漲生效  2. 下跌生效  3. 上漲失效  4. 下跌失效  、  boundray_type1 = 3 -> 1. DKI  2. DKO)
        boundary: 界線值
        boundary_up: 上限水準
        boundary_down: 下限水準
        global_trigger: 界線回饋  (1. 無  2. 固定金額)
        rebate: 回饋金額

        ***價內目標***
        itm_type: 目標類型  (1. 無  2. 大數  3. 現金)
        itm_boundary: 目標值
        itm_trigger: 觸價付款  (1. 全額付款  2. 上限付款  3. 無支付)

        ***價外目標***
        otm_type: 目標類型  (1. 無  2. 大數  3. 現金)
        otm_boundary: 目標值
        otm_trigger: 觸價付款  (1. 全額付款  2. 上限付款  3. 無支付)

        ***客製化報酬***
        buy_sell: 方向  (1. buy  2. sell)
        call_put: 選擇權類型  (1. call  2.put  3. 數位call  4. 數位put)
        p: 本金
        itm: 價內/價外  (1. 價內  2. 價外 )
        barrier_type: 界線  (1. 無  2. 上漲生效  3. 下跌生效  4. 上漲失效  5. 下跌失效 )
        barrier: 界線水準

        ***定價日程***
        settledate: 比價日 
        r: 本金幣別利率
        f: 交易幣別利率
        sigma: 波動率
        K: 履約價
        
        ***評價***
        value_type: 權利金幣別

        ***評價結果***
        value: 權利金
    '''


    '''單期Black-scholes'''
    #Monte Carlo模擬未來可能的價格，其中利率、殖利率、波動率不變
    def BS(self, S, T, r, f, sigma, steps, N):
        dt = T / steps#每一步的時間
        
        #np.random.normal(): [N(情境) x step(時間)] 的隨機常態分配值
        #np.cumsum(): 累加 (axis = 1，按row累加)
        St = S *  np.exp(np.cumsum(((r - f - sigma**2 / 2) * dt +
                                    sigma * np.sqrt(dt) * np.random.normal(size=(N, steps))),
                                    axis=1))
        
        return St


    '''多期Black-scholes'''
    #每期用其對應的波動率和利率來模擬股價路徑，shape = (期數, 路徑數, 合約總天數)
    def mutiple_period_BS(self, S, T, r, f, sigma, steps, N):
        # print((S, T[-1], r[0], f[0], sigma[0], steps[-1], N)
        St = self.BS(S, T[-1], r[0], f[0], sigma[0], steps[-1], N).reshape(1, N, -1)
        for i in range(1, len(r)):
            St1 = self.BS(S, T[-1], r[i], f[i], sigma[i], steps[-1], N).reshape(1, N, -1)
            St = np.vstack([St, St1])
        
        return St


    '''普通期權的損益'''
    def Spread(self, call_put, St, K, S):
        if call_put == 1:#Call
            spread = max(St - K, 0)#逐位比較取其大者
        elif call_put == 2:#Put
            spread = max(K - St, 0)
        elif call_put == 3:#數位Call
            spread = St if (St - K) > 0 else 0
        else:#數位Put
            spread = St if (K - St) > 0 else 0
            
        return spread


    '''障礙期權的損益'''
    def Barrier_Payoff(self, barrier_type, St, barrier, spread):
        if (barrier_type == 1):#無
            return spread
        elif (barrier_type == 2):#上漲生效
            #flag = 1才計算payoff
            flag = 1 if (St >= barrier) else 0
        elif (barrier_type == 3):#下跌生效
            flag = 1 if (St <= barrier) else 0
        elif (barrier_type == 4):#上漲失效
            flag = 1 if (St < barrier) else 0
        elif (barrier_type == 5):#下跌失效
            flag = 1 if (St > barrier) else 0
            
        spread = spread * flag
        
        return spread


    '''買/賣選擇權'''
    #未來payoff為正，現在要付錢
    def BuySell(self, payoff, buy_sell):
        if buy_sell == 1:# buy
            payoff *= -1
            
        return payoff


    '''計算當期損益'''
    def Payoff(self, p, spread, payoff):
        try:
            P = p[np.where(spread != 0)[0][0]]
        except:
            P = 0
        payoff *= P#價差＊本金=實際payoff
        
        return payoff, P


    '''全球界線'''
    def Global_boundary(self, St, boundary_type1, boundary_type2, boundary, boundary_up, boundary_down):
        KO = 1
        if boundary_type1 == 1:#無
            return KO

        elif boundary_type1 == 2:#單一界線
            if boundary_type2 == 1:#上漲生效
                if St < boundary:
                    KO = 0
            elif boundary_type2 == 2:#下跌生效
                if St > boundary:
                    KO = 0
            elif boundary_type2 == 3:#上漲失效
                if St >= boundary:
                    KO = 0
            else:#下跌失效
                if St <= boundary:
                    KO = 0
        else:#雙重界線
            if boundary_type2 == 1:#DKI
                if (St < boundary_up) and (St > boundary_down):
                    KO = 0
            else:#DKO
                if (St >= boundary_up) or (St <= boundary_down):
                    KO = 0
        
        return KO
        

    '''價內/價外目標'''
    def ITM_OTM_Target(self, target_type, accu_payoff, boundary, p):
        KO, excess = 1, 0
        if (target_type == 2) and (accu_payoff >= boundary * p):#大數
            KO = 0
            excess = accu_payoff - (boundary * p)#正值

        elif (target_type == 3) and (accu_payoff >= boundary):#現金
            KO = 0
            excess = accu_payoff - boundary#正值

        return KO, excess


    '''觸價時payoff處理方法'''
    def trigger(self, KO1, KO2, KO3, global_trigger, itm_trigger, otm_trigger, payoff, excess, rebate, KI_flag, boundary_type1, boundary_type2):
        if (KO1 == 0):#全球界線
            if global_trigger == 1:#無
                payoff = 0
            elif global_trigger == 2:#固定金額
                if ((boundary_type1 != 2) or (boundary_type2 not in [1, 2])) and ((boundary_type1 != 3) or (boundary_type2 != 1)):#KI要進場才能拿rebate
                    payoff = -rebate #未來賺到錢，現在要付錢
                else:
                    payoff = 0

        elif (KI_flag == 0) and (KO1 == 1) and (((boundary_type1 == 2) and (boundary_type2 in [1, 2])) or ((boundary_type1 == 3) and (boundary_type2 == 1))):
            if global_trigger == 2:#固定金額
                KI_flag = 1

        elif KO2 == 0:#價內目標
            if itm_trigger == 2:#上限付款
                payoff += excess # 該期payoff(正) = 原該期payoff(正) - (獲得)超出的部分(正)  #未來賺錢，現在付錢，payoff已轉為負 -> payoff(負) + excess(正) ->賺的比較少
            elif itm_trigger == 3:#無支付
                payoff = 0

        elif KO3 == 0:#價外目標
            if otm_trigger == 2:#上限付款
                payoff -= excess# 該期payoff(負) = 原該期payoff(負) + (不用付)超出的部分(正)  #未來賠錢，現在收錢，payoff已轉為正 -> payoff(正) - excess(正) ->賠得比較少
            elif otm_trigger == 3:#無支付
                payoff = 0
        
        return payoff, KI_flag


    '''模擬情境'''
    def Scenario(self, N, St, t, steps, payoff1):
            scenario = random.randint(0, N)#隨機挑選一個模擬情境來呈現
            s = St[t, scenario][steps-1]#股價
            payoff = (-payoff1[scenario])#損益

            accum = np.zeros(len(payoff))#累積損益
            itm_accum = np.zeros(len(payoff))#價內累積損益
            otm_accum = np.zeros(len(payoff))#價外累積損益
            for i in range(len(payoff)):
                accum[i] = payoff[i]
                if payoff[i] > 0:
                    itm_accum[i] = payoff[i]
                else:
                    otm_accum[i] = payoff[i]
            accum = np.cumsum(accum)
            itm_accum = np.cumsum(itm_accum)
            otm_accum = np.cumsum(otm_accum)
            
            scenario = list()
            for i in range(len(s)):
                scenario.append({
                    "term": str(i+1),
                    "spot": "{:.4f}".format(s[i]),
                    "payoff": "{:.2f}".format(payoff[i]), 
                    "accum_payoff": "{:.2f}".format(accum[i]),
                    "itm_accum_payoff": "{:.2f}".format(itm_accum[i]),
                    "otm_accum_payoff": "{:.2f}".format(otm_accum[i])
                    })
                    
            return scenario


    '''計算權利金'''
    def TARF_Pricing(self, N = 20000, S = 6.466, trade_date = datetime(2021, 8, 30, 0, 0),
                    boundary_type1 = 1, boundary_type2 = 1, boundary = 0, boundary_up = 0, boundary_down = 0, global_trigger = 1, rebate = 0,
                    itm_type = 1, itm_boundary = 0, itm_trigger = 1, otm_type = 1, otm_boundary = 0, otm_trigger = 1,
                    buy_sell = [1,2], call_put = [1,2], p = [1000000, 2000000], itm = [1, 2], barrier_type = [1, 1], barrier = [0, 0],
                    settledate = [datetime(2021, 9, 30), datetime(2021, 11, 1)],
                    r = [0.03087458, 0.02979089], f = [0.00049300, 0.00050600], sigma = [0.04033, 0.04033],
                    K = [[6.3922, 6.3922], [6.3922, 6.3922]], value_type = 1):
                
        steps = np.array([(date - trade_date).days for date in settledate])#每一期距離交易日幾天
        T = steps / 365#期間單位由天轉為年

        St = self.mutiple_period_BS(S, T, r, f, sigma, steps, N)

        total = 0
        for t in tqdm(range(len(steps))): #第幾期
            rebate_flag = np.ones(N) #偵測KI用
            payoff1 = np.zeros([N, t+1])#存放每期payoff
            for n in range(N):#第幾個情境
                ITM_accu_payoff, OTM_accu_payoff = 0, 0#價內/外累計目標
                KO1, KO2, KO3 = 1, 1, 1
                excess = 0
                KI_flag = 0
                for j in range(t+1):#每一期的sigma和rate都不同，payoff須從第一期開始重算
                    spread = np.zeros(len(p))#存放重算的payoff，但僅使用最後一期(t)之payoff
                    for i in range(len(p)): #第幾隻腳
                        spread[i] = self.Spread(call_put[i], St[t, n, steps[j]-1], K[j][i], S)
                        spread[i] = self.Barrier_Payoff(barrier_type[i], St[t, n, steps[j]-1], barrier[i], spread[i])
                        spread[i] = self.BuySell(spread[i], buy_sell[i])#未來payoff為正，現在要付錢，payoff為負

                    payoff1[n, j] = sum(spread)#加總每隻腳的payoff，產生當期payoff
                    payoff1[n, j], P = self.Payoff(p, spread, payoff1[n, j])
                    

                    if KI_flag == 0:#KI觸價後，後面期數不再有KI
                        KO1 = self.Global_boundary(St[t, n, steps[j]-1], boundary_type1, boundary_type2, boundary, boundary_up, boundary_down)

                    if (itm_type != 1) and (-payoff1[n, j] > 0):#價內，前面把payoff轉為負，須轉換回來
                        ITM_accu_payoff += -payoff1[n, j]
                        KO2, excess = self.ITM_OTM_Target(itm_type, ITM_accu_payoff, itm_boundary, P)

                    if (otm_type != 1) and (payoff1[n, j] > 0):#價外
                        OTM_accu_payoff += payoff1[n, j]
                        KO3, excess = self.ITM_OTM_Target(otm_type, OTM_accu_payoff, otm_boundary, P)

                    payoff1[n, j], KI_flag = self.trigger(KO1, KO2, KO3, global_trigger, itm_trigger, otm_trigger, payoff1[n, j], excess, rebate, KI_flag, boundary_type1, boundary_type2)

                    if (KO1 == 0) and (((boundary_type1 != 2) or (boundary_type2 not in [1, 2])) and ((boundary_type1 != 3) or (boundary_type2 != 1))):
                        break
                    elif (KO2 == 0) or (KO3 == 0):#出場後面期數都不算
                        break

                if (KI_flag == 1) and (global_trigger == 2) and (((boundary_type1 == 2) and (boundary_type2 in [1, 2])) or ((boundary_type1 == 3) and (boundary_type2 == 1))):
                    rebate_flag[n] = 0#如果KI觸價，該情境不給rebate

            if (t == (len(steps)-1)) and (global_trigger == 2) and (((boundary_type1 == 2) and (boundary_type2 in [1, 2])) or ((boundary_type1 == 3) and (boundary_type2 == 1))):
                payoff1[:, t] = payoff1[:, t] + rebate_flag * -rebate#KI沒觸價，最後一期給rebate

            value = np.mean(payoff1[:, t]) * np.exp(-(r[t]-f[t]) * T[t])#貼現
            total += value

        total = round(total/S, 2) if value_type == 1 else round(total, 2)
        scenario = self.Scenario(N, St, t, steps, payoff1)

        result = {"value" : "{:.2f}".format(total),
                  "scenario" : scenario,
                  }
        
        return result



class database():
    def __init__(self):
        self.con_syting = "mysql+pymysql://{user}:{pw}@{localhost}:{port}/{db}"
        self.engine   = create_engine(self.con_syting.format(user="XXXX",
                                                               pw="XXXX",
                                                               localhost="XXXX",
                                                               db="XXXX",
                                                               port=3306))
        self.currency = {
            1:  {"CcyCode": "AUD", "BaseCcy": "AUD", "TermCcy": "USD"},#Australia
            2:  {"CcyCode": "EUR", "BaseCcy": "EUR", "TermCcy": "USD"},#Europe
            3:  {"CcyCode": "GBP", "BaseCcy": "GBP", "TermCcy": "USD"},#U.K.
            4:  {"CcyCode": "NZD", "BaseCcy": "NZD", "TermCcy": "USD"},#New Zealand
            5:  {"CcyCode": "CAD", "BaseCcy": "USD", "TermCcy": "CAD"},#Canada
            6:  {"CcyCode": "CHF", "BaseCcy": "USD", "TermCcy": "CHF"},#Switzerland
            7:  {"CcyCode": "CNH", "BaseCcy": "USD", "TermCcy": "CNH"},#China 境外
            8:  {"CcyCode": "CNY", "BaseCcy": "USD", "TermCcy": "CNY"},#China 境內
            9:  {"CcyCode": "HKD", "BaseCcy": "USD", "TermCcy": "HKD"},#Hong Kong
            10: {"CcyCode": "JPY", "BaseCcy": "USD", "TermCcy": "JPY"},#Japan
            11: {"CcyCode": "SGD", "BaseCcy": "USD", "TermCcy": "SGD"},#Singapore
            12: {"CcyCode": "TWD", "BaseCcy": "USD", "TermCcy": "TWD"},#Taiwan
            13: {"CcyCode": "ZAR", "BaseCcy": "USD", "TermCcy": "ZAR"},#South Africa
            }

    '''獲取合約中的即期匯率'''
    def get_spot(self, exchange, date):
        exchange = self.currency[exchange]["CcyCode"]
        while True:
            sqrl = '''SELECT st.* FROM Currency_CDP st
                    WHERE CcyCode = "{}" AND PriceDate = "{}"'''\
                    .format(exchange, str(date))
            df = pd.read_sql(sqrl, self.engine)
            df.drop_duplicates(subset=['BaseCcy', 'TermCcy','PriceDate'], inplace = True)
            try:
                S = float(df["Price"][0])
                print(f"日期：{date} 幣別：{exchange} 匯率：{S}")
                break
            except:
                date = (date - timedelta(days = 1))

        return S

    
    '''計算合約中的波動率'''
    def get_sigma(self, exchange, trade_date, end_date):
        exchange = self.currency[exchange]["CcyCode"]
        day_counts = (end_date - trade_date).days #計算合約天數
        end = trade_date - timedelta(days = 1)
        start = end - timedelta(days = day_counts)#回推相同天數
        addition = start - timedelta(days = 10)#若期間都在假日會抓不到資料，往前多抓幾天

        sqrl = '''SELECT st.* FROM Currency_CDP st
                WHERE CcyCode = "{}" AND PriceDate >= "{}" AND PriceDate <= "{}"'''\
                .format(exchange, str(addition), str(end))
        df = pd.read_sql(sqrl, self.engine)
        df.drop_duplicates(subset=['BaseCcy', 'TermCcy','PriceDate'], inplace = True)
        df = df[["PriceDate", "Price"]]

        #填補缺失的日期
        df["PriceDate"] = pd.to_datetime(df["PriceDate"])
        date_check = pd.date_range(start = addition, end = end, freq = "D").rename("PriceDate")# 創建完整時間軸
        df = pd.DataFrame(date_check).merge(df, on = "PriceDate", how = "left")# 填補缺失的日期
        df = df.fillna(method='ffill').fillna(method='bfill')# 填補缺失值

        #計算歷史波動率
        df["Price"] = df["Price"].astype(float)
        df["PreClose"] = df["Price"].shift(1)
        df.dropna(inplace = True)

        df = df[df["PriceDate"].dt.date >= start]# 抓出需要的時間

        df["Price%"] = np.log(df["Price"]) - np.log(df["PreClose"])# 價格百分比
        sigma = np.std(df["Price"], ddof=1)*(252**0.5) # 年化標準差
        print(f"日期：{start}~{end} 幣別：{exchange} 波動率：{sigma}")

        return sigma


    '''獲取DB的利率'''
    def get_DB_rate(self, exchange, code, term, date):
        exchange = self.currency[exchange][code]
        while True:
            sqrl = '''SELECT bond.BondName, yield.Price, yield.PriceDate FROM 
                    (Country_Ccy ccy LEFT JOIN Bond_List bond ON ccy.Region = bond.Region)
                    LEFT JOIN Bond_Yield yield ON bond.Bond_ID = yield.Bond_ID
                    WHERE ccy.CcyCode = "{}" AND bond.Term = "{}" AND yield.PriceDate = "{}"'''\
                    .format(exchange, term, str(date))
            df = pd.read_sql(sqrl, self.engine)
            df.drop_duplicates(subset=['BondName', 'PriceDate'], inplace = True)
            try:
                rate = float(df["Price"][0])
                print(f'日期：{date} 幣別：{exchange} 債券：{df["BondName"][0]} 利率：{df["Price"][0]}')
                break
            except:
                date = (date - timedelta(days = 1))
                
        return rate
    

    '''獲取合約中的利率'''
    def get_rate(self, trade_date, end_date, exchange, code):
        day_counts = (end_date - trade_date).days #計算合約天數

        #獲取DB中該幣別的國債種類(不同到期日)
        ex = self.currency[exchange]["CcyCode"]
        sqrl = '''SELECT bond.* FROM 
                (Country_Ccy ccy LEFT JOIN Bond_List bond ON ccy.Region = bond.Region)
                WHERE ccy.CcyCode = "{}"'''.format(ex)
        df = pd.read_sql(sqrl, self.engine)

        #將國債到期日轉為天數存進dictionary，ex: 1Y->360
        Unit = {"W": 7, "M": 30, "Y": 360}
        term = dict()
        for i in set(df["Term"]):
            if i == "Overnight":
                continue
            num = int(i[:-1])
            unit = i[-1]
            days = num * Unit[unit]
            term[days] = i

        #抓出合約天數位於哪兩種國債期間中，以做後續插補
        term1 = sorted(term)
        period = [0, term1[0]]#使用的插補區間[start, end]
        for i in range(len(term1)):
            if day_counts <= term1[i]:
                if i > 0:
                    period = [term1[i-1], term1[i]]
                break
            period = [term1[i], term1[i]]#超過資料庫的最長國債期間
        period1 = period[1] - period[0]#兩種國債的天數差
        period2 = day_counts - period[0]#合約與到期日較短的國債之天數差

        #兩種國債期間的利率價差
        if period[0] != 0:
            start = self.get_DB_rate(exchange, code, term[period[0]], trade_date)
        else: 
            start = 0
        end = self.get_DB_rate(exchange, code, term[period[1]], trade_date)
        spread = end - start

        #插補
        if period1 != 0:
            rate = start + (spread / period1 * period2)
        else:
            rate = start

        return rate
    
    '''關閉read_sql engine的DB連線'''
    def close_DB(self):
        self.engine.dispose()
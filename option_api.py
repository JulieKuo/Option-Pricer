import option_pricing as opp
import numpy as np
import datetime

class OptionPricerController():
    def Basic_Pricing(self, r_data):
        get_data = opp.database()

        # r_data        = request.get_json()
        startdate     = datetime.datetime.strptime(r_data['start_date'], "%Y-%m-%d").date()
        maturity      = datetime.datetime.strptime(r_data['maturity'], "%Y-%m-%d").date()
        exchange      = int(r_data['exchange'])
        if r_data['spot'] == "":
            S         = get_data.get_spot(exchange, startdate)
        else:
            S         = float(r_data['spot'])
        K             = float(r_data['strike'])
        p             = int(r_data['principle'])
        direction     = int(r_data['direction'])
        CallPut       = int(r_data['call_put'])
        option_type   = int(r_data['option_type'])
        barrier_type1 = int(r_data['barrier_type1'])
        barrier_type2 = int(r_data['barrier_type2'])
        barrier       = float(r_data['barrier'])
        model         = int(r_data['model'])
        steps         = 100
        N             = 100000

        if r_data['r'] == "":
            r = get_data.get_rate(startdate, maturity, exchange, "TermCcy")
        else:
            r         = float(r_data['r'])

        if r_data['f'] == "":
            f = get_data.get_rate(startdate, maturity, exchange, "BaseCcy")
        else:
            f         = float(r_data['f'])

        spread        = r - f
        q             = 0

        if r_data['sigma'] == "":
            sigma     = get_data.get_sigma(exchange, startdate, maturity)
        else:
            sigma     = float(r_data['sigma'])

        xi            = float(r_data['xi'])
        rho           = float(r_data['rho'])
        v0            = float(r_data['v0'])
        theta         = float(r_data['theta'])
        kappa         = float(r_data['kappa'])

        spread *= 0.01; sigma *= 0.01; xi *= 0.01
        pricer = opp.basic_option()
        barrier_type = pricer.Barrier_type(barrier_type1, barrier_type2)

        if option_type == 1:#歐式
            npv, price, value = pricer.European_Pricing(startdate, maturity, S, K, p, direction,
                                                                            CallPut, barrier_type, barrier, model, steps, N,
                                                                            spread, q, sigma, xi, rho, v0, theta, kappa)
        else:#美式
            npv, price, value = pricer.American_pricing(startdate, maturity, S, K, p, direction,
                                                                            CallPut, barrier_type, barrier, steps, N,
                                                                            spread, q, sigma, xi, rho, v0, theta, kappa)

        scenario = pricer.Scenario(str(barrier_type), str(CallPut), str(direction))

        param = {
            "spot": "{:.4f}".format(S),
            "r": "{:.3f}".format(r),
            "f": "{:.3f}".format(f),
            "sigma": "{:.3f}".format(sigma*100)
            }

        result = {
            'npv' : "{:.3f}".format(npv),
            'price' : "{:.3f}".format(price) ,
            'value': "{:.2f}".format(value),
            'scenario': scenario,
            'param': param
            }
        
        
        get_data.close_DB()#關閉SQL連線

        return result



    def TARF_Pricing(self, r_data):
        get_data = opp.database()

        # r_data     = request.get_json()
        exchange   = int(r_data["exchange"])
        N          = 20000
        trade_date = datetime.datetime.strptime(r_data["trade_date"], "%Y-%m-%d").date()
        s_flag     = int(r_data["s_flag"])
        if s_flag == 1:#使用資料庫
            S = get_data.get_spot(exchange, trade_date)
        else:#自訂
            S = float(r_data["s"])

        boundary_type1 = int(r_data["global_boundary"]["boundary_type1"])
        boundary_type2 = int(r_data["global_boundary"]["boundary_type2"])
        boundary       = float(r_data["global_boundary"]["boundary"])
        boundary_up    = float(r_data["global_boundary"]["boundary_up"])
        boundary_down  = float(r_data["global_boundary"]["boundary_down"])
        global_trigger = float(r_data["global_boundary"]["global_trigger"])
        rebate         = float(r_data["global_boundary"]["rebate"]) * S

        itm_type     = int(r_data["itm"]["itm_type"])
        itm_boundary = float(r_data["itm"]["itm_boundary"])
        if itm_type == 3:#價內目標選現金，須將本金轉換為交易幣別
            itm_boundary *= S
        itm_trigger  = int(r_data["itm"]["itm_trigger"])

        otm_type     = int(r_data["otm"]["otm_type"])
        otm_boundary = float(r_data["otm"]["otm_boundary"])
        if otm_type == 3:#價外目標選現金，須將本金轉換為交易幣別
            otm_boundary *= S
        otm_trigger  = int(r_data["otm"]["otm_trigger"])

        buy_sell, call_put, p, itm, barrier_type, barrier = [], [], [], [], [], []
        for obj in r_data["options"]:
            buy_sell.append(int(obj["buy_sell"]))
            call_put.append(int(obj["call_put"]))
            p.append(float(obj["p"]))
            itm.append(int(obj["itm"]))
            barrier_type.append(int(obj["barrier_type"]))
            barrier.append(float(obj["barrier"]))

        settledate, r, f, sigma, K  = [], [], [], [], []
        for obj in r_data["dates"]:
            settledate.append(datetime.datetime.strptime(obj["settledate"], "%Y-%m-%d").date())
            r.append(float(obj["r"]))
            f.append(float(obj["f"]))
            sigma.append(float(obj["sigma"]))
            K.append(list(map(float, obj["k"])))

        rate_sigma_flag = int(r_data["rate_sigma_flag"])
        if rate_sigma_flag == 0:#使用資料庫
            r, f = [], []
            for date in settledate:
                f1 = get_data.get_rate(trade_date, date, exchange, "BaseCcy")
                r1 = get_data.get_rate(trade_date, date, exchange, "TermCcy")
                f.append(f1)
                r.append(r1)

            sigma = get_data.get_sigma(exchange, trade_date, settledate[-1])
            sigma = [sigma] * len(settledate)

        r     = np.array(r) * 0.01
        f     = np.array(f) * 0.01
        sigma = np.array(sigma) * 0.01

        value_type      = int(r_data["value_type"])


        pricer = opp.TARF()
        result = pricer.TARF_Pricing(N, S , trade_date, boundary_type1, boundary_type2, boundary, boundary_up, boundary_down,
                    global_trigger, rebate, itm_type, itm_boundary, itm_trigger, otm_type, otm_boundary,
                    otm_trigger, buy_sell, call_put, p, itm, barrier_type, barrier, settledate, r, f, sigma, K, value_type)

        rate = list()
        for i in range(len(r)):
            rate.append({
                "r": "{:.3f}".format(r[i]*100),
                "f": "{:.3f}".format(f[i]*100),
                "sigma": "{:.3f}".format(sigma[i]*100),
            })
            
        result["param"] = {
            "spot": "{:.4f}".format(S),
            "rate": rate,
        }

        get_data.close_DB()

        return result


if __name__ == '__main__':
    basic_rdata = {
        'start_date': '2021-10-21',
        'maturity': '2022-01-21',
        'exchange': '7',
        'spot': '6.3849',
        'strike': '6.4337',
        'principle': '1000000',
        'direction': '1',
        'call_put': '1',
        'option_type': '1', 
        'barrier_type1': '3',
        'barrier_type2': '1',
        'barrier': '6.7041',
        'model': '1',
        'f': '0.133',
        'r': '3.122',
        'sigma': '4.459',
        'xi': '26.944',
        'rho': '0.3371',
        'v0': '0.0020',
        'theta': '0.0038',
        'kappa': '5.0000'
    }

    tarf_rdata = {
        'exchange': '7',
        'trade_date': '2021-08-30',
        's_flag': '2',
        's': '6.466',
        'global_boundary': {
            'boundary_type1': '1',
            'boundary_type2': '1',
            'boundary': '0',
            'boundary_up': '0',
            'boundary_down': '0',
            'global_trigger': '1',
            'rebate': '0'
        },
        'itm': {
            'itm_type': '1', 
            'itm_boundary': '0', 
            'itm_trigger': '1'
        },
        'otm': {
            'otm_type': '1',
            'otm_boundary': '0',
            'otm_trigger': '1'
        },
        'options': [
            {
                'id': '1',
                'buy_sell': '1',
                'call_put': '1',
                'p': '1000000',
                'itm': '1',
                'barrier_type': '1',
                'barrier': '0'
            },
            {
                'id': '2',
                'buy_sell': '2',
                'call_put': '2',
                'p': '2000000',
                'itm': '2',
                'barrier_type': '1',
                'barrier': '0'
            }
        ],
        'dates': [
            {
                'term': '1',
                'settledate': '2021-09-30',
                'f': '0.049300',
                'r': '3.017187',
                'sigma': '4.033',
                'k': ['6.3922', '6.3922']
            },
            {
                'term': '2',
                'settledate': '2021-11-01',
                'f': '0.050600',
                'r': '2.909819',
                'sigma': '4.033',
                'k': ['6.3922', '6.3922']
            }
        ],
        'rate_sigma_flag': '1',
        'value_type': '1'
    }

    calculate = OptionPricerController()
    basic_result = calculate.Basic_Pricing(r_data = basic_rdata)
    tarf_result = calculate.TARF_Pricing(r_data = tarf_rdata)
    print(basic_result)
    print(tarf_result)
import pandas as pd
import numpy as np
from tqdm import tqdm

def calculate_all_factors(df):
    """
    因子计算主入口逻辑
    """
    print("开始计算因子...")
    # 1. 确保排序
    df = df.sort_values(['Symbol', 'TradingDate']).reset_index(drop=True)
    
    # 2. 预计算一些基础列
    # 如果你的 df 已经有 ret_1d，这一行可以省去，如果没有则加上
    if 'ret_1d' not in df.columns:
        df['ret_1d'] = df.groupby('Symbol')['ClosePrice'].pct_change()
    
    # --- 分类计算 (调用你定义的那些 add_xxx 函数) ---
    df = add_momentum_factors(df)    # F01-F20
    df = add_valuation_factors(df)   # F21-35
    df = add_quality_factors(df)     # F36-50
    df = add_risk_factors(df)        # F51-65
    df = add_sentiment_factors(df)   # F66-80
    
    # --- 3. 构建标签 (直接写在这里，不调用外部函数) ---
    print("正在生成预测标签 (Next-day Log Return)...")
    # 确保你的 df 里确实有 'Log_Ret' 这一列
    df['target_log_ret'] = df.groupby('Symbol')['Log_Ret'].shift(-1)
    
    print(f"所有因子计算完成！当前因子维度: {len([c for c in df.columns if 'factor_' in c])}")
    return df

def add_momentum_factors(df):
    print("正在计算：动量趋势类因子 (优化版：最大窗口 20d)...")
    
    # 基础分组准备
    if 'ret_1d' not in df.columns:
        df['ret_1d'] = df.groupby('Symbol')['ClosePrice'].pct_change()
    
    grp = df.groupby('Symbol')['ClosePrice']
    grp_high = df.groupby('Symbol')['HighPrice']
    grp_low = df.groupby('Symbol')['LowPrice']
    
    # --- F01-F04: N日收益率 (调整窗口为短中期) ---
    for n in [2, 5, 10, 20]:
        df[f'factor_mom_{n}d'] = grp.transform(lambda x: x.shift(1) / x.shift(n+1) - 1)
    
    # --- F05: 短期反转 ---
    df['factor_reversal_5d'] = grp.transform(lambda x: -(x.shift(1) / x.shift(6) - 1))
    
    # --- F06-F07: 价格位置 (删除250d，统一为10d和20d，增加min_periods) ---
    for n in [10, 20]:
        ln = grp_low.transform(lambda x: x.shift(1).rolling(n, min_periods=1).min())
        hn = grp_high.transform(lambda x: x.shift(1).rolling(n, min_periods=1).max())
        df[f'factor_pos_{n}d'] = (df['ClosePrice'].shift(1) - ln) / (hn - ln + 1e-9)

    # --- F08: 10日移动平均乖离率 (BIAS) ---
    ma_10 = grp.transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    df['factor_bias_10d'] = df['ClosePrice'].shift(1) / ma_10 - 1

    # --- F09: 均线交叉信号 (改为 5d/20d) ---
    ma_5 = grp.transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    ma_20 = grp.transform(lambda x: x.shift(1).rolling(20, min_periods=1).mean())
    df['factor_ma_cross'] = ma_5 / (ma_20 + 1e-9)

    # --- F10: MACD 柱值 (EWM自带处理初始值的功能，空值极少) ---
    ema_12 = grp.transform(lambda x: x.shift(1).ewm(span=12, adjust=False).mean())
    ema_26 = grp.transform(lambda x: x.shift(1).ewm(span=26, adjust=False).mean())
    diff = ema_12 - ema_26
    dea = diff.groupby(df['Symbol']).transform(lambda x: x.ewm(span=9, adjust=False).mean())
    df['factor_macd_hist'] = (diff - dea) * 2

    # --- F11: RSI (14日) ---
    def calc_rsi(s, n=14):
        delta = s.diff()
        up = delta.clip(lower=0).rolling(n, min_periods=1).mean()
        down = -delta.clip(upper=0).rolling(n, min_periods=1).mean()
        return 100 - (100 / (1 + up / (down + 1e-9)))
    df['factor_rsi_14d'] = grp.transform(lambda x: calc_rsi(x.shift(1)))

    # --- F12: 动量加速度 (对比 20d 和 10d 的速度差) ---
    df['factor_mom_accel'] = df['factor_mom_20d'] - df['factor_mom_10d']

    # --- F13: 价格路径平滑度 ---
    abs_ret_sum = df['ret_1d'].abs().groupby(df['Symbol']).transform(lambda x: x.shift(1).rolling(20, min_periods=1).sum())
    total_ret_abs = grp.transform(lambda x: (x.shift(1) / x.shift(21) - 1).abs())
    df['factor_path_smooth'] = total_ret_abs / (abs_ret_sum + 1e-9)

    # --- F14: 涨停因子 ---
    df['factor_limit_count'] = (df['ret_1d'] > 0.098).astype(int).groupby(df['Symbol']).transform(lambda x: x.shift(1).rolling(20, min_periods=1).sum())

    # --- F15: Aroon Up (缩减窗口至 20d) ---
    def calc_aroon(s, n=20):
        # 增加 min_periods=1
        return s.rolling(n, min_periods=1).apply(lambda x: (len(x) - 1 - x.argmax()) / (len(x) - 1) if len(x)>1 else 0, raw=False)
    df['factor_aroon_up'] = grp_high.transform(lambda x: calc_aroon(x.shift(1)))

    # --- F16: TRIX ---
    def calc_trix(s, n=12):
        ema1 = s.ewm(span=n, adjust=False).mean()
        ema2 = ema1.groupby(df['Symbol']).transform(lambda x: x.ewm(span=n, adjust=False).mean())
        ema3 = ema2.groupby(df['Symbol']).transform(lambda x: x.ewm(span=n, adjust=False).mean())
        return ema3.pct_change()
    df['factor_trix'] = grp.transform(lambda x: calc_trix(x.shift(1)))

    # --- F17: CCI (顺势指标) ---
    tp = (df['HighPrice'] + df['LowPrice'] + df['ClosePrice']) / 3
    tp_ma = tp.groupby(df['Symbol']).transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean())
    tp_md = tp.groupby(df['Symbol']).transform(lambda x: x.shift(1).rolling(14, min_periods=1).apply(lambda y: np.abs(y - y.mean()).mean(), raw=True))
    df['factor_cci'] = (tp.shift(1) - tp_ma) / (0.015 * tp_md + 1e-9)

    # --- F18: 改为短期回归趋势 (20d) ---
    ma_20 = grp.transform(lambda x: x.shift(1).rolling(20, min_periods=1).mean())
    df['factor_dist_ma20'] = df['ClosePrice'].shift(1) / (ma_20 + 1e-9)

    # --- F19: PSY 心理线 ---
    df['factor_psy'] = (df['ret_1d'] > 0).astype(int).groupby(df['Symbol']).transform(lambda x: x.shift(1).rolling(20, min_periods=1).mean())

    # --- F20: 阶段性新高距离 (改为 20d) ---
    high_20 = grp_high.transform(lambda x: x.shift(1).rolling(20, min_periods=1).max())
    df['factor_near_new_high'] = df['ClosePrice'].shift(1) / (high_20 + 1e-9)

    return df

def add_valuation_factors(df):
    print("正在计算：估值市值类因子 (优化版：最大窗口 20d)...")
    
    # 基础分组准备
    grp_pe = df.groupby('Symbol')['PE']
    
    # --- F21-F23: 基础价值因子倒数 (EP, BP, SP) ---
    df['factor_ep'] = 1 / (df['PE'] + 1e-9)
    df['factor_bp'] = 1 / (df['PB'] + 1e-9)
    df['factor_sp'] = 1 / (df['PS'] + 1e-9)
    
    # --- F24-F25: 市值特征 ---
    df['factor_size_ln'] = np.log(df['CirculatedMarketValue'] + 1e-9)
    df['factor_size_sq'] = df['factor_size_ln'] ** 2
    
    # --- F26: 相对行业 PE 偏离度 (截面计算，不产生冷启动缺失) ---
    if 'Indexcode' in df.columns:
        ind_pe_mean = df.groupby(['TradingDate', 'Indexcode'])['PE'].transform('mean')
        df['factor_rel_pe_ind'] = df['PE'] / (ind_pe_mean + 1e-9)
    else:
        df['factor_rel_pe_ind'] = 0  # 若无行业数据，填0以保住样本行

    # --- F27: PEG 因子 (增长率若缺失，则填补极小值) ---
    growth_rate = df['F081001B'].fillna(0) # 假设缺失增长率的公司增长为0
    df['factor_peg'] = df['PE'] / (growth_rate * 100 + 1e-9)

    # --- F28: 权益乘数 ---
    df['factor_float_ratio'] = df['CirculatedMarketValue'] / (df['MarketValue'] + 1e-9)

    # --- F29: 现金流市值比 (CFP) ---
    df['factor_cfp'] = df['F060101C'] / (df['PE'] + 1e-9)

    # --- F30: 动态估值偏离度 (改为 20d 窗口) ---
    # 使用 min_periods=1 彻底消除冷启动产生的 NaN
    pe_mean_20 = grp_pe.transform(lambda x: x.shift(1).rolling(20, min_periods=1).mean())
    pe_std_20 = grp_pe.transform(lambda x: x.shift(1).rolling(20, min_periods=1).std())
    df['factor_pe_zscore'] = (df['PE'].shift(1) - pe_mean_20) / (pe_std_20 + 1e-9)

    # --- F31: 市场份额权重 ---
    mkt_sum = df.groupby('TradingDate')['CirculatedMarketValue'].transform('sum')
    df['factor_mkt_weight'] = df['CirculatedMarketValue'] / (mkt_sum + 1e-9)

    # --- F32: PB-ROE 剩余收益偏离度 ---
    df['factor_pb_roe_gap'] = df['PB'] - (df['F050504C'].fillna(0) * 10) 

    # --- F33: 价值修复动量 (EP的变化) ---
    # 同样使用倒数，并确保 diff 后的第一个值不被丢弃
    df['factor_ep_diff_20'] = df['factor_ep'].groupby(df['Symbol']).transform(lambda x: x.diff(20).fillna(0))

    # --- F34: 营业收入市值比 ---
    df['factor_revenue_to_mkt'] = df['factor_sp']

    # --- F35: 自由现金流收益率 (FCF Yield) ---
    df['factor_fcf_yield'] = df['F060301C'].fillna(0) * df['factor_sp']

    return df

def add_quality_factors(df):
    print("正在计算：质量营运类因子 (优化版：最大窗口 20d)...")
    
    # 预处理：为防止财务数据中有极少数 NaN 导致计算崩溃，进行前向填充
    # 注意：这里只针对基础财务列进行局部填充，不影响整体逻辑
    f_cols = ['F050204C', 'F050504C', 'F053301C', 'F051401C', 'F051501C', 
              'F040304C', 'F040604C', 'F041405C', 'F041705C', 
              'F060101C', 'F060201C', 'F060401C', 'F080701B', 'F081001B', 'F081601B']
    
    # --- F36-F39: 核心盈利能力 ---
    df['factor_roa'] = df['F050204C']
    df['factor_roe'] = df['F050504C']
    df['factor_gpm'] = df['F053301C']
    df['factor_op_margin'] = df['F051401C']
    df['factor_npm'] = df['F051501C']
    
    # --- F40: 财务杠杆 ---
    df['factor_equity_multiplier'] = df['factor_roe'] / (df['factor_roa'] + 1e-9)
    
    # --- F41-F43: 营运周转效率 ---
    df['factor_ar_turnover'] = 365 / (df['F040304C'] + 1e-9)
    df['factor_inv_turnover'] = 365 / (df['F040604C'] + 1e-9)
    df['factor_fa_turnover'] = df['F041405C']
    df['factor_asset_turnover'] = df['F041705C']
    
    # --- F44: 盈余质量 ---
    df['factor_cash_to_profit'] = df['F060101C']
    df['factor_rev_cash_content'] = df['F060201C']
    df['factor_op_profit_cash'] = df['F060401C']
    
    # --- F45: 盈利稳定性 (缩减至 20d, 增加 min_periods) ---
    # 衡量近期盈利能力的波动，避开 250d 的冷启动损耗
    df['factor_roe_std'] = df.groupby('Symbol')['factor_roe'].transform(
        lambda x: x.rolling(20, min_periods=1).std().fillna(0)
    )
    
    # --- F46: Sloan 权责发生制近似 ---
    df['factor_accruals_alt'] = 1 - df['F060101C'].fillna(0)
    
    # --- F47: 营运资本变动 ---
    df['factor_operating_cash_efficiency'] = df['F060301C']

    # --- F48-F49: 成长性指标 ---
    df['factor_roe_growth_a'] = df['F080701B']
    df['factor_profit_growth_a'] = df['F081001B']
    df['factor_rev_growth_a'] = df['F081601B']

    # --- F50: 杜邦复合因子 ---
    df['factor_profit_efficiency'] = df['factor_roe'] * df['factor_asset_turnover']

    return df

def add_risk_factors(df):
    print("正在计算：风险波动类因子 (最大窗口 20d)...")
    
    # 基础分组准备
    grp_ret = df.groupby('Symbol')['ret_1d']
    grp_close = df.groupby('Symbol')['ClosePrice']
    grp_high = df.groupby('Symbol')['HighPrice']
    grp_low = df.groupby('Symbol')['LowPrice']

    # --- F51-F52: 历史波动率 (统一为 10d 和 20d) ---
    df['factor_vol_10d'] = grp_ret.transform(lambda x: x.shift(1).rolling(10, min_periods=1).std() * np.sqrt(250))
    df['factor_vol_20d'] = grp_ret.transform(lambda x: x.shift(1).rolling(20, min_periods=1).std() * np.sqrt(250))

    # --- F53-F54: 分布特征 (增加 min_periods) ---
    df['factor_skew_20d'] = grp_ret.transform(lambda x: x.shift(1).rolling(20, min_periods=5).skew())
    df['factor_kurt_20d'] = grp_ret.transform(lambda x: x.shift(1).rolling(20, min_periods=5).kurt())

    # --- F55: 20日最大回撤 ---
    def calc_max_drawdown(p):
        rolling_max = p.rolling(20, min_periods=1).max()
        drawdown = (p - rolling_max) / (rolling_max + 1e-9)
        return drawdown.rolling(20, min_periods=1).min()
    df['factor_mdd_20d'] = grp_close.transform(lambda x: calc_max_drawdown(x.shift(1)))

    # --- F56: 特质波动率 ---
    market_vol = df.groupby('TradingDate')['ret_1d'].transform('std')
    df['factor_idiosyncratic_vol'] = df['factor_vol_20d'] / (market_vol * np.sqrt(250) + 1e-9)

    # --- F57: 下行波动率 ---
    neg_ret = df['ret_1d'].clip(upper=0)
    df['factor_downside_vol'] = neg_ret.groupby(df['Symbol']).transform(
        lambda x: x.shift(1).rolling(20, min_periods=1).std() * np.sqrt(250)
    )

    # --- F58: 20日收益极差 ---
    h_max = grp_high.transform(lambda x: x.shift(1).rolling(20, min_periods=1).max())
    l_min = grp_low.transform(lambda x: x.shift(1).rolling(20, min_periods=1).min())
    df['factor_range_20d'] = (h_max - l_min) / (df['ClosePrice'].shift(1) + 1e-9)

    # --- F59: Beta (20日, 修复警告并增加 min_periods) ---
    mkt_ret = df.groupby('TradingDate')['ret_1d'].transform('mean')
    df['mkt_ret_temp'] = mkt_ret
    # 消除 FutureWarning
    df['factor_beta_20d'] = df.groupby('Symbol').apply(
        lambda x: x['ret_1d'].shift(1).rolling(20, min_periods=5).corr(x['mkt_ret_temp'].shift(1)),
        include_groups=False
    ).reset_index(level=0, drop=True)
    
    # 这里的截面计算确保不会因为除以0产生大量空值
    df['factor_beta_20d'] = df['factor_beta_20d'] * (df['factor_vol_20d'] / (market_vol.fillna(0.02) * np.sqrt(250) + 1e-9))
    df.drop(columns=['mkt_ret_temp'], inplace=True)

    # --- F60: 日内振幅均值 ---
    daily_amplitude = (df['HighPrice'] - df['LowPrice']) / (df['ClosePrice'].shift(1) + 1e-9)
    df['factor_avg_amplitude_20d'] = daily_amplitude.groupby(df['Symbol']).transform(
        lambda x: x.shift(1).rolling(20, min_periods=1).mean()
    )

    # --- F61: 波动率变动比率 (改为 10d / 20d) ---
    df['factor_vol_ratio'] = df['factor_vol_10d'] / (df['factor_vol_20d'] + 1e-9)

    # --- F62: 夏普比率 (20日) ---
    r_mean = grp_ret.transform(lambda x: x.shift(1).rolling(20, min_periods=1).mean())
    r_std = grp_ret.transform(lambda x: x.shift(1).rolling(20, min_periods=1).std())
    df['factor_sharpe_20d'] = r_mean / (r_std + 1e-9)

    # --- F63: 价格跳空因子 ---
    gaps = (df['OpenPrice'] - df['ClosePrice'].shift(1)).abs() / (df['ClosePrice'].shift(1) + 1e-9)
    df['factor_gap_std'] = gaps.groupby(df['Symbol']).transform(lambda x: x.shift(1).rolling(20, min_periods=1).std())

    # --- F64: 高低价差稳定性 ---
    df['factor_hi_lo_std'] = daily_amplitude.groupby(df['Symbol']).transform(
        lambda x: x.shift(1).rolling(20, min_periods=1).std()
    )

    # --- F65: 尾部风险 (VaR 5%) ---
    df['factor_var_5pct'] = grp_ret.transform(lambda x: x.shift(1).rolling(20, min_periods=5).quantile(0.05))

    return df

def add_sentiment_factors(df):
    print("正在计算：成交情绪类因子 (优化版：最大窗口 20d)...")
    
    # 基础分组准备
    grp_to = df.groupby('Symbol')['TurnoverRate1']
    grp_vol = df.groupby('Symbol')['Volume']
    grp_close = df.groupby('Symbol')['ClosePrice']
    
    # --- F66-F67: 换手率特征 (增加 min_periods) ---
    df['factor_avg_turnover_20d'] = grp_to.transform(lambda x: x.shift(1).rolling(20, min_periods=1).mean())
    df['factor_std_turnover_20d'] = grp_to.transform(lambda x: x.shift(1).rolling(20, min_periods=1).std())
    
    # --- F68: 成交量比率 ---
    avg_vol_20 = grp_vol.transform(lambda x: x.shift(1).rolling(20, min_periods=1).mean())
    df['factor_vol_ratio'] = df['Volume'].shift(1) / (avg_vol_20 + 1e-9)
    
    # --- F69: 量价相关性 (修复警告并增加 min_periods) ---
    df['factor_corr_pv_20d'] = df.groupby('Symbol').apply(
        lambda x: x['ClosePrice'].shift(1).rolling(20, min_periods=5).corr(x['Volume'].shift(1)),
        include_groups=False
    ).reset_index(level=0, drop=True)
    
    # --- F70: MFI (资金流量指标) ---
    tp = (df['HighPrice'] + df['LowPrice'] + df['ClosePrice']) / 3
    mf = tp * df['Volume']
    # 窗口缩短至 14d，并增加 min_periods
    pos_mf = mf.where(tp > tp.shift(1), 0).groupby(df['Symbol']).transform(lambda x: x.shift(1).rolling(14, min_periods=1).sum())
    neg_mf = mf.where(tp < tp.shift(1), 0).groupby(df['Symbol']).transform(lambda x: x.shift(1).rolling(14, min_periods=1).sum())
    df['factor_mfi'] = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-9)))

    # --- F71: 价升量行能量 (PV Energy) ---
    vol_chg = grp_vol.transform(lambda x: x.pct_change().fillna(0))
    df['factor_pv_energy'] = (df['ret_1d'].shift(1) * vol_chg.shift(1)).rolling(20, min_periods=1).mean()

    # --- F72: 换手率动量 (窗口控制在 5d 左右) ---
    df['factor_turnover_chg'] = df['TurnoverRate1'].shift(1) / (df['TurnoverRate1'].shift(6).fillna(method='ffill') + 1e-9)

    # --- F73: Amihud 非流动性因子 ---
    df['factor_illiquidity'] = df['ret_1d'].abs().shift(1) / (df['Amount'].shift(1) / 1e8 + 1e-9)

    # --- F74: BRAR 情绪指标 (窗口从 26 缩短至 20) ---
    br_up = (df['HighPrice'] - df['ClosePrice'].shift(1)).clip(lower=0).groupby(df['Symbol']).transform(lambda x: x.shift(1).rolling(20, min_periods=1).sum())
    br_dn = (df['ClosePrice'].shift(1) - df['LowPrice']).clip(lower=0).groupby(df['Symbol']).transform(lambda x: x.shift(1).rolling(20, min_periods=1).sum())
    df['factor_br'] = br_up / (br_dn + 1e-9)

    # --- F75: 均价偏离度 ---
    df['factor_avg_px_bias'] = df['AvgPrice'].shift(1) / (df['ClosePrice'].shift(1) + 1e-9) - 1

    # --- F76: 上涨成交占比 ---
    up_vol = df['Volume'].where(df['ret_1d'] > 0, 0).groupby(df['Symbol']).transform(lambda x: x.shift(1).rolling(20, min_periods=1).sum())
    total_vol_20 = grp_vol.transform(lambda x: x.shift(1).rolling(20, min_periods=1).sum())
    df['factor_up_vol_ratio'] = up_vol / (total_vol_20 + 1e-9)

    # --- F77: 日内收盘价位置 ---
    df['factor_daily_pos'] = (df['ClosePrice'].shift(1) - df['LowPrice'].shift(1)) / \
                             (df['HighPrice'].shift(1) - df['LowPrice'].shift(1) + 1e-9)

    # --- F78: PVT (修正逻辑：避免无限累加，改为 20 日窗口动态值) ---
    # 机器学习中，窗口化的特征通常比全历史累计值更稳定
    pvt_raw = (df['ret_1d'] * df['Volume']).groupby(df['Symbol']).transform(lambda x: x.shift(1).rolling(20, min_periods=1).sum())
    df['factor_pvt_20d'] = pvt_raw

    # --- F79: 换手率乖离率 (窗口从 60 缩短至 20) ---
    ma_to_20 = grp_to.transform(lambda x: x.shift(1).rolling(20, min_periods=1).mean())
    df['factor_turnover_bias'] = df['TurnoverRate1'].shift(1) / (ma_to_20 + 1e-9)

    # --- F80: 量价背离预警 ---
    # 逻辑：股价创 20 日新高但成交量低于 20 日均量
    high_20_prev = grp_close.transform(lambda x: x.shift(2).rolling(20, min_periods=1).max())
    price_new_high = (df['ClosePrice'].shift(1) > high_20_prev).astype(int)
    vol_low = (df['Volume'].shift(1) < avg_vol_20).astype(int)
    df['factor_pv_divergence'] = price_new_high * vol_low

    return df
# 本地数据契约

以下是基础母版兼容的输入格式，不是下载脚本。所有预测时点需要的数据都应由调用者提前按历史可用时间处理；本模板不会通过读取当前快照补全过去。

## 1. 原始月线

| 文件 | 必需字段 | 口径 |
|---|---|---|
| raw/ts_sw_industry_monthly.csv | ts_code, date, close, pe, pb | 固定行业池，代码为字符串，close为正数；由相邻自然月close重新计算小数收益 |
| raw/ts_csi300_monthly.csv | date, close | 市场基准月线；可以替换基准但需重新定义其经济含义 |
| raw/ts_macro_factors.csv | date, shibor_1y, shibor_1m, pmi_mfg, m1_yoy_pct, m2_yoy_pct, sf_inc_month | 分别为利率、PMI、货币增速与月新增社融；不得把月增量与累计值混用 |

日期推荐ISO `YYYY-MM-DD`，原行情的 `YYYYMMDD` 文本亦兼容。行业行情按行业和自然月规范化，保留月内最后一条实际观察；缺自然月不把多月收益当单月收益。训练要求至少74个月、至少10个行业，默认训练60个月＋验证12个月，并保留至少两个后续月份。

HMM用PMI变化、SHIBOR一年减一月、M1减M2、滚动12月社融同比，以及市场1/3/6月动量、3/6月波动、波动比率、6月均线偏离、12月区间位置。无公告时间的PMI、货币、社融特征滞后一个月。标准化只拟合各次HMM的历史60个月，不用全样本参数。

## 2. 季度基本面

`processed/prosperity_indicators_clean.pkl` 必须是可信的pandas DataFrame：

```text
l1_code, year, quarter,
totprofit, nptocostexpense, netprofitexcl, netprofitincl,
ebittointerest, opercash, grossprofitmargin,
con_op_pr_yoy, con_op_rt_yoy, con_np_cagh,
con_tp_cagh, con_op_pr_cagh, con_tp_yoy, operrev
```

同一行业、年、季度只允许一行。保留原接口：Q1映射4—6月，Q2映射7—9月，Q3映射10—12月，Q4映射下一年1—3月。这个映射没有逐家公司公告日，不能自动证明基本面在季度第一天已经公开。生产数据应通过自己的时点适配层解决公告滞后和修订，再明确记录口径。

每个行情行业月份都必须有对应映射；缺失时在训练前报错。原行业排除列表通过配置 `SW_EXCLUDE` 控制，不把行业文件名当分类锁定。

## 3. 技术与形态因子

`processed/price_volume_factors.pkl`：`ts_code, date` 加9个数值因子。原名为：

```text
mom_1m, mom_3m, mom_6m, vol_1m, vol_3m,
turnover_chg, vol_price_corr, max_drawdown_1m, high_low_pos
```

`processed/pattern_factors.pkl`：

```text
ts_code, date, pattern_median, bullish_ratio, signal_strength
```

总输入严格为12个数值因子；每个行业月份唯一。原兼容逻辑对缺少的形态值填0，但形态列必须存在，9个技术因子的整行覆盖不能缺失。形态算法本身不在母版中，使用者必须保证匹配库不包含决策之后的历史结果。合成示例的三列只是当前／过去合成收益的代理，不能视为原形态算法复现。

所有行业共有训练参数，额外附加行业one-hot。每个行业的MinMaxScaler只由训练期间拟合；验证／预测必须带对应训练期scaler，不允许遇到新行业临时拟合未来数据。

## 安全与扩展

- pickle反序列化能够执行代码，仅加载自己生成或明确可信的输入。
- 仓库不需要Tushare或其他接口密钥；下载程序应在外部数据层，凭证由环境变量管理，不能写入CSV、配置示例或Git。
- 自带校验检查格式、连续性和覆盖，不证明供应商数据真伪、分红复权或历史可见性。
- 动态股票池、基金公告时点、日频执行、成本和整手约束都属于后续适配层，不应凭更改文件名宣称支持。

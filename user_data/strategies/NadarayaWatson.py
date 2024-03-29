# --- Do not remove these libs ---
from datetime import datetime
from typing import Any, Optional
from freqtrade.strategy import IStrategy, informative, stoploss_from_absolute
from pandas import DataFrame
import talib.abstract as ta
import indicators as indicators
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
# --------------------------------

class NadarayaWatson(IStrategy):
    INTERFACE_VERSION: int = 3

    minimal_roi = { "0":  1 }

    # Optimal stoploss designed for the strategy
    stoploss = -0.1
    use_custom_stoploss = True

    # Optimal timeframe for the strategy
    timeframe = '1h'

    custom_info: dict = {}

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                        proposed_stake: float, min_stake: Optional[float], max_stake: float,
                        leverage: float, entry_tag: Optional[str], side: str,
                        **kwargs) -> float:
        if self.wallets is None:
            return proposed_stake
        return self.wallets.get_total_stake_amount() * .1

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['kr'] = indicators.kernel_regression(df['close'], loop_back=24)
        df['gr'] = indicators.gaussian_regression(df['close'], loop_back=8)
        df['ema'] = ta.EMA(df, timeperiod=50)
        df['atr'] = ta.ATR(df, timeperiod=14)
        df['gr_atr'] = indicators.gaussian_regression(df['atr'], loop_back=8)
        df['rsi'] = ta.RSI(df, timeperiod=14)
        df['rsi_gr'] = indicators.gaussian_regression(df['rsi'], loop_back=2)
        df['rsi_kr'] = indicators.kernel_regression(df['rsi'], loop_back=24)
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata['pair']
        df.loc[
            (
                (df['close'].pct_change(periods=12) < 0.2) &
                (df['close'].pct_change(periods=2) < 0.05) &
                (df['atr'] > df['gr_atr']) &
                (df['rsi_gr'] > df['rsi_kr']) &
                (df['gr'].shift(1) < df['gr']) &
                (
                    (
                        (df['close'] > df['ema']) &
                        qtpylib.crossed_above(df['gr'], df['kr'])
                    ) | (
                        qtpylib.crossed_above(df['close'], df['ema']) &
                        (df['kr'] < df['gr'])
                    )
                )
            ),
            'enter_long'
        ] = 1
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            (qtpylib.crossed_below(df['kr'], df['kr'].shift(1)) & False),
            'exit_long'
        ] = 1
        return df

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        profit = trade.calc_profit_ratio(rate)
        if (exit_reason == 'force_exit'):
            return False
        if pair in self.custom_info:
            del self.custom_info[pair]
        return True

    def trade_candle(self, pair, trade, current_rate):
        if pair not in self.custom_info:
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            last_candle = df.iloc[-1].squeeze()
            # Swing Low Stoploss
            low = max(trade.open_rate * .9, df['low'].rolling(14).min().iloc[-1])
            diff = (trade.open_rate - low) * 1.2

            self.custom_info[pair] = {
                'last_candle': last_candle,
                'sl': trade.open_rate - diff,
                'pt': trade.open_rate + (diff * 1.5),
                'diff': diff,
                'open_rate': trade.open_rate,
            }

        last_candle = self.custom_info[pair]
        return last_candle

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> float:
        candle = self.trade_candle(pair, trade, current_rate)
        
        def get_percent(up, dn):
            return (dn - up) / up # get -ve value

        if (candle['pt'] < current_rate):
            candle['open_rate'] = current_rate
            gap = (candle['diff'] * 0.5)
            candle['sl'] = current_rate - gap
            candle['pt'] = current_rate + gap

        break_even = candle['open_rate'] + candle['diff']
        if (break_even < current_rate):
            candle['open_rate'] = current_rate 
            candle['sl'] = candle['open_rate'] - (candle['diff'] * 0.9)

        return get_percent(current_rate, candle['sl'])

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 8
            },
        ]
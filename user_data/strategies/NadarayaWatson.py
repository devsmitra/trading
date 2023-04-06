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
    sl = 4.1
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
        return self.wallets.get_total_stake_amount() * .06

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        data = indicators.Nadaraya_Watson(df, loop_back= 14)
        df['yhat'] = data['yhat']
        df['ema'] = indicators.smma(df, timeperiod=32)

        df['atr'] = ta.ATR(df, timeperiod=14)
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata['pair']
        df.loc[
            (
                (df['close'] > df['ema']) &
                qtpylib.crossed_above(df['yhat'], df['yhat'].shift(1))
            ),
            'enter_long'
        ] = 1
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            (qtpylib.crossed_below(df['yhat'], df['yhat'].shift(1)) & False),
            'exit_long'
        ] = 1
        return df

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        profit = trade.calc_profit_ratio(rate)
        if (((exit_reason == 'force_exit')) and (profit < 0.01)):
            return False
        if pair in self.custom_info:
            del self.custom_info[pair]
        return True

    def trade_candle(self, pair, trade, current_rate):
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()
        if pair not in self.custom_info:
            self.custom_info[pair] = {
                'last_candle': last_candle,
            }
        last_candle = self.custom_info[pair]['last_candle']
        return last_candle

    def custom_exit(self, pair: str, trade: Trade, current_time: 'datetime',
                    current_rate: float, current_profit: float, **kwargs):

        last_candle = self.trade_candle(pair, trade, current_rate)
        mul = self.sl
        pt = trade.open_rate + (last_candle['atr'] * mul * 1.5)
        if (pt < current_rate):
            return 'Profit Booked'

        sl = trade.open_rate - (last_candle['atr'] * mul)
        if (sl > current_rate):
            return 'Stop Loss Hit'

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> float:
        candle = self.trade_candle(pair, trade, current_rate)

        def get_stoploss(atr):
            return stoploss_from_absolute(
                current_rate - (candle['atr'] * atr), 
                current_rate, 
                is_short=trade.is_short
            ) * -1

        pt = trade.open_rate + (candle['atr'] * self.sl)
        if (pt < current_rate):
            return get_stoploss(self.sl/2)

        return 1
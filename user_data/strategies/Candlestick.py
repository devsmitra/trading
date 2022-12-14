# --- Do not remove these libs ---
from datetime import datetime
from typing import Any, Optional
from freqtrade.strategy import IStrategy, stoploss_from_absolute
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade

from support import identify_df_trends

# --------------------------------


class Candlestick(IStrategy):
    cache: Any = {}

    INTERFACE_VERSION: int = 3
    process_only_new_candles: bool = False
    # Optimal timeframe for the strategy
    timeframe = '15m'

    minimal_roi = {
        "0": 1
    }

    # Optimal stoploss designed for the strategy
    stoploss = -0.1
    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        sl = 4.1
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        candle = dataframe.iloc[-1].squeeze()   
        def get_stoploss(atr):
            return stoploss_from_absolute(current_rate - (candle['atr'] * atr), current_rate, is_short=trade.is_short)

        if current_rate > (trade.open_rate + (candle['atr'] * 1.5 * sl)):
            return -.0001
        return get_stoploss(sl) * -1

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
        if self.wallets is None:
            return proposed_stake
        return self.wallets.get_total_stake_amount() * .06

    def get_trend(self, dataframe: DataFrame, metadata: dict):
        pair = metadata['pair']
        prev = self.cache.get(pair,  {'date': dataframe.iloc[-2]['date'], 'Trend': 0})
        date = dataframe.iloc[-1]['date']
        if (date != prev['date']):
            df = identify_df_trends(dataframe, 'close', window_size=3)
            self.cache[pair] = {'date': date, 'Trend': df['Trend']}
        else:
            dataframe['Trend'] = prev['Trend']

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self.get_trend(dataframe, metadata)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['atr'] = ta.ATR(dataframe)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        crossed = False
        for i in range(0, 2):
            crossed = crossed | (
                qtpylib.crossed_above(dataframe.shift(i)['Trend'], 0) &
                (dataframe.shift(i)['adx'] > 20)
            )

        dataframe.loc[
            crossed,
            'enter_long'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['adx'], 25) & (dataframe['Trend'] == -1)) |
                (qtpylib.crossed_below(dataframe['Trend'], 0) & (dataframe['adx'] < 25)) |
                (qtpylib.crossed_below(dataframe.shift()['Trend'], 0) & (dataframe['adx'] < 25))
            ),
            'exit_long'] = 1
        return dataframe

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        profit = trade.calc_profit_ratio(rate)
        if (((exit_reason == 'force_exit') | (exit_reason == 'exit_signal')) and (profit < 0)):
            return False
        return True

# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy, stoploss_from_absolute
from pandas import DataFrame
from freqtrade.persistence import Trade
from datetime import datetime
from typing import Optional

# --------------------------------
import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
import indicators as indicators


class MACD(IStrategy):
    INTERFACE_VERSION: int = 3

    minimal_roi = {
        "0":  1
    }

    # Optimal stoploss designed for the strategy
    stoploss = -0.1
    use_custom_stoploss = True

    # Optimal timeframe for the strategy
    timeframe = '1h'

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
        if self.wallets is None:
            return proposed_stake
        return self.wallets.get_total_stake_amount() * .06

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        heikinashi = qtpylib.heikinashi(df)
        df['ha_close'] = heikinashi['close']
        df['ha_open'] = heikinashi['open']
        df['ha_high'] = heikinashi['high']
        df['ha_low'] = heikinashi['low']

        df['atr'] = ta.ATR(df, timeperiod=14)
        df['zlsma'] = indicators.zlsma(df, period=50, offset=0, column='ha_close')
        df['chandelier_exit'] = indicators.chandelier_exit(
            df, timeperiod=22, multiplier=1.85, column='ha_close'
        )
        df['cmf'] = indicators.cmf(df)
        st = indicators.supertrend(df)
        df['trend'] = st['ST']

        df['rsi'] = ta.RSI(df, timeperiod=25)
        df['rsi_ma'] = ta.SMA(df['rsi'], timeperiod=150)
        # vol = indicators.volatility_osc(df)
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        enter_long = (
            ((df['ha_close'].diff() / df['ha_close']) < .2) &
            (df['ha_close'] > df['zlsma']) &
            (df['cmf'] > 0) &
            (df['rsi'] > df['rsi_ma']) &
            (
                ((df['trend'] < df['ha_close']) & qtpylib.crossed_above(df['chandelier_exit'], 0)) |
                (qtpylib.crossed_above(df['ha_close'], df['trend']) & (df['chandelier_exit'] > 0))
            )
        )
        df.loc[enter_long, 'enter_long'] = 1
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            (
                (df['trend'] > df['ha_close']) &
                (df['chandelier_exit'] < 1) &
                (df['ha_close'] < df['zlsma'])
            ),
            'exit_long'
        ] = 1
        return df

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        # df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        # candle = df.iloc[-1].squeeze()
        # atr = candle['atr']

        # def get_stoploss(multiplier):
        #     return stoploss_from_absolute(current_rate - (
        #        atr * multiplier), current_rate, is_short=trade.is_short
        #     ) * -1

        if (current_profit > .075):
            return -.02

        if pair not in self.custom_info:
            return 1

        details = self.custom_info[pair]
        swing_low = details['info']['trend']
        return stoploss_from_absolute(trade.open_rate - swing_low, trade.open_rate,
                                      is_short=trade.is_short) * -1

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 6
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 6,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "only_per_pair": True
            },
        ]

    custom_info: dict = {}

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        profit = trade.calc_profit_ratio(rate)
        if (((exit_reason == 'force_exit') | (exit_reason == 'exit_signal')) and (profit < 0.005)):
            return False
        if pair in self.custom_info:
            del self.custom_info[pair]
        return True

    def custom_exit(self, pair: str, trade: Trade, current_time: 'datetime',
                    current_rate: float, current_profit: float, **kwargs):
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()

        if pair not in self.custom_info:
            self.custom_info[pair] = {
                'info': last_candle,
                'hit': False,
            }

        details = self.custom_info[pair]
        swing_low = details['info']['trend']
        hit = details['hit']

        diff = trade.open_rate - swing_low
        if ((((1.5 * diff) + trade.open_rate) < current_rate) | hit) & (current_profit > .005):
            self.custom_info[pair]['hit'] = True
            if (last_candle['chandelier_exit'] < 1):
                del self.custom_info[pair]
                return 'Profit Booked'
        elif (current_rate < swing_low):
            del self.custom_info[pair]
            return 'Stop Loss Hit'

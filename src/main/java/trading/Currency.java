package trading;

import com.binance.api.client.BinanceApiWebSocketClient;
import com.binance.api.client.domain.market.Candlestick;
import com.binance.api.client.domain.market.CandlestickInterval;
import com.binance.api.client.exception.BinanceApiException;
import data.PriceBean;
import data.PriceReader;
import indicators.*;
import system.ConfigSetup;
import system.Formatter;
import system.Mode;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.time.Duration;
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.StringJoiner;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;

public class Currency {
    private static final String FIAT = "USDT";

    private final String pair;
    private Trade activeTrade;
    private long candleTime;
    private final List<Indicator> indicators = new ArrayList<>();
    private final AtomicBoolean currentlyCalculating = new AtomicBoolean(false);
    private double backtestingResult;

    private double currentPrice;
    private long currentTime;

    //Backtesting data
    private final StringBuilder log = new StringBuilder();
    private PriceBean firstBean;


    //Used for SIMULATION and LIVE
    public Currency(String coin) throws BinanceApiException {
        this.pair = coin + FIAT;

        //Every currency needs to contain and update our indicators
        List<Candlestick> history = CurrentAPI.get().getCandlestickBars(pair, CandlestickInterval.FIVE_MINUTES);
        List<Double> closingPrices = history.stream().map(candle -> Double.parseDouble(candle.getClose())).collect(Collectors.toList());
        indicators.add(new RSI(closingPrices, 14));
        indicators.add(new MACD(closingPrices, 12, 26, 9));
        indicators.add(new DBB(closingPrices, 20));

        //We set the initial values to check against in onMessage based on the latest candle in history
        currentTime = System.currentTimeMillis();
        candleTime = history.get(history.size() - 1).getCloseTime();
        currentPrice = Double.parseDouble(history.get(history.size() - 1).getClose());

        BinanceApiWebSocketClient client = CurrentAPI.getFactory().newWebSocketClient();
        //We add a websocket listener that automatically updates our values and triggers our strategy or trade logic as needed
        client.onAggTradeEvent(pair.toLowerCase(), response -> {
            //Every message and the resulting indicator and strategy calculations is handled concurrently
            //System.out.println(Thread.currentThread().getId());
            double newPrice = Double.parseDouble(response.getPrice());
            long newTime = response.getEventTime();

            //We want to toss messages that provide no new information
            if (currentPrice == newPrice && newTime <= candleTime) {
                return;
            }

            if (newTime > candleTime) {
                accept(new PriceBean(candleTime, currentPrice, true));
                candleTime += 300000L;
            }

            accept(new PriceBean(newTime, newPrice));
        });
        System.out.println("---SETUP DONE FOR " + this);
    }

    //Used for BACKTESTING
    public Currency(String pair, String filePath) throws BinanceApiException {
        this.pair = pair;
        try (PriceReader reader = new PriceReader(filePath)) {
            PriceBean bean = reader.readPrice();

            firstBean = bean;
            List<Double> closingPrices = new ArrayList<>();
            while (bean.isClosing()) {
                closingPrices.add(bean.getPrice());
                bean = reader.readPrice();
            }
            //TODO: Fix slight mismatch between MACD backtesting and server values.
            indicators.add(new RSI(closingPrices, 14));
            indicators.add(new MACD(closingPrices, 12, 26, 9));
            indicators.add(new DBB(closingPrices, 20));
            while (bean != null) {
                accept(bean);
                bean = reader.readPrice();
            }
            for (Trade trade : BuySell.getAccount().getActiveTrades()) {
                trade.setExplanation(trade.getExplanation() + "Manually closed");
                BuySell.close(trade);
            }
            backtestingResult = BuySell.getAccount().getProfit() - ((currentPrice - firstBean.getPrice()) / firstBean.getPrice());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    //Used for ML
    public Currency(String pair, String filePath, String outputPath) throws BinanceApiException {
        this.pair = pair;
        try (PriceReader reader = new PriceReader(filePath)) {
            PriceBean bean = reader.readPrice();

            firstBean = bean;
            List<Double> closingPrices = new ArrayList<>();
            while (bean.isClosing()) {
                closingPrices.add(bean.getPrice());
                bean = reader.readPrice();
            }

            indicators.add(new RSI(closingPrices, 14));
            indicators.add(new MACD(closingPrices, 12, 26, 9));
            indicators.add(new DBB(closingPrices, 20));
            indicators.add(new EMA(closingPrices, 9, false));
            indicators.add(new EMA(closingPrices, 21, false));
            indicators.add(new EMA(closingPrices, 50, false));

            if (outputPath != null) {
                long count = 0;
                long totalCount = 1;
                try (PriceReader tempReader = new PriceReader(filePath)) {
                    PriceBean tempBean = tempReader.readPrice();
                    while (tempBean != null) {
                        totalCount++;
                        tempBean = tempReader.readPrice();
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
                System.out.println("Data file contains " + Formatter.formatLarge(totalCount) + " entries");
                System.out.println("Features: " + getCsvHeader().replace(",", ", "));
                System.out.println("Calculating indicator values for each entry...");
                try (PrintWriter writer = new PrintWriter(outputPath)) {
                    writer.write(getCsvHeader());
                    int printCount = 0;
                    while (bean != null) {
                        count++;
                        printCount++;
                        if (printCount >= 100000) {
                            System.out.print('\r' + Formatter.formatPercent((double) count / totalCount));
                            printCount = 0;
                        }
                        acceptMl(bean, true);
                        writer.write(getCsvLine(bean));
                        bean = reader.readPrice();
                    }
                }
                System.out.print('\r' + Formatter.formatPercent(1));
                System.out.println("\nDone");
            } else {
                while (bean != null) {
                    acceptMl(bean, false);
                    bean = reader.readPrice();
                }
            }

            for (Trade trade : BuySell.getAccount().getActiveTrades()) {
                trade.setExplanation(trade.getExplanation() + "Manually closed");
                BuySell.close(trade);
            }
            backtestingResult = BuySell.getAccount().getProfit() - ((currentPrice - firstBean.getPrice()) / firstBean.getPrice());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void accept(PriceBean bean) {
        //Make sure we dont get concurrency issues
        if (currentlyCalculating.get()) {
            System.out.println("------------WARNING, NEW THREAD STARTED ON " + pair + " MESSAGE DURING UNFINISHED PREVIOUS MESSAGE CALCULATIONS");
        }
        currentlyCalculating.set(true);

        currentPrice = bean.getPrice();
        currentTime = bean.getTimestamp();

        if (bean.isClosing()) {
            indicators.forEach(indicator -> indicator.update(bean.getPrice()));
            if (Mode.get().equals(Mode.BACKTESTING)) {
                appendLogLine(system.Formatter.formatDate(currentTime) + "  " + toString());
            }
        }


        //We can disable the strategy and trading logic to only check indicator and price accuracy
        int confluence = check();
        if (hasActiveTrade()) { //We only allow one active trade per currency, this means we only need to do one of the following:
            activeTrade.update(currentPrice, confluence);//Update the active trade stop-loss and high values
        } else {
            if (confluence >= 2) {
                StringJoiner joiner = new StringJoiner("", "Trade opened due to: ", "");
                for (Indicator indicator : indicators) {
                    String explanation = indicator.getExplanation();
                    joiner.add(explanation.equals("") ? "" : explanation + "\t");
                }
                BuySell.open(Currency.this, joiner.toString(), bean.getTimestamp());
            }
        }

        currentlyCalculating.set(false);
    }

    private void acceptMl(PriceBean bean, boolean onlyUpdate) {
        currentPrice = bean.getPrice();
        currentTime = bean.getTimestamp();

        if (bean.isClosing()) {
            indicators.forEach(indicator -> indicator.update(bean.getPrice()));
        }

        if (!onlyUpdate) {
            //TODO: Use tensorflow model here to check for signals\
        }
    }

    public double getBacktestingResult() {
        return backtestingResult;
    }

    public int check() {
        return indicators.stream().mapToInt(indicator -> indicator.check(currentPrice)).sum();
    }

    public String getPair() {
        return pair;
    }

    public double getPrice() {
        return currentPrice;
    }

    public long getCurrentTime() {
        return currentTime;
    }

    public boolean hasActiveTrade() {
        return activeTrade != null;
    }

    public void setActiveTrade(Trade activeTrade) {
        this.activeTrade = activeTrade;
    }

    public void appendLogLine(String s) {
        log.append(s).append("\n");
    }

    public void log(String path) {
        List<Trade> tradeHistory = new ArrayList<>(BuySell.getAccount().getTradeHistory());
        try (FileWriter writer = new FileWriter(path)) {
            writer.write("Test ended " + system.Formatter.formatDate(LocalDateTime.now()) + " \n");
            writer.write("\n\nCONFIG:\n");
            writer.write(ConfigSetup.getSetup());
            writer.write("\n\nMarket performance: " + system.Formatter.formatPercent((currentPrice - firstBean.getPrice()) / firstBean.getPrice()));
            if (!tradeHistory.isEmpty()) {
                tradeHistory.sort(Comparator.comparingDouble(Trade::getProfit));
                double maxLoss = tradeHistory.get(0).getProfit();
                double maxGain = tradeHistory.get(tradeHistory.size() - 1).getProfit();
                int lossTrades = 0;
                double lossSum = 0;
                int gainTrades = 0;
                double gainSum = 0;
                long tradeDurs = 0;
                for (Trade trade : tradeHistory) {
                    double profit = trade.getProfit();
                    if (profit < 0) {
                        lossTrades += 1;
                        lossSum += profit;
                    } else if (profit > 0) {
                        gainTrades += 1;
                        gainSum += profit;
                    }
                    tradeDurs += trade.getDuration();
                }

                double tradePerWeek = 604800000.0 / (((double) currentTime - firstBean.getTimestamp()) / tradeHistory.size());

                writer.write("\nBot performance: " + system.Formatter.formatPercent(BuySell.getAccount().getProfit()) + "\n\n");
                writer.write(BuySell.getAccount().getTradeHistory().size() + " closed trades"
                        + " (" + system.Formatter.formatDecimal(tradePerWeek) + " trades per week) with an average holding length of "
                        + system.Formatter.formatDuration(Duration.of(tradeDurs / tradeHistory.size(), ChronoUnit.MILLIS)) + " hours");
                if (lossTrades != 0) {
                    writer.write("\nLoss trades:\n");
                    writer.write(lossTrades + " trades, " + system.Formatter.formatPercent(lossSum / (double) lossTrades) + " average, " + system.Formatter.formatPercent(maxLoss) + " max");
                }
                if (gainTrades != 0) {
                    writer.write("\nProfitable trades:\n");
                    writer.write(gainTrades + " trades, " + system.Formatter.formatPercent(gainSum / (double) gainTrades) + " average, " + system.Formatter.formatPercent(maxGain) + " max");
                }
                writer.write("\n\nClosed trades (least to most profitable):\n");
                for (Trade trade : tradeHistory) {
                    writer.write(trade.toString() + "\n");
                }
            } else {
                writer.write("\n(Not trades made)\n");
                System.out.println("---No trades made in the time period!");
            }
            writer.write("\n\nFULL LOG:\n\n");
            writer.write(log.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("---Log file generated at " + path);
    }

    private String getCsvLine(PriceBean bean) {
        StringBuilder s = new StringBuilder();
        s.append(bean.getTimestamp());
        s.append(",");
        s.append(bean.getPrice());
        s.append(",");
        s.append(bean.isClosing() ? 1 : 0);
        s.append(",");
        for (int i = 0; i < indicators.size(); i++) {
            Indicator indicator = indicators.get(i);
            if (indicator.getClass() == DBB.class) {
                s.append(((DBB) indicator).getStdevRelative(bean.getPrice()));
                s.append(',');
                s.append(indicator.getTemp(bean.getPrice()));
            } else if (indicator.getClass() == EMA.class) {
                s.append(((EMA) indicator).getTempRelative(bean.getPrice()));
            } else if (indicator.getClass() == MACD.class) {
                s.append(((MACD) indicator).getTempRelative(bean.getPrice()));
            } else {
                s.append(indicator.getTemp(bean.getPrice()));
            }
            if (i != indicators.size() - 1) s.append(',');
        }
        s.append('\n');
        return s.toString();
    }

    private String getCsvHeader() {
        StringBuilder s = new StringBuilder();
        s.append("time,price,close,");
        for (int i = 0; i < indicators.size(); i++) {
            Indicator indicator = indicators.get(i);
            if (indicator.getClass() == DBB.class) {
                s.append("RSD");
                s.append(',');
                s.append("DBB");
            } else if (indicator.getClass() == EMA.class) {
                s.append("REMA_");
                s.append(((EMA) indicator).getPeriod());
            } else if (indicator.getClass() == MACD.class) {
                s.append("RMACD");
            } else {
                s.append(indicator.getClass().getSimpleName());
            }
            if (i != indicators.size() - 1) s.append(',');
        }
        s.append('\n');
        return s.toString();
    }

    @Override
    public String toString() {
        StringBuilder s = new StringBuilder(pair + " price: " + currentPrice);
        if (currentTime == candleTime)
            indicators.forEach(indicator -> s.append(", ").append(indicator.getClass().getSimpleName()).append(": ").append(system.Formatter.formatDecimal(indicator.get())));
        else
            indicators.forEach(indicator -> {
                s.append(", ").append(indicator.getClass().getSimpleName()).append(": ");
                if (indicator.getClass() == DBB.class) {
                    s.append(Formatter.formatDecimal(((DBB) indicator).getStdevRelative(currentPrice)));
                } else if (indicator.getClass() == EMA.class) {
                    s.append(Formatter.formatDecimal(((EMA) indicator).getTempRelative(currentPrice)));
                } else if (indicator.getClass() == MACD.class) {
                    s.append(Formatter.formatDecimal(((MACD) indicator).getTempRelative(currentPrice)));
                } else {
                    s.append(Formatter.formatDecimal(indicator.getTemp(currentPrice)));
                }
            });
        s.append(", hasActive: ").append(hasActiveTrade()).append(")");
        return s.toString();
    }

    @Override
    public int hashCode() {
        return pair.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null) return false;
        if (obj.getClass() != Currency.class) return false;
        return pair.equals(((Currency) obj).pair);
    }
}

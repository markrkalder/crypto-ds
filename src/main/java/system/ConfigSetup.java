package system;

import com.binance.api.client.domain.general.RateLimit;
import com.binance.api.client.domain.general.RateLimitType;
import indicators.MACD;
import indicators.RSI;
import modes.Backtesting;
import modes.Live;
import modes.Simulation;
import trading.BuySell;
import trading.CurrentAPI;
import trading.Trade;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.TimeZone;
import java.util.stream.Collectors;

//TODO: Remove boilerplate from ConfigSetup
//TODO: Create FIAT config option and replace "USDT" in code with it
public class ConfigSetup {
    private static double moneyPerTrade;
    private static long minutesForCollection;
    private static double startingValue;
    private static String[] currencies;
    private static double MACDChange;
    private static int RSIPosMax;
    private static int RSIPosMin;
    private static int RSINegMax;
    private static int RSINegMin;
    private static double trailingSL;
    private static double takeP;
    private static final int requestLimit = CurrentAPI.get().getExchangeInfo().getRateLimits().stream()
            .filter(rateLimit -> rateLimit.getRateLimitType().equals(RateLimitType.REQUEST_WEIGHT))
            .findFirst().map(RateLimit::getLimit).orElse(1200);

    private static String setup;

    private ConfigSetup() {
        throw new IllegalStateException("Utility class");
    }

    public static String getSetup() {
        return setup;
    }

    public static void init() {
        //TODO: Move this somewhere else
        Formatter.getSimpleFormatter().setTimeZone(TimeZone.getDefault());
        int items = 0;
        File file = new File("config.txt");
        try (FileReader reader = new FileReader(file);
             BufferedReader br = new BufferedReader(reader)) {
            String line;
            while ((line = br.readLine()) != null) {

                String[] arr = line.strip().split(":");
                items++;
                switch (arr[0]) {
                    case "MACD change indicator":
                        MACDChange = Double.parseDouble(arr[1]);
                        break;
                    case "RSI positive side minimum":
                        RSIPosMin = Integer.parseInt(arr[1]);
                        break;
                    case "RSI positive side maximum":
                        RSIPosMax = Integer.parseInt(arr[1]);
                        break;
                    case "RSI negative side minimum":
                        RSINegMin = Integer.parseInt(arr[1]);
                        break;
                    case "RSI negative side maximum":
                        RSINegMax = Integer.parseInt(arr[1]);
                        break;
                    case "Collection mode chunk size(minutes)":
                        minutesForCollection = Long.parseLong(arr[1]);
                        break;
                    case "Simulation mode starting value":
                        startingValue = Integer.parseInt(arr[1]);
                        break;
                    case "Simulation mode currencies":
                        currencies = arr[1].split(", ");
                        break;
                    case "Percentage of money per trade":
                        moneyPerTrade = Double.parseDouble(arr[1]);
                        break;
                    case "Trailing SL":
                        trailingSL = Double.parseDouble(arr[1]);
                        break;
                    case "Take profit":
                        takeP = Double.parseDouble(arr[1]);
                        break;
                    default:
                        break;
                }
            }
            if (items < 11) { //12 is the number of configuration elements in the file.
                throw new ConfigException("Config file has some missing elements.");

            }

        } catch (IOException e) {
            e.printStackTrace();
        } catch (ConfigException e) {
            e.printStackTrace();
            System.exit(0);
        }

        //LIVE
        Live.setCurrencyArr(getCurrencies());

        //SIMULATION
        Simulation.setCurrencyArr(getCurrencies());
        Simulation.setStartingValue(getStartingValue()); //How much money does the simulated acc start with.
        //The currencies that the simulation MODE will trade with.

        //TRADING
        BuySell.setMoneyPerTrade(getMoneyPerTrade()); //How many percentages of the money you have currently
        Trade.setTakeProfit(takeP);
        Trade.setTrailingSl(trailingSL);
        //will the program put into one trade.

        //BACKTESTING
        Backtesting.setStartingValue(getStartingValue());

        //INDICATORS

        //MACD
        MACD.setChange(getMACDChange()); //How much change does the program need in order to give a positive signal from MACD

        //RSI
        RSI.setPositiveMin(getRSIPosMin()); //When RSI reaches this value, it returns 2 as a signal.
        RSI.setPositivseMax(getRSIPosMax()); //When RSI reaches this value, it returns 1 as a signal.
        RSI.setNegativeMin(getRSINegMin()); //When RSI reaches this value, it returns -1 as a signal.
        RSI.setNegativeMax(getRSINegMax()); //When RSI reaches this value it returns -2 as a signal.

        setup = getConfig();
    }

    public static int getRequestLimit() {
        return requestLimit;
    }

    public static double getMoneyPerTrade() {
        return moneyPerTrade;
    }

    public static double getStartingValue() {
        return startingValue;
    }

    public static String[] getCurrencies() {
        return currencies;
    }

    public static double getMACDChange() {
        return MACDChange;
    }

    public static int getRSIPosMax() {
        return RSIPosMax;
    }

    public static int getRSIPosMin() {
        return RSIPosMin;
    }

    public static int getRSINegMax() {
        return RSINegMax;
    }

    public static int getRSINegMin() {
        return RSINegMin;
    }

    public static String getConfig() {
        return "MACD change indicator:" + MACDChange + "\n" +
                "RSI positive side minimum:" + RSIPosMin + "\n" +
                "RSI positive side maximum:" + RSIPosMax + "\n" +
                "RSI negative side minimum:" + RSINegMin + "\n" +
                "RSI negative side maximum:" + RSINegMax + "\n" +
                "Collection mode chunk size(minutes):" + minutesForCollection + "\n" +
                "Simulation mode starting value:" + startingValue + "\n" +
                "Percentage of money per trade:" + moneyPerTrade + "\n" +
                "Trailing SL:" + trailingSL + "\n" +
                "Take profit:" + takeP + "\n" +
                "Simulation mode currencies:" + Arrays.stream(currencies).map(currency -> currency + " ").collect(Collectors.joining()) + "\n";
    }
}

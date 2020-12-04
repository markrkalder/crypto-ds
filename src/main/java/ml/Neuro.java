package ml;

import system.ConfigSetup;
import system.Formatter;
import system.Mode;
import trading.BuySell;
import trading.Currency;
import trading.LocalAccount;

import java.io.File;

public class Neuro {
    public static void main(String[] args) {
        Mode.set(Mode.BACKTESTING);
        ConfigSetup.init();
        BuySell.setAccount(new LocalAccount("Investor Toomas", 1000));

        String path = "backtesting/BTCUSDT_2019.01.01-2020.01.01.dat";
        Currency currency = new Currency(new File(path).getName().split("_")[0], path);
        double result = currency.getBacktestingResult();
        System.out.println(Formatter.formatPercent(result));
        currency.log("log.txt");
    }
}

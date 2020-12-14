package ml;

import system.ConfigSetup;
import system.Mode;
import trading.BuySell;
import trading.Currency;
import trading.LocalAccount;

import java.io.File;

public class DataCalculator {
    public static void main(String[] args) {
        Mode.set(Mode.BACKTESTING);
        ConfigSetup.init();
        BuySell.setAccount(new LocalAccount("Investor Toomas", 1000));


        String path = "backtesting/BTCUSDT_2019.01.01-2020.01.01.dat";
        new Currency(new File(path).getName().split("_")[0], path, path.replace(".dat", "_ML.csv"));
        System.exit(0);
    }
}

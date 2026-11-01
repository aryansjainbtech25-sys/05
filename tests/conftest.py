import sys, types
if "yfinance" not in sys.modules:
    sys.modules["yfinance"] = types.ModuleType("yfinance")
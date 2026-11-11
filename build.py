"""Assemble l'application autonome : insère la librairie lightweight-charts (Apache-2.0, TradingView) dans le HTML."""
import pathlib, sys
here = pathlib.Path(__file__).parent
lib = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
tpl = (here / "app_template.html").read_text(encoding="utf-8")
banner = "/* lightweight-charts 4.2.3 — Copyright TradingView, Inc. — Apache License 2.0 — https://github.com/tradingview/lightweight-charts */\n"
(here.parent / "CAC40_Swing.html").write_text(tpl.replace("/*__LWC__*/", banner + lib.replace("</script", "<\\/script")), encoding="utf-8")
print("ok")

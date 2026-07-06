# The Waterfall — anatomy of a CLO

Interactive CLO cash-flow simulator built for a structuring-desk rotation capstone:
*introducing CLOs to finance professionals*, with the analytics done from scratch.

## Live demo

Enable GitHub Pages (Settings → Pages → deploy from `main` / `docs/`) and the
simulator is served at `https://<user>.github.io/clo-presentation/`.

`docs/index.html` is fully self-contained — no CDN, no external scripts —
so it also runs offline: download the single file and open it in any browser.

## What's inside

| Path | Contents |
|---|---|
| `docs/index.html` | Offline simulator — hand-rolled SVG charts, zero dependencies |
| `docs/simulator-cdn.html` | Same simulator using Plotly via CDN (richer chart interactions) |
| `src/clo_engine.js` | Standalone waterfall engine + node test harness compatibility |

## The model

$500M synthetic broadly-syndicated loan pool · 6 tranches (AAA 63% → equity 10%)
· 8y maturity, 5y reinvestment · quarterly waterfall with senior/junior OC tests
that divert interest to AAA paydown on breach.

Validation: cash conservation holds every quarter (sources = uses), equity IRR is
monotone decreasing in defaults, losses cascade strictly bottom-up. Assumptions
and simplifications are documented in the deep-dive panel of the simulator.

## Run the engine tests

```bash
node -e "const {runDeal}=require('./src/clo_engine.js'); console.log(runDeal({cdr:0.03,recovery:0.6,cpr:0.15}).eqIRR)"
```

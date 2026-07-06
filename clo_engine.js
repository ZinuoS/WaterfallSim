// CLO Cash Flow Waterfall Engine
// Synthetic deal: $500M pool, 6 tranches, quarterly periods, 8y maturity, 5y reinvestment
// Simplifications (documented for the appendix):
//  - Immediate recoveries (no lag), flat base rate, single asset spread
//  - Two OC tests (senior / junior), no IC tests, no call option
//  - Equity issued at par; defaults hit performing balance at start of quarter

const DEAL = {
  poolSize: 500e6,
  assetSpread: 0.0385, // SOFR+350 running + ~35bp amortized OID
  baseRate: 0.04,
  nQuarters: 32,
  reinvestEnd: 20,
  seniorFee: 0.004,
  subFee: 0.001,
  tranches: [
    { name: 'AAA', pct: 0.63, spread: 0.0150 },
    { name: 'AA',  pct: 0.12, spread: 0.0190 },
    { name: 'A',   pct: 0.06, spread: 0.0250 },
    { name: 'BBB', pct: 0.05, spread: 0.0360 },
    { name: 'BB',  pct: 0.04, spread: 0.0700 },
    { name: 'EQ',  pct: 0.10, spread: null },
  ],
  ocSeniorTrigger: 1.17,  // collateral / (AAA+AA+A); initial ratio 1.235 → ~5% cushion
  ocJuniorTrigger: 1.04,  // collateral / (AAA..BB); initial ratio 1.111 → ~6% cushion
};

function q(rate) { return 1 - Math.pow(1 - rate, 0.25); } // annual -> quarterly

function runDeal(p) {
  // p: { cdr, recovery, cpr } annual rates
  const d = DEAL;
  const bal = d.tranches.map(t => t.pct * d.poolSize); // tranche balances
  const eqIdx = d.tranches.length - 1;
  let collateral = d.poolSize;
  let cumDefaults = 0, cumLosses = 0;

  const hist = {
    collateral: [collateral],
    tranches: d.tranches.map((t, i) => [bal[i]]),
    eqCF: [-bal[eqIdx]],
    ocSenior: [], ocJunior: [],
    ocSeniorPass: [], ocJuniorPass: [],
    checks: [],
    trancheIntPaid: d.tranches.map(() => 0),
    trancheIntDue: d.tranches.map(() => 0),
  };

  for (let t = 1; t <= d.nQuarters; t++) {
    // --- Collateral dynamics ---
    const defaults = collateral * q(p.cdr);
    const recoveries = defaults * p.recovery;
    const loss = defaults - recoveries;
    cumDefaults += defaults;
    cumLosses += loss;
    const afterDef = collateral - defaults;
    const prepays = afterDef * q(p.cpr);

    const interestColl = collateral * (d.baseRate + d.assetSpread) / 4;
    let principalColl = recoveries + prepays;
    collateral = afterDef - prepays;

    // Final quarter: pool liquidates at par
    if (t === d.nQuarters) { principalColl += collateral; collateral = 0; }

    // --- Reinvestment ---
    if (t <= d.reinvestEnd && t < d.nQuarters) {
      collateral += principalColl;
      principalColl = 0;
    }

    // --- Interest waterfall ---
    let cash = interestColl;
    const totalColl = collateral + principalColl; // for OC ratio, measured after flows
    const pay = (amt) => { const x = Math.min(cash, amt); cash -= x; return x; };

    let distributed = 0;
    distributed += pay(d.poolSize > 0 ? (collateral + principalColl) * d.seniorFee / 4 : 0);

    // Senior interest AAA -> A (senior OC test sits before BBB, per priority of payments)
    const intDue = d.tranches.map((tr, i) =>
      tr.spread === null ? 0 : bal[i] * (d.baseRate + tr.spread) / 4);
    for (let i = 0; i < 3; i++) { // AAA, AA, A
      const paid = pay(intDue[i]);
      hist.trancheIntPaid[i] += paid; hist.trancheIntDue[i] += intDue[i];
      distributed += paid;
    }

    // OC tests
    const seniorDebt = bal[0] + bal[1] + bal[2];
    const juniorDebt = seniorDebt + bal[3] + bal[4];
    const ocSenior = seniorDebt > 0 ? totalColl / seniorDebt : Infinity;
    const ocJunior = juniorDebt > 0 ? totalColl / juniorDebt : Infinity;
    hist.ocSenior.push(ocSenior); hist.ocJunior.push(ocJunior);
    hist.ocSeniorPass.push(ocSenior >= d.ocSeniorTrigger);
    hist.ocJuniorPass.push(ocJunior >= d.ocJuniorTrigger);

    // Senior OC failure: divert remaining interest to pay down AAA until cured
    if (ocSenior < d.ocSeniorTrigger && cash > 0) {
      // amount of AAA paydown needed to cure: totalColl/(seniorDebt - x) = trigger
      const need = seniorDebt - totalColl / d.ocSeniorTrigger;
      const divert = Math.min(cash, Math.max(0, need), bal[0]);
      bal[0] -= divert; cash -= divert; distributed += divert;
    }

    // BBB then BB interest
    for (const i of [3, 4]) {
      const paid = pay(intDue[i]);
      hist.trancheIntPaid[i] += paid; hist.trancheIntDue[i] += intDue[i];
      distributed += paid;
    }

    // Junior OC failure: divert to AAA paydown
    if (ocJunior < d.ocJuniorTrigger && cash > 0) {
      const jd = bal[0] + bal[1] + bal[2] + bal[3] + bal[4];
      const need = jd - totalColl / d.ocJuniorTrigger;
      const divert = Math.min(cash, Math.max(0, need), bal[0]);
      bal[0] -= divert; cash -= divert; distributed += divert;
    }

    distributed += pay((collateral + principalColl) * d.subFee / 4);

    // Residual interest -> equity
    let eqCash = cash; distributed += cash; cash = 0;

    // --- Principal waterfall (sequential) ---
    let prin = principalColl;
    for (let i = 0; i < 5 && prin > 0; i++) {
      const x = Math.min(prin, bal[i]);
      bal[i] -= x; prin -= x; distributed += x;
    }
    eqCash += prin; distributed += prin; prin = 0; // residual principal -> equity

    // Cash conservation check
    hist.checks.push(Math.abs(interestColl + principalColl - distributed) < 1e-6);

    hist.eqCF.push(eqCash);
    hist.collateral.push(collateral);
    d.tranches.forEach((tr, i) => hist.tranches[i].push(bal[i]));
  }

  // Tranche principal losses = unpaid balance at maturity
  const losses = d.tranches.map((tr, i) => bal[i]);
  const eqIRR = irr(hist.eqCF);

  return {
    hist, losses, cumDefaults, cumLosses,
    eqIRR: eqIRR === null ? null : Math.pow(1 + eqIRR, 4) - 1,
    intShortfall: d.tranches.map((tr, i) => hist.trancheIntDue[i] - hist.trancheIntPaid[i]),
  };
}

function irr(cfs) {
  const npv = r => cfs.reduce((s, c, i) => s + c / Math.pow(1 + r, i), 0);
  let lo = -0.99, hi = 5;
  if (npv(lo) * npv(hi) > 0) return npv(0) > 0 ? hi : null;
  for (let k = 0; k < 200; k++) {
    const mid = (lo + hi) / 2;
    if (npv(lo) * npv(mid) <= 0) hi = mid; else lo = mid;
  }
  return (lo + hi) / 2;
}

if (typeof module !== 'undefined') module.exports = { runDeal, DEAL };

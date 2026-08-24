/* ---------------------------------------------------------------------------
   Reference orbit, computed at arbitrary precision.

   Perturbation rendering needs exactly one high-precision orbit. Every pixel
   is then computed as a small offset from it, in ordinary float32 — which is
   what lets the GPU go far below the depth double precision can reach.

   Fixed-point via BigInt: a number x is stored as round(x * 2^SHIFT).
   --------------------------------------------------------------------------- */

function makeCtx(shift){
  const S = BigInt(shift);
  const ONE = 1n << S;
  return {
    S, ONE,
    mul: (a, b) => (a * b) >> S,
    fromString(str){
      let s = String(str).trim();
      let neg = false;
      if (s[0] === '-'){ neg = true; s = s.slice(1); }
      else if (s[0] === '+') s = s.slice(1);
      const [ip, fp = ''] = s.split('.');
      let v = BigInt(ip || '0') << S;
      if (fp.length){
        const den = 10n ** BigInt(fp.length);
        // round-to-nearest, so a decimal string survives the round-trip exactly
        v += ((BigInt(fp) << S) + den / 2n) / den;
      }
      return neg ? -v : v;
    },
    toNumber(v){
      // take the top 64 significant bits and scale — Number(ONE) overflows
      // once SHIFT passes 1024, so we must not divide by it directly
      if (v === 0n) return 0;
      const neg = v < 0n; if (neg) v = -v;
      const bits = v.toString(2).length;
      const drop = bits > 64 ? bits - 64 : 0;
      const r = Number(v >> BigInt(drop)) * Math.pow(2, drop - shift);
      return neg ? -r : r;
    }
  };
}

/* decimal string ↔ fixed point, so coordinates survive a URL round-trip */
function toDecimalString(v, shift, digits){
  const S = BigInt(shift);
  const neg = v < 0n;
  if (neg) v = -v;
  const ip = v >> S;
  let frac = v - (ip << S);
  const ds = [];
  for (let i = 0; i <= digits; i++){          // one guard digit, then round
    frac *= 10n;
    const d = frac >> S;
    ds.push(Number(d));
    frac -= d << S;
  }
  if (ds.pop() >= 5){                          // propagate the carry
    let i = ds.length - 1;
    for (; i >= 0; i--){ if (++ds[i] < 10) break; ds[i] = 0; }
    if (i < 0) return (neg ? '-' : '') + (ip + 1n).toString() + '.' + ds.join('');
  }
  return (neg ? '-' : '') + ip.toString() + '.' + ds.join('');
}

self.onmessage = (e) => {
  const { cxStr, cyStr, maxIter, shift, id } = e.data;
  const ctx = makeCtx(shift);

  const cx = ctx.fromString(cxStr);
  const cy = ctx.fromString(cyStr);

  // Z is bounded by 2 while it stays in the set, so float32 holds it fine
  const zx = new Float32Array(maxIter);
  const zy = new Float32Array(maxIter);

  let x = 0n, y = 0n;
  const FOUR = 4n << ctx.S;
  let n = 0;

  for (; n < maxIter; n++){
    zx[n] = ctx.toNumber(x);
    zy[n] = ctx.toNumber(y);

    const x2 = ctx.mul(x, x);
    const y2 = ctx.mul(y, y);
    if (x2 + y2 > FOUR){ n++; break; }          // reference escaped

    const nx = x2 - y2 + cx;
    const ny = (ctx.mul(x, y) << 1n) + cy;
    x = nx; y = ny;

    if ((n & 1023) === 0) {
      self.postMessage({ type: 'progress', id, n, maxIter });
    }
  }

  self.postMessage({
    type: 'orbit', id, length: n,
    zx: zx.subarray(0, n), zy: zy.subarray(0, n)
  }, [zx.buffer, zy.buffer]);
};

self.__test = { makeCtx, toDecimalString };

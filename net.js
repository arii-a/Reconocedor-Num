// CNN de inferencia en JS puro (misma arquitectura que train.py, con BatchNorm ya fusionada).
(function (root) {
  function decode(b64) {
    const bin = atob(b64);
    const u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return new Float32Array(u8.buffer);
  }

  // conv 3x3, padding 1, + ReLU. input: [ci, h, w] -> output: [co, h, w]
  function conv3(input, ci, co, h, w, wt, bias) {
    const out = new Float32Array(co * h * w);
    const hw = h * w;
    for (let o = 0; o < co; o++) {
      const outBase = o * hw;
      out.fill(bias[o], outBase, outBase + hw);
      for (let c = 0; c < ci; c++) {
        const inBase = c * hw;
        const wBase = (o * ci + c) * 9;
        for (let ky = 0; ky < 3; ky++) {
          for (let kx = 0; kx < 3; kx++) {
            const k = wt[wBase + ky * 3 + kx];
            if (k === 0) continue;
            const dy = ky - 1, dx = kx - 1;
            const y0 = Math.max(0, -dy), y1 = Math.min(h, h - dy);
            const x0 = Math.max(0, -dx), x1 = Math.min(w, w - dx);
            for (let y = y0; y < y1; y++) {
              const ro = outBase + y * w, ri = inBase + (y + dy) * w + dx;
              for (let x = x0; x < x1; x++) out[ro + x] += k * input[ri + x];
            }
          }
        }
      }
      for (let i = outBase; i < outBase + hw; i++) if (out[i] < 0) out[i] = 0;
    }
    return out;
  }

  function maxpool(input, c, h, w) {
    const oh = h >> 1, ow = w >> 1;
    const out = new Float32Array(c * oh * ow);
    for (let k = 0; k < c; k++)
      for (let y = 0; y < oh; y++)
        for (let x = 0; x < ow; x++) {
          const b = k * h * w + 2 * y * w + 2 * x;
          out[k * oh * ow + y * ow + x] = Math.max(input[b], input[b + 1], input[b + w], input[b + w + 1]);
        }
    return out;
  }

  function linear(input, nIn, nOut, wt, bias, relu) {
    const out = new Float32Array(nOut);
    for (let o = 0; o < nOut; o++) {
      let s = bias[o];
      const b = o * nIn;
      for (let i = 0; i < nIn; i++) s += wt[b + i] * input[i];
      out[o] = relu && s < 0 ? 0 : s;
    }
    return out;
  }

  function createNet(width, data) {
    const f = decode(data);
    const chans = [1, 32 * width, 64 * width, 128 * width];
    let pos = 0;
    const take = (n) => { const a = f.subarray(pos, pos + n); pos += n; return a; };
    const convs = [];
    for (let b = 0; b < 3; b++) {
      const ci = chans[b], co = chans[b + 1];
      convs.push({ ci, co, w: take(co * ci * 9), b: take(co) });
      convs.push({ ci: co, co, w: take(co * co * 9), b: take(co) });
    }
    const fcIn = chans[3] * 9;
    const fc1 = { w: take(256 * fcIn), b: take(256) };
    const fc2 = { w: take(10 * 256), b: take(10) };
    if (pos !== f.length) throw new Error("Pesos con tamano inesperado");

    // input: Float32Array(784) ya normalizada ((x-media)/desv). Devuelve logits[10].
    function logits(input) {
      let x = input, h = 28, w = 28, c = 1;
      for (let b = 0; b < 3; b++) {
        const a = convs[2 * b], bb = convs[2 * b + 1];
        x = conv3(x, a.ci, a.co, h, w, a.w, a.b);
        x = conv3(x, bb.ci, bb.co, h, w, bb.w, bb.b);
        x = maxpool(x, bb.co, h, w);
        h >>= 1; w >>= 1; c = bb.co;
      }
      x = linear(x, fcIn, 256, fc1.w, fc1.b, true);
      return linear(x, 256, 10, fc2.w, fc2.b, false);
    }

    function softmax(l) {
      const m = Math.max(...l);
      const e = Array.from(l, (v) => Math.exp(v - m));
      const s = e.reduce((a, b) => a + b, 0);
      return e.map((v) => v / s);
    }

    return { logits, predict: (input) => softmax(logits(input)) };
  }

  root.createNet = createNet;
})(typeof window !== "undefined" ? window : globalThis);

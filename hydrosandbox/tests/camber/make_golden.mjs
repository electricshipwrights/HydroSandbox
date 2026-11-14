// Reference data for hydrosandbox.hull.camber, produced by camber itself.
//
// camber (github.com/electricshipwrights/camber) is TypeScript; this script is
// bundled against a checkout of it with esbuild and run under Node:
//
//   npx esbuild hydrosandbox/tests/camber/make_golden.mjs --bundle --platform=node \
//       --format=esm --alias:@camber=$CAMBER/src --outfile=/tmp/make_golden.bundle.mjs
//   node /tmp/make_golden.bundle.mjs hydrosandbox/tests/camber/documents \
//       > hydrosandbox/tests/camber/golden.json
//
// It reads every *.json document in the given directory and writes, for each,
// what the port has to reproduce: the decoded state (after camber's v1 → v2
// conversion), the plan curve and its tangent, the sheer trim, the lofted
// stations, the section curves, the trimmed half-sections, and hydrostatics; and
// for families of documents, camber's promote + blend.

import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { parseDocument } from "@camber/core/json";
import { assemble } from "@camber/core/runtime";
import { sweptSection, computeHullSampling } from "@camber/core/mesh";
import { frameAt, sectionAt } from "@camber/core/model";
import { hydrostatics } from "@camber/core/hydro";
import { pchipSlopes, knuckleSlopes } from "@camber/core/pchip";
import { crChain, centripetalParams } from "@camber/core/spline";
import { blendState } from "@camber/interpolate/blend";
import { promoteFamily } from "@camber/core/promote";

const dir = process.argv[2];
const range = (n, f) => Array.from({ length: n }, (_, i) => f(i));
const U_GRID = range(41, (i) => i / 40);
const clone = (x) => JSON.parse(JSON.stringify(x));

const HYDRO_KEYS = [
  "lwl", "bwl", "draft", "vol", "waterplaneArea", "wettedArea", "lcb", "lcf", "kb",
  "bmt", "bml", "midshipArea", "maxSectionArea", "xAft", "xFwd", "closed",
];
function hydroOf(model, ns = 72, nl = 20) {
  const h = hydrostatics(model, computeHullSampling(model, ns, nl));
  return h ? Object.fromEntries(HYDRO_KEYS.map((k) => [k, h[k]])) : null;
}

function describe(state) {
  const model = assemble(state);
  const out = { state: clone(state) };
  out.plan = U_GRID.map((u) => ({ u, at: model.plan.at(u), d: model.plan.d(u) }));
  const x0 = state.sheerPlan[0].x,
    x1 = state.sheerPlan[state.sheerPlan.length - 1].x;
  out.planUAtX = range(9, (i) => {
    const x = x0 + ((x1 - x0) * (i + 0.5)) / 9;
    return { x, u: model.plan.uAtX(x) };
  });
  out.trimZ = range(33, (i) => {
    const x = x0 - 0.05 * (x1 - x0) + (1.1 * (x1 - x0) * i) / 32;
    return { x, z: model.trimZ(x) };
  });
  out.loft = U_GRID.map((u) => ({ u, ...model.loft.at(u) }));
  out.frames = U_GRID.map((u) => ({ u, ...frameAt(model, u) }));
  out.sections = range(11, (i) => {
    const u = i / 10,
      sec = sectionAt(model, u),
      vs = range(4 * sec.vmax + 1, (j) => j / 4);
    return { u, v: vs, at: vs.map((v) => sec.at(v)), d: vs.map((v) => sec.d(v)) };
  });
  out.swept = U_GRID.map((u) => {
    const s = sweptSection(model, u, 4, true);
    return { u, empty: s.empty, keel: s.keel, pts: s.pts };
  });
  out.hydro = hydroOf(model);
  out.hydroFine = hydroOf(model, 240, 40);
  return out;
}

const docs = {};
for (const f of readdirSync(dir).filter((f) => f.endsWith(".json")).sort()) {
  const text = readFileSync(join(dir, f), "utf8");
  docs[f.replace(/\.json$/, "")] = { text, parsed: parseDocument(text) };
}

const golden = { documents: {}, families: [], primitives: {} };
for (const [name, { parsed }] of Object.entries(docs))
  golden.documents[name] = describe(parsed.state);

// families: promote to a common topology, then blend
const FAMILIES = [
  { names: ["skiff", "dory"], weights: [[0.3, 0.7], [0.5, 0.5]] },
  { names: ["launch", "skiff"], weights: [[0.6, 0.4]] },
  { names: ["skiff", "dory", "three_template_v1"], weights: [[0.2, 0.5, 0.3]] },
];
for (const fam of FAMILIES) {
  if (!fam.names.every((n) => n in docs)) continue;
  const datas = fam.names.map((n) => clone(docs[n].parsed.hull));
  promoteFamily(datas);
  const first = docs[fam.names[0]].parsed.state;
  const trim = { waterline: first.waterline, deckTrim: first.deckTrim };
  const hulls = datas.map((data, i) => ({ name: fam.names[i], data }));
  golden.families.push({
    names: fam.names,
    trim,
    promoted: clone(datas),
    blends: fam.weights.map((w) => ({ weights: w, ...describe(blendState(hulls, w, trim)) })),
  });
}

// the spline primitives on their own, on awkward data
const xs = [0, 1, 2.5, 3, 4.2, 6],
  ys = [0, 0.9, 1.0, 0.2, 0.25, 3];
golden.primitives.pchip = { xs, ys, m: pchipSlopes(xs, ys) };
golden.primitives.knuckle = { xs, ys, ks: [0, 1, 0.5, 0, 0.25, 0], ...knuckleSlopes(xs, ys, [0, 1, 0.5, 0, 0.25, 0]) };
const pts = [[0, 0], [0.1, -0.8], [0.6, -1.2], [1.5, -1.3], [2.0, -1.5]];
const t = centripetalParams(pts);
golden.primitives.crChain = { pts, ks: [1, 0.4, 1, 0, 1], t, chain: crChain(pts, t, [1, 0.4, 1, 0, 1]) };

process.stdout.write(JSON.stringify(golden));

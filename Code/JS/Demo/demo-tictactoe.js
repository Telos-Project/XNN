/**
 * demo-tictactoe.js
 *
 * Demonstrates xnn-utils.js by training an XNN to play tic-tac-toe via
 * perspective-canonicalized self-play (the same weights play both X and O)
 * with TD(0) value learning and DFA-based hidden-layer credit assignment,
 * then evaluating it against a random opponent and a one-ply-lookahead
 * opponent, logging metrics to the console, and writing the trained model
 * and test metrics to disk (each as its own file, auto-numbered if the
 * target filename already exists).
 *
 * SELF-DESCRIBING SERIALIZATION: every piece of architecture metadata this
 * demo needs (which neurons are inputs/outputs, which hidden neurons double
 * as the value readout, each DFA-trained hidden neuron's fixed feedback
 * weight) lives inside that neuron's own `factors` field -- using
 * xnn-utils.js's existing `factors.role` convention plus two demo-specific
 * factors (`isValueNeuron`, `dfaWeight`). The saved model file is therefore
 * exactly `{ matrix, vector }`, xnn-utils.js's own native shape, with
 * nothing external required to reconstruct which neuron does what -- load
 * the file, call deriveArchitecture() on it, done.
 *
 * ARCHITECTURE (51 neurons total):
 *   inputs  [0..17]  9 cells x {my piece, opponent's piece} (perspective-
 *                    invariant: "mine"/"opponent's" rather than "X"/"O", so
 *                    a single set of weights can play either side)
 *   hidden  [18..41] 24 neurons; 4 of them ALSO have factors.isValueNeuron
 *                    = true (the value readout), the remaining 20 each carry
 *                    factors.dfaWeight (their fixed DFA feedback weight)
 *   outputs [42..50] one per cell, argmax among legal moves selects a move
 *
 * A DESIGN NOTE ON DEVIATING FROM THE ORIGINAL PYTHON SELF-PLAY ALGORITHM:
 * xnn-utils.js's train() resets each neuron's eligibility trace once it's
 * used. The original Python implementation deferred training a move's
 * transition until the NEXT move's value estimate was available as a
 * bootstrap target -- but doing that here would mean two moves' worth of
 * ticks accumulate in the trace before the first move is ever trained,
 * incorrectly crediting the second move's activity to the first move's
 * outcome. Since step() and train() never mutate their input (see
 * xnn-utils.js), the clean fix is a discardable PEEK: after a move is
 * decided (trace still contains ONLY that move's ticks), simulate the next
 * mover's deliberation on a throwaway model copy purely to read a bootstrap
 * value, train the current move immediately using its own uncontaminated
 * trace, and only THEN commit the real next move by continuing from the
 * (still-untrained-on-this-transition) live model.
 */

const fs = require("fs");
const path = require("path");
const { create, step, train } = require("./xnnUtils.js");

const N_INPUT = 18;
const N_HIDDEN = 24;
const N_OUTPUT = 9;
const N_VALUE = 4;
const N_TOTAL = N_INPUT + N_HIDDEN + N_OUTPUT;
const STEPS_PER_DECISION = 3;

// ---------------------------------------------------------------------------
// Deterministic PRNG (so a run is reproducible given a seed)
// ---------------------------------------------------------------------------

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------------------
// Architecture setup and derivation -- the only two functions in this file
// that know anything about the fixed [0..17][18..41][42..50] layout. Once a
// model has been created, every other function reads its architecture back
// out of factors.role / factors.isValueNeuron / factors.dfaWeight, exactly
// as it would if you had just loaded a saved model.json from disk instead.
// ---------------------------------------------------------------------------

function createTicTacToeModel(seed) {
  const roles = [
    ...Array(N_INPUT).fill("input"),
    ...Array(N_HIDDEN).fill("hidden"),
    ...Array(N_OUTPUT).fill("output"),
  ];

  const model = create(N_TOTAL, {
    seed,
    weightRange: [-1, 1],
    biasRange: [0, 1],
    targetActivity: 0.25,
    roles,
  });

  // Annotate the hidden layer: the first N_VALUE hidden neurons double as
  // the value readout; the rest each get a fixed (never trained) DFA
  // feedback weight. Demo-specific setup, done as one extra pass over the
  // freshly-created factors -- not part of xnn-utils.js itself.
  const rng = mulberry32(seed + 500);
  let hiddenSeen = 0;
  const vector = model.vector.map((n) => {
    if (n.factors.role !== "hidden") return n;
    hiddenSeen++;
    const isValueNeuron = hiddenSeen <= N_VALUE;
    const factors = { ...n.factors, isValueNeuron };
    if (!isValueNeuron) factors.dfaWeight = rng() * 2 - 1;
    return { state: n.state, factors };
  });

  return { matrix: model.matrix, vector };
}

// Reconstructs everything the rest of this file needs to know about a
// model purely from its own factors -- works identically whether `model`
// was just created above or loaded fresh from a saved JSON file.
function deriveArchitecture(model) {
  const inputIdx = [];
  const outputIdx = [];
  const valueIdx = [];
  const dfaHiddenIdx = [];
  const feedbackB = [];

  model.vector.forEach((n, i) => {
    const f = n.factors;
    if (f.role === "input") inputIdx.push(i);
    else if (f.role === "output") outputIdx.push(i);
    else if (f.role === "hidden") {
      if (f.isValueNeuron) valueIdx.push(i);
      else {
        dfaHiddenIdx.push(i);
        feedbackB.push(f.dfaWeight);
      }
    }
  });

  return { inputIdx, outputIdx, valueIdx, dfaHiddenIdx, feedbackB };
}

// ---------------------------------------------------------------------------
// Tic-tac-toe mechanics
// ---------------------------------------------------------------------------

const WIN_LINES = [
  [0, 1, 2], [3, 4, 5], [6, 7, 8],
  [0, 3, 6], [1, 4, 7], [2, 5, 8],
  [0, 4, 8], [2, 4, 6],
];

function checkWinner(board) {
  for (const [a, b, c] of WIN_LINES) {
    if (board[a] !== 0 && board[a] === board[b] && board[b] === board[c]) return board[a];
  }
  if (!board.includes(0)) return 0;
  return null;
}

function legalMoves(board) {
  const out = [];
  for (let i = 0; i < 9; i++) if (board[i] === 0) out.push(i);
  return out;
}

// Perspective-invariant encoding: "my pieces" / "opponent's pieces", not
// "X pieces" / "O pieces" -- lets one set of weights play either side.
function canonicalize(board, mover) {
  const other = mover === 1 ? 2 : 1;
  const vec = new Array(N_INPUT).fill(0);
  for (let i = 0; i < 9; i++) {
    if (board[i] === mover) vec[i] = 1;
    else if (board[i] === other) vec[9 + i] = 1;
  }
  return vec;
}

function toInputMap(vec, inputIdx) {
  const map = {};
  for (let i = 0; i < vec.length; i++) map[inputIdx[i]] = vec[i];
  return map;
}

// ---------------------------------------------------------------------------
// Model-state helpers not provided by xnn-utils.js itself
// ---------------------------------------------------------------------------

function resetGameState(model) {
  return {
    matrix: model.matrix,
    vector: model.vector.map((n) => ({ state: 0, factors: { ...n.factors, trace: 0 } })),
  };
}

function clearStates(model, indices) {
  const idxSet = new Set(indices);
  return {
    matrix: model.matrix,
    vector: model.vector.map((n, i) => (idxSet.has(i) ? { state: 0, factors: n.factors } : n)),
  };
}

function readValue(model, valueIdx) {
  const avg = valueIdx.reduce((s, idx) => s + model.vector[idx].state, 0) / valueIdx.length;
  return 2 * avg - 1; // map [0,1] -> [-1,1]
}

function deliberate(model, board, mover, arch) {
  const inputMap = toInputMap(canonicalize(board, mover), arch.inputIdx);
  let m = model;
  for (let s = 0; s < STEPS_PER_DECISION; s++) m = step(m, inputMap, {});
  const outStates = arch.outputIdx.map((idx) => m.vector[idx].state);
  const value = readValue(m, arch.valueIdx);
  return { model: m, outStates, value };
}

function chooseMove(outStates, legal, epsilon, rng) {
  if (rng() < epsilon) return legal[Math.floor(rng() * legal.length)];
  let best = -Infinity;
  let bestMove = legal[0];
  for (const i of legal) {
    if (outStates[i] > best) {
      best = outStates[i];
      bestMove = i;
    }
  }
  return bestMove;
}

// ---------------------------------------------------------------------------
// Self-play training (no labels, no external opponent)
// ---------------------------------------------------------------------------

function playSelfPlayGame(model, epsilon, rng, doTrain, arch, cw, cb) {
  let m = resetGameState(model);
  const board = new Array(9).fill(0);
  let mover = 1;

  while (true) {
    const legal = legalMoves(board);
    const { model: afterDeliberation, outStates, value } = deliberate(m, board, mover, arch);
    const move = chooseMove(outStates, legal, epsilon, rng);
    const chosenGlobalIdx = arch.outputIdx[move];
    m = clearStates(afterDeliberation, arch.outputIdx);

    board[move] = mover;
    const winner = checkWinner(board);

    if (winner !== null) {
      const r = winner === mover ? 1 : 0;
      const tdError = r - value;
      if (doTrain) {
        m = train(m, {
          mode: "td",
          chosenIndex: chosenGlobalIdx,
          valueIndices: arch.valueIdx,
          tdError,
          hiddenIndices: arch.dfaHiddenIdx,
          feedbackVectorB: arch.feedbackB,
          cw,
          cb,
        });
      }
      return { winner, model: m };
    }

    // Discardable peek (see file-level doc comment): m's trace here still
    // reflects only this move's ticks, so training immediately below is
    // correct regardless of what this peek does afterward.
    const nextMover = mover === 1 ? 2 : 1;
    const peek = deliberate(m, board, nextMover, arch); // m itself is untouched by this
    const target = -peek.value; // zero-sum: good for them now = bad for us

    const tdError = target - value;
    if (doTrain) {
      m = train(m, {
        mode: "td",
        chosenIndex: chosenGlobalIdx,
        valueIndices: arch.valueIdx,
        tdError,
        hiddenIndices: arch.dfaHiddenIdx,
        feedbackVectorB: arch.feedbackB,
        cw,
        cb,
      });
    }

    mover = nextMover;
  }
}

// ---------------------------------------------------------------------------
// Evaluation against fixed external opponents (never used for training)
// ---------------------------------------------------------------------------

function randomOpponent(board, rng) {
  const legal = legalMoves(board);
  return legal[Math.floor(rng() * legal.length)];
}

function smartOpponent(board, rng) {
  const legal = legalMoves(board);
  for (const m of legal) {
    const trial = board.slice();
    trial[m] = 2;
    if (checkWinner(trial) === 2) return m;
  }
  return legal[Math.floor(rng() * legal.length)];
}

function evaluateVsOpponent(model, opponentFn, numGames, seed, arch) {
  const rng = mulberry32(seed);
  let wins = 0, draws = 0, losses = 0;

  for (let g = 0; g < numGames; g++) {
    let m = resetGameState(model);
    const board = new Array(9).fill(0);
    let turn = 1;

    while (true) {
      if (turn === 1) {
        const legal = legalMoves(board);
        const { model: after, outStates } = deliberate(m, board, 1, arch);
        const move = chooseMove(outStates, legal, 0, rng);
        m = clearStates(after, arch.outputIdx);
        board[move] = 1;
      } else {
        const move = opponentFn(board, rng);
        board[move] = 2;
        m = deliberate(m, board, 1, arch).model; // passive observation, no Clearing
      }

      const winner = checkWinner(board);
      if (winner !== null) {
        if (winner === 1) wins++;
        else if (winner === 0) draws++;
        else losses++;
        break;
      }
      turn = turn === 1 ? 2 : 1;
    }
  }

  return { win: wins / numGames, draw: draws / numGames, loss: losses / numGames };
}

// ---------------------------------------------------------------------------
// File-writing helper: auto-number the filename if it already exists
// ---------------------------------------------------------------------------

function getAvailablePath(basePath) {
  if (!fs.existsSync(basePath)) return basePath;
  const ext = path.extname(basePath);
  const stem = basePath.slice(0, basePath.length - ext.length);
  let i = 1;
  while (fs.existsSync(`${stem}-${i}${ext}`)) i++;
  return `${stem}-${i}${ext}`;
}

function writeJson(basePath, data) {
  const target = getAvailablePath(basePath);
  fs.writeFileSync(target, JSON.stringify(data, null, 2));
  return target;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

function main() {
  const SEED = 1;
  const GAMES = 6000;
  const EVAL_EVERY = 500;
  const EVAL_GAMES = 100;
  const CW = 0.1;
  const CB = 0.1;
  const EPS_START = 0.3;
  const EPS_END = 0.02;

  const rng = mulberry32(SEED);
  let model = createTicTacToeModel(SEED);
  const arch = deriveArchitecture(model); // exactly what you'd call after loading a saved file, too

  console.log(`Training XNN on tic-tac-toe via self-play: ${GAMES} games, ${N_TOTAL} neurons`);
  console.log(
    `(${arch.inputIdx.length} input, ${arch.valueIdx.length} value + ${arch.dfaHiddenIdx.length} DFA-trained hidden, ${arch.outputIdx.length} output)`
  );
  console.log("");

  const history = [];
  let bestScore = -Infinity;
  let bestModel = model;

  for (let g = 0; g < GAMES; g++) {
    const epsilon = EPS_START + (EPS_END - EPS_START) * (g / GAMES);
    const result = playSelfPlayGame(model, epsilon, rng, true, arch, CW, CB);
    model = result.model;

    if (g % EVAL_EVERY === 0 || g === GAMES - 1) {
      const vsRandom = evaluateVsOpponent(model, randomOpponent, EVAL_GAMES, 9000, arch);
      const vsSmart = evaluateVsOpponent(model, smartOpponent, EVAL_GAMES, 9000, arch);
      const score = vsRandom.win + vsSmart.win - vsRandom.loss - vsSmart.loss;

      history.push({ game: g, epsilon, vsRandom, vsSmart, score });

      if (score > bestScore) {
        bestScore = score;
        bestModel = model;
      }

      console.log(
        `game ${String(g).padStart(5)}  eps=${epsilon.toFixed(2)}  ` +
        `vs_random W${vsRandom.win.toFixed(2)}/D${vsRandom.draw.toFixed(2)}/L${vsRandom.loss.toFixed(2)}  ` +
        `vs_smart W${vsSmart.win.toFixed(2)}/D${vsSmart.draw.toFixed(2)}/L${vsSmart.loss.toFixed(2)}  ` +
        `score=${score.toFixed(2)}${score === bestScore ? "  <- best" : ""}`
      );
    }
  }

  console.log("");
  console.log("Training complete. Evaluating best checkpoint (500 games each)...");
  const finalVsRandom = evaluateVsOpponent(bestModel, randomOpponent, 500, 42, arch);
  const finalVsSmart = evaluateVsOpponent(bestModel, smartOpponent, 500, 42, arch);

  console.log(`  vs random: win=${finalVsRandom.win.toFixed(3)} draw=${finalVsRandom.draw.toFixed(3)} loss=${finalVsRandom.loss.toFixed(3)}`);
  console.log(`  vs smart:  win=${finalVsSmart.win.toFixed(3)} draw=${finalVsSmart.draw.toFixed(3)} loss=${finalVsSmart.loss.toFixed(3)}`);

  // The saved file is exactly bestModel -- { matrix, vector } -- xnn-utils.js's
  // own native shape. No wrapper, no external metadata: every neuron's role,
  // value-readout membership, and fixed DFA weight already lives in its own
  // factors, so deriveArchitecture(loadedModel) reconstructs everything this
  // script needs from the file alone.
  const modelPath = writeJson(path.join(__dirname, "tictactoe-model.json"), bestModel);
  console.log(`\nTrained model written to: ${modelPath}`);

  const metricsOut = {
    seed: SEED,
    games: GAMES,
    bestCheckpointScore: bestScore,
    finalEvaluation: { vsRandom: finalVsRandom, vsSmart: finalVsSmart },
    trainingHistory: history,
  };
  const metricsPath = writeJson(path.join(__dirname, "tictactoe-metrics.json"), metricsOut);
  console.log(`Test metrics written to: ${metricsPath}`);

  // Sanity check that the serialization is genuinely self-sufficient: reload
  // the file from disk exactly as a fresh process would, re-derive the
  // architecture from nothing but its factors, and confirm evaluation
  // reproduces the same numbers as the in-memory model just evaluated above.
  const reloaded = JSON.parse(fs.readFileSync(modelPath, "utf8"));
  const reloadedArch = deriveArchitecture(reloaded);
  const reloadedVsRandom = evaluateVsOpponent(reloaded, randomOpponent, 500, 42, reloadedArch);
  const matches =
    reloadedVsRandom.win === finalVsRandom.win &&
    reloadedVsRandom.draw === finalVsRandom.draw &&
    reloadedVsRandom.loss === finalVsRandom.loss;
  console.log(
    `\nReload-from-disk sanity check (re-derive architecture from factors alone): ${matches ? "PASSED" : "FAILED"}`
  );
}

main();

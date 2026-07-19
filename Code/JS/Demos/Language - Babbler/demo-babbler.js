/**
 * demo-babbler.js
 *
 * A "babbler": an XNN that receives tokens one at a time in continuous
 * time, must decide FOR ITSELF, with no external cue, when it is
 * appropriate to speak (a dedicated gate neuron, separate from word
 * choice), and if so, which word to say. Trained on a tiny, genuine
 * English-subset grammar so a human observer can read the exchanges as
 * real (if extremely compressed) conversation.
 *
 * WHAT IS ACTUALLY NEW HERE, relative to every prior experiment in this
 * project: every previous task (tic-tac-toe, delayed-recall, foraging) had
 * an externally-imposed decision point -- a probe tick, a move request, an
 * environment step. Nothing before this asked the network to decide,
 * unprompted, "now I act." The gate neuron exists specifically to test
 * that, separated from word-choice quality so the two questions (did it
 * pick the right MOMENT vs did it pick the right WORD) can be measured
 * independently.
 *
 * STAGED SUCCESS CRITERIA (deliberately not "does it converse" -- see the
 * accompanying discussion for why that would be an undiagnosable first
 * milestone):
 *   1. Does the gate neuron's firing correlate with grammatically
 *      appropriate moments, above the untrained baseline?
 *   2. Conditional on correct timing, is word choice correct above chance?
 *
 * EXCHANGE STRUCTURE (fixed, regular timing -- deliberate for a first
 * build, so timing-accuracy and word-accuracy can be measured cleanly
 * without also confounding "did it infer irregular timing"):
 *   trigger word, gap, trigger word, gap, ... trigger word,
 *   [3 ticks end-of-turn pause],
 *   [1 response tick -- the ONLY tick where speaking is correct, if a
 *    response is expected at all],
 *   [1 cleanup tick]
 * Nothing in the input stream marks the response tick specially -- it is
 * only distinguishable by three ticks of accumulated silence following
 * particular prior content, which is exactly the "self-initiated, timing
 * inferred from internal dynamics" property this experiment tests.
 *
 * TRAINING: supervised DFA (reusing xnn-utils.js unchanged), not TD/value
 * learning -- the scripted grammar gives an oracle-known correct target at
 * every tick, closer in kind to the delayed-recall task than to tic-tac-
 * toe's self-play. Two kinds of training events per exchange:
 *   - an INTERRUPT CORRECTION, applied immediately if the gate fires at any
 *     non-response tick (the only way "don't speak now" is actually taught,
 *     rather than assumed from a network that starts near-silent anyway)
 *   - the RESPONSE-TICK update, training gate + all 16 word outputs
 *     together in one supervised call, using the trace accumulated across
 *     the whole exchange since the last reset (matching the "credit must
 *     reach back across ticks" fix from the delayed-recall work -- training
 *     every tick with the default resetTrace would reproduce that exact bug).
 */

const fs = require("fs");
const path = require("path");
const { create, step, train } = require("./xnn-utils.js");

// ---------------------------------------------------------------------------
// Vocabulary and grammar
// ---------------------------------------------------------------------------

const VOCAB = [
  "hi", "bye", "yes", "no", "how", "are", "you", "i",
  "am", "good", "bad", "fine", "thanks", "help", "please", "sorry",
];
const WORD_TO_IDX = Object.fromEntries(VOCAB.map((w, i) => [w, i]));
const N_WORDS = VOCAB.length; // 16

// null response = silence is the grammatically correct reply
const GRAMMAR = [
  { trigger: ["hi"], responses: [["hi"]] },
  { trigger: ["bye"], responses: [["bye"]] },
  { trigger: ["how", "are", "you"], responses: [["good"], ["fine"]] },
  { trigger: ["help", "please"], responses: [["yes"]] },
  { trigger: ["thanks"], responses: null },
  { trigger: ["sorry"], responses: [["no"]] },
];

const N_HIDDEN = 32;
const N_OUTPUT_WORDS = N_WORDS; // 16
const N_OUTPUT = 1 + N_OUTPUT_WORDS; // gate + 16 words = 17
const N_TOTAL = N_WORDS + N_HIDDEN + N_OUTPUT;
const STEPS_PER_TICK = 2;
const GATE_THRESHOLD = 0.5;

// ---------------------------------------------------------------------------
// Deterministic PRNG
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
// Architecture: creation and self-describing derivation (same principle as
// demo-tictactoe.js -- every piece of metadata lives in factors, so the
// saved model is exactly xnn-utils.js's own { matrix, vector } shape)
// ---------------------------------------------------------------------------

function createBabblerModel(seed) {
  const roles = [
    ...Array(N_WORDS).fill("input"),
    ...Array(N_HIDDEN).fill("hidden"),
    ...Array(N_OUTPUT).fill("output"),
  ];

  const model = create(N_TOTAL, {
    seed,
    weightRange: [-1, 1],
    biasRange: [0, 1],
    targetActivity: 0.2, // slightly below default: most ticks, correct behavior is silence
    roles,
  });

  const rng = mulberry32(seed + 500);
  let inputSeen = 0;
  let outputSeen = 0;

  const vector = model.vector.map((n) => {
    if (n.factors.role === "input") {
      const wordIndex = inputSeen++;
      return { state: n.state, factors: { ...n.factors, wordIndex } };
    }
    if (n.factors.role === "hidden") {
      // one fixed DFA feedback weight per output neuron (17: gate + 16 words)
      const dfaWeights = Array.from({ length: N_OUTPUT }, () => rng() * 2 - 1);
      return { state: n.state, factors: { ...n.factors, dfaWeights } };
    }
    if (n.factors.role === "output") {
      const outputIndex = outputSeen++;
      if (outputIndex === 0) {
        return { state: n.state, factors: { ...n.factors, outputType: "gate" } };
      }
      return {
        state: n.state,
        factors: { ...n.factors, outputType: "word", wordIndex: outputIndex - 1 },
      };
    }
    return n;
  });

  return { matrix: model.matrix, vector };
}

function deriveArchitecture(model) {
  const inputIdx = new Array(N_WORDS);
  const wordOutputIdx = new Array(N_WORDS);
  let gateIdx = null;
  const hiddenIdx = [];
  const feedbackB = []; // hiddenIdx.length x N_OUTPUT

  model.vector.forEach((n, i) => {
    const f = n.factors;
    if (f.role === "input") inputIdx[f.wordIndex] = i;
    else if (f.role === "hidden") {
      hiddenIdx.push(i);
      feedbackB.push(f.dfaWeights);
    } else if (f.role === "output") {
      if (f.outputType === "gate") gateIdx = i;
      else wordOutputIdx[f.wordIndex] = i;
    }
  });

  return { inputIdx, wordOutputIdx, gateIdx, hiddenIdx, feedbackB };
}

// ---------------------------------------------------------------------------
// Model-state helpers not provided by xnn-utils.js
// ---------------------------------------------------------------------------

function resetTraceOnly(model) {
  return {
    matrix: model.matrix,
    vector: model.vector.map((n) => ({ state: n.state, factors: { ...n.factors, trace: 0 } })),
  };
}

function resetGameState(model) {
  return {
    matrix: model.matrix,
    vector: model.vector.map((n) => ({ state: 0, factors: { ...n.factors, trace: 0 } })),
  };
}

// ---------------------------------------------------------------------------
// Exchange generation
// ---------------------------------------------------------------------------

function zeroInput() {
  return new Array(N_WORDS).fill(0);
}

function oneHot(word) {
  const v = zeroInput();
  v[WORD_TO_IDX[word]] = 1;
  return v;
}

// Returns an array of tick descriptors:
//   { input: number[16], isResponseTick: bool, correctGate: 0|1, correctWordIdx: number|null }
function buildExchangeTicks(rule, rng) {
  const ticks = [];

  rule.trigger.forEach((word, i) => {
    ticks.push({ input: oneHot(word), isResponseTick: false, correctGate: 0, correctWordIdx: null });
    if (i < rule.trigger.length - 1) {
      ticks.push({ input: zeroInput(), isResponseTick: false, correctGate: 0, correctWordIdx: null });
    }
  });

  for (let i = 0; i < 3; i++) {
    ticks.push({ input: zeroInput(), isResponseTick: false, correctGate: 0, correctWordIdx: null });
  }

  const expectsResponse = rule.responses !== null;
  const correctWordIdx = expectsResponse
    ? WORD_TO_IDX[rule.responses[Math.floor(rng() * rule.responses.length)][0]]
    : null;
  ticks.push({
    input: zeroInput(),
    isResponseTick: true,
    correctGate: expectsResponse ? 1 : 0,
    correctWordIdx,
  });

  ticks.push({ input: zeroInput(), isResponseTick: false, correctGate: 0, correctWordIdx: null });

  return ticks;
}

// ---------------------------------------------------------------------------
// Training: one exchange
// ---------------------------------------------------------------------------

function outputTarget(correctGate, correctWordIdx) {
  // order matches arch: [gate, word0, word1, ..., word15]
  const target = new Array(N_OUTPUT).fill(0);
  target[0] = correctGate;
  if (correctWordIdx != null) target[1 + correctWordIdx] = 1;
  return target;
}

function trainExchange(model, rule, rng, arch, cw, cb) {
  const ticks = buildExchangeTicks(rule, rng);
  let m = resetTraceOnly(model);

  let interrupted = 0;

  for (const tick of ticks) {
    const inputMap = {};
    arch.inputIdx.forEach((idx, k) => {
      inputMap[idx] = tick.input[k];
    });
    for (let s = 0; s < STEPS_PER_TICK; s++) m = step(m, inputMap, {});

    const gateState = m.vector[arch.gateIdx].state;
    const gateFired = gateState > GATE_THRESHOLD;

    if (!tick.isResponseTick) {
      if (gateFired) {
        interrupted++;
        m = train(m, {
          mode: "supervised",
          outputIndices: [arch.gateIdx],
          target: [0],
          hiddenIndices: arch.hiddenIdx,
          feedbackMatrixB: arch.feedbackB.map((row) => [row[0]]), // gate column only
          cw,
          cb,
          resetTrace: true,
        });
      }
    } else {
      const outputIndices = [arch.gateIdx, ...arch.wordOutputIdx];
      const target = outputTarget(tick.correctGate, tick.correctWordIdx);
      m = train(m, {
        mode: "supervised",
        outputIndices,
        target,
        hiddenIndices: arch.hiddenIdx,
        feedbackMatrixB: arch.feedbackB, // full 17-wide projection
        cw,
        cb,
        resetTrace: true,
      });
    }
  }

  return { model: m, interrupted };
}

// ---------------------------------------------------------------------------
// Evaluation (greedy, no training)
// ---------------------------------------------------------------------------

function evaluate(model, numExchangesPerRule, seed, arch) {
  const rng = mulberry32(seed);
  let m = resetGameState(model);

  let totalTicks = 0, correctTicks = 0;
  let totalExchanges = 0, timingCorrectExchanges = 0;
  let respondedCorrectlyTimed = 0, wordCorrectGivenTiming = 0;

  for (const rule of GRAMMAR) {
    for (let e = 0; e < numExchangesPerRule; e++) {
      const ticks = buildExchangeTicks(rule, rng);
      let exchangeTimingOk = true;
      let sawFireAtResponse = false;
      let firedWordIdx = null;

      for (const tick of ticks) {
        const inputMap = {};
        arch.inputIdx.forEach((idx, k) => {
          inputMap[idx] = tick.input[k];
        });
        for (let s = 0; s < STEPS_PER_TICK; s++) m = step(m, inputMap, {});

        const gateState = m.vector[arch.gateIdx].state;
        const gateFired = gateState > GATE_THRESHOLD;
        const shouldFire = tick.correctGate === 1;

        totalTicks++;
        if (gateFired === shouldFire) correctTicks++;
        else exchangeTimingOk = false;

        if (tick.isResponseTick && gateFired) {
          sawFireAtResponse = true;
          let best = -Infinity, bestIdx = 0;
          arch.wordOutputIdx.forEach((idx, w) => {
            if (m.vector[idx].state > best) {
              best = m.vector[idx].state;
              bestIdx = w;
            }
          });
          firedWordIdx = bestIdx;
        }
      }

      totalExchanges++;
      if (exchangeTimingOk) {
        timingCorrectExchanges++;
        const expectsResponse = rule.responses !== null;
        if (expectsResponse && sawFireAtResponse) {
          respondedCorrectlyTimed++;
          const acceptable = rule.responses.map((r) => WORD_TO_IDX[r[0]]);
          if (acceptable.includes(firedWordIdx)) wordCorrectGivenTiming++;
        } else if (!expectsResponse) {
          respondedCorrectlyTimed++; // "correctly silent" counts toward the denominator too
          wordCorrectGivenTiming++; // vacuously correct -- no word was needed
        }
      }
    }
  }

  return {
    tickLevelAccuracy: correctTicks / totalTicks,
    exchangeTimingAccuracy: timingCorrectExchanges / totalExchanges,
    wordAccuracyGivenCorrectTiming:
      respondedCorrectlyTimed > 0 ? wordCorrectGivenTiming / respondedCorrectlyTimed : null,
  };
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
  const EXCHANGES = 12000;
  const EVAL_EVERY = 1000;
  const CW = 0.08;
  const CB = 0.08;

  const rng = mulberry32(SEED);
  let model = createBabblerModel(SEED);
  const arch = deriveArchitecture(model);

  console.log(`Training babbler: ${EXCHANGES} exchanges, ${N_TOTAL} neurons`);
  console.log(`(${arch.inputIdx.length} input, ${arch.hiddenIdx.length} DFA hidden, 1 gate + ${arch.wordOutputIdx.length} word outputs)`);
  console.log(`Grammar: ${GRAMMAR.length} rules, vocabulary: ${N_WORDS} words`);
  console.log("");

  console.log("Untrained baseline:");
  const baseline = evaluate(model, 20, 8000, arch);
  console.log(`  tick-level accuracy=${baseline.tickLevelAccuracy.toFixed(3)}  exchange timing accuracy=${baseline.exchangeTimingAccuracy.toFixed(3)}  word accuracy|correct timing=${baseline.wordAccuracyGivenCorrectTiming}`);
  console.log("");

  const history = [];
  let bestScore = -Infinity;
  let bestModel = model;
  let totalInterrupts = 0;

  for (let e = 0; e < EXCHANGES; e++) {
    const rule = GRAMMAR[Math.floor(rng() * GRAMMAR.length)];
    const result = trainExchange(model, rule, rng, arch, CW, CB);
    model = result.model;
    totalInterrupts += result.interrupted;

    if (e % EVAL_EVERY === 0 || e === EXCHANGES - 1) {
      const ev = evaluate(model, 15, 8000, arch);
      const score = ev.exchangeTimingAccuracy + (ev.wordAccuracyGivenCorrectTiming || 0);
      history.push({ exchange: e, ...ev, score });

      if (score > bestScore) {
        bestScore = score;
        bestModel = model;
      }

      console.log(
        `exchange ${String(e).padStart(6)}  tick_acc=${ev.tickLevelAccuracy.toFixed(3)}  ` +
        `timing_acc=${ev.exchangeTimingAccuracy.toFixed(3)}  ` +
        `word_acc|timing=${ev.wordAccuracyGivenCorrectTiming != null ? ev.wordAccuracyGivenCorrectTiming.toFixed(3) : "n/a"}  ` +
        `score=${score.toFixed(3)}${score === bestScore ? "  <- best" : ""}`
      );
    }
  }

  console.log("");
  console.log(`Total interrupt-correction events across training: ${totalInterrupts}`);
  console.log("");
  console.log("Final evaluation of best checkpoint (40 exchanges per rule):");
  const finalEval = evaluate(bestModel, 40, 42, arch);
  console.log(`  tick-level accuracy: ${finalEval.tickLevelAccuracy.toFixed(3)}`);
  console.log(`  exchange timing accuracy: ${finalEval.exchangeTimingAccuracy.toFixed(3)}`);
  console.log(`  word accuracy | correct timing: ${finalEval.wordAccuracyGivenCorrectTiming}`);
  console.log("");
  console.log("Baseline (untrained) for comparison:");
  console.log(`  tick-level accuracy: ${baseline.tickLevelAccuracy.toFixed(3)}`);
  console.log(`  exchange timing accuracy: ${baseline.exchangeTimingAccuracy.toFixed(3)}`);

  const modelPath = writeJson(path.join(__dirname, "babbler-model.json"), bestModel);
  console.log(`\nTrained model written to: ${modelPath}`);

  const metricsOut = {
    seed: SEED,
    exchanges: EXCHANGES,
    vocabulary: VOCAB,
    grammar: GRAMMAR,
    untrainedBaseline: baseline,
    bestCheckpointScore: bestScore,
    finalEvaluation: finalEval,
    totalInterruptCorrections: totalInterrupts,
    trainingHistory: history,
  };
  const metricsPath = writeJson(path.join(__dirname, "babbler-metrics.json"), metricsOut);
  console.log(`Test metrics written to: ${metricsPath}`);
}

main();

export const meta = {
  name: 'verify-75-merge-readiness',
  description: 'Two independent checks of #75 head 308d68b (findings re-run + adversarial hunt on the new near-rule) to decide merge',
  phases: [{ title: 'Verify', detail: 'findings re-run; adversarial hunt' }],
}
const REPO = '/home/user/stt-api'
const SCRATCH = '/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad'
const PY = SCRATCH + '/venv/bin/python'
const BR = 'claude/awesome-archimedes-xtcwcp'
const OUT = {
  type: 'object',
  properties: {
    head: { type: 'string' },
    tests: { type: 'string' },
    results: { type: 'string', description: 'case-by-case results with exact ranges (main vs prev head e8811de vs new head)' },
    blockers: { type: 'array', items: { type: 'object', properties: { severity: { type: 'string', enum: ['High', 'Medium', 'Low', 'Nit'] }, title: { type: 'string' }, file: { type: 'string' }, line: { type: 'integer' }, detail: { type: 'string', description: 'concrete repro input -> actual vs expected, whether main does the same (regression or not), suggested fix' }, verified: { type: 'boolean' } }, required: ['severity', 'title', 'file', 'line', 'detail', 'verified'] } },
    verdict: { type: 'string', enum: ['merge', 'merge_with_followups', 'do_not_merge'] },
    why: { type: 'string' },
  },
  required: ['head', 'tests', 'results', 'blockers', 'verdict', 'why'],
}
const setup = tag => `Setup: \`git -C ${REPO} fetch -q origin main ${BR}:refs/remotes/origin/${BR}\`. PR scagood/stt-api#75 head should be 308d68b (one commit on top of e8811de: "half a second per sound, and short words near one, not summed"); confirm with ls-remote, report the full sha; if it moved, use the newest and say so. origin/main is 308d7d6 (#71-#73 merged). Read-only: no push, no GitHub writes, don't modify ${REPO}. Worktrees under ${SCRATCH}/${tag}-* (remove at end), scripts under ${SCRATCH}/${tag}-scripts/. Venv ${PY}. Earlier repro scripts: ${SCRATCH}/v75-scripts/ (repro.py, f2_and_retime.py, manyclicks.py, yesbreath.py, measure.py, equiv.py) and ${SCRATCH}/rr75-scripts/. Repro audio: 16 kHz float32, Gaussian room tone -55 dBFS, two 40 s Gaussian turns at -20 dBFS, volume VAD (VAD_GATE_DB unset), chunker.plan_chunks(target_sec=60, max_sec=75, context_sec as given); retime.pauses for retime. The change: in chunker.loud_frames, a re-heard quiet run now keeps heard frames only in sounds (heard runs joined across dips < VAD_MIN_SILENCE_MS = 20 frames) of >= 0.5 s (_HEARD_ENOUGH = 25 frames), plus shorter heard frames within \`relisten\` frames of such a sound ('near'); _volume_speech_segments now uses a shared _joined helper.`

const a = () => agent(`Check 1 of 2 — re-run every known case on #75 and judge merge readiness.
${setup('m75a')}
Run on main, e8811de and the new head:
1. Full test suite on the head, and merged with origin/main.
2. Multi-breath (the open Medium): (a) 15 s pause, three 250 ms breaths at -42 dBFS at 43/48/53 s, ctx 0 — main (0,40.12),(54.88,95.0); (b) 30 s pause, 200 ms breath at -45 dBFS every 4 s from 42 s, ctx 5 — main (0,40.12),(69.88,110.0); (c) 10 s pause, two 300 ms breaths at -42 dBFS at 42.6/47.0, 43/46, 42.5/47.5 — main (0,40.12),(49.88,90.0); (d) 350 ms reply + one 300 ms breath in a 10 s pause. Also retime.pauses on (a): main (40,55).
3. Earlier fixes: 10 ms click at -30 dBFS; 30 ms knock at -35 dBFS; single 300 ms breath at -42 dBFS in 5 s and 10 s pauses; thirty 10 ms clicks 0.27 s apart; retime crossing words (chunk 0-30, gap 12-18, Then 10.0-11.8, said 13.0-13.4, him. 14.0-14.4, Next 18.5-19.0 -> left alone).
4. Quiet speaker kept: the PR's quiet-turn tests (test_volume_keeps_a_quieter_speakers_turn -34/-44, retime test_a_quieter_speaker_is_no_pause); quiet replies at -40 dBFS of 350/450/550/700 ms and a 2 s syllabic quiet turn; and a quiet speaker with short words between phrases (e.g. phrases 0.6-1.2 s with 0.12-0.32 s words in the gaps, 26 dB down) — are the short words kept and no long stretch cut out mid-turn?
5. _volume_speech_segments refactor: equivalence vs e8811de on 300 random multi-level wavs and with a fixed gate (equiv.py), incl. empty/silent input.
6. Test pinning (earlier nit): do the tests now catch mean-vs-median and a changed _HEARD_ENOUGH? Is the click-level comment corrected?
Verdict: 'merge' if the Medium is fixed, earlier fixes hold, tests/merge clean, and no Medium+ issue; 'merge_with_followups' if only Low/Nit remain; else 'do_not_merge'.`, { label: 'verify:#75 cases', phase: 'Verify', schema: OUT })

const b = () => agent(`Check 2 of 2 — adversarial hunt for NEW failure modes introduced by #75's latest commit. Try hard to break it; report only what you reproduce.
${setup('m75b')}
Focus on the new 'near' rule and per-sound minimum in chunker.loud_frames (used by the splitter with relisten = CHUNK_TRIM_SILENCE_SEC frames = 150, and by retime.pauses at 0.6x with relisten 150):
- Breaths or rustles within relisten (3 s) of a genuine >= 0.5 s quiet sound: do they now become speech, and does that stop a long pause from being cut out or create standalone breath chunks, where main or e8811de didn't? E.g. a quiet 'Yes.' of 600 ms followed by breaths every 2.5 s through a 20 s pause; a loud speaker, long pause holding one quiet 0.6 s cough plus breaths around it.
- Chaining: can 'near' frames plus re-listening of remaining quiet runs cascade so a whole long pause turns loud?
- Joining across dips < 400 ms: can several clicks/breaths spaced < 400 ms apart join into one 'sound' >= 0.5 s of heard frames (e.g. typing, a ticking clock, rain), turning a pause into speech? Compare main.
- Edge cases: relisten larger than the run, runs at the very start/end of the file, empty heard, all heard, very short files, NaN-free, dtype, int overflow on indices, near[] slicing with negative start (max(0, a - relisten) used?) — look for off-by-one.
- retime.pauses consequences: pauses split by kept near-frames changing word retiming in a realistic two-speaker clip.
- Performance: loud_frames on 2 h of audio vs e8811de and main (the Python loop over joined spans).
For each issue: severity, whether main/e8811de do the same (regression or not), exact repro, line on the head inside the PR diff vs origin/main, suggested fix. Then give a merge verdict as defined: 'merge' if no Medium+ new issue; 'merge_with_followups' if only Low/Nit; else 'do_not_merge'.`, { label: 'verify:#75 adversarial', phase: 'Verify', schema: OUT })

const [r1, r2] = await parallel([a, b])
return { cases: r1, adversarial: r2 }
